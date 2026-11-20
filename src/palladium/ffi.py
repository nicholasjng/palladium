"""`metal_call`: a Pallas kernel dispatched to Metal from the CPU platform.

The kernel is registered as a jax.ffi target backed by the runtime's C API
(`native/ffi/`, `native/runtime/`), so it composes with jax.jit and jax.vmap
and needs no PJRT plugin.
"""

from __future__ import annotations

import functools
import hashlib
import math
import threading
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.custom_batching
import numpy as np

from palladium import _runtime
from palladium.diagnostics import KernelDiagnostics, explain_spec, log_compile
from palladium.emit import emit_msl, emit_msl_stats
from palladium.emit.core import CTYPES
from palladium.errors import DispatchError
from palladium.launch import (
    CompilerParams,
    check_threadgroup,
    launch_geometry,
    lowering_options,
)
from palladium.trace import KernelSpec, trace

__all__ = ["FfiCallable", "metal_call"]


_TARGET_NAME = "palladium_dispatch"
_REGISTER_LOCK = threading.Lock()
_registered = False


def _register() -> None:
    """Load the native handler and register it, once per process."""
    global _registered
    if _registered:
        return
    with _REGISTER_LOCK:
        if _registered:
            return
        capsule = jax.ffi.pycapsule(_runtime.library().palladium_dispatch)
        jax.ffi.register_ffi_target(_TARGET_NAME, capsule, platform="cpu")
        _registered = True


def check_dtypes(args: Sequence[Any]) -> None:
    """Raise DispatchError for an input whose dtype has no Metal type.

    `args` are arrays or shape-and-dtype examples. This checks inputs only;
    individual primitives may support fewer dtypes.
    """
    for i, a in enumerate(args):
        dtype = np.dtype(a.dtype)
        if dtype not in CTYPES:
            hint = (
                "; float64 usually means jax_enable_x64 is on, disable it or cast to float32"
                if dtype == np.float64
                else ""
            )
            supported = ", ".join(d.name for d in CTYPES)
            raise DispatchError(
                f"argument {i} has dtype {dtype.name}, which palladium cannot "
                f"lower (supported: {supported}){hint}"
            )


# Traced shapes kept per callable.
_CACHE_SIZE = 256


class FfiCallable:
    """Specialize a Pallas kernel by input shape and dispatch it through CPU FFI.

    Parameters
    ----------
    staged : callable
        The Pallas call to trace: a ``pl.pallas_call`` or ``pl.kernel`` callable.
    interpret : callable
        The same call with ``interpret=True``.
    params : CompilerParams or None
        Palladium compiler options; None uses the defaults. Prefer ``metal_call``
        or ``plmtl.kernel`` as the constructor.

    Attributes
    ----------
    interpret : callable
        The same Pallas call with ``interpret=True``; provides an interpreter reference
        without native Metal dispatch.

    Raises
    ------
    TypeError
        The compiler options are not ``CompilerParams``.

    Notes
    -----
    A bounded per-instance LRU caches specifications and MSL for up to 256 input shape/dtype
    signatures. Native pipeline compilation has a separate cache. The CPU FFI call has no
    automatic JVP or transpose rule; pair it with a backward callable through ``with_vjp``.
    """

    def __init__(self, staged: Callable, interpret: Callable, params: CompilerParams | None):
        if params is None:
            params = CompilerParams()
        elif not isinstance(params, CompilerParams):
            raise TypeError(
                f"metal_call takes palladium.CompilerParams, not {type(params).__name__}"
            )
        self._params = params
        self._staged = staged
        self.interpret = interpret
        self._specialize = functools.lru_cache(maxsize=_CACHE_SIZE)(self._trace_and_emit)
        # lru_cache alone lets concurrent first calls each trace and emit.
        self._lock = threading.Lock()
        self._pipelined = self._build_pipelined()

    def _spec_and_msl(self, args: Sequence[Any]) -> tuple[KernelSpec, str, dict[str, Any]]:
        """The spec, MSL and launch attributes for arguments of these shapes and
        dtypes, traced and emitted once per signature."""
        signature = tuple((tuple(a.shape), np.dtype(a.dtype)) for a in args)
        with self._lock:
            return self._specialize(signature)

    def _trace_and_emit(
        self, signature: tuple[tuple[tuple[int, ...], np.dtype], ...]
    ) -> tuple[KernelSpec, str, dict[str, Any]]:
        params = self._params
        shapes = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in signature]
        check_dtypes(shapes)
        spec = trace(self._staged, *shapes)
        check_threadgroup(params.threadgroup_dims)
        msl, stats = emit_msl_stats(spec, dot_general=params.dot_general, program=params.program)
        log_compile(spec, msl, stats, params.threadgroup_dims)
        return spec, msl, launch_attributes(spec, msl, params)

    def explain(self, *args) -> KernelDiagnostics:
        """Report launch geometry, declared storage, and source size without Metal compilation.

        Parameters
        ----------
        *args : array-like or jax.ShapeDtypeStruct
            Input shape and dtype examples. Array contents are not read.

        Returns
        -------
        KernelDiagnostics
            Geometry and storage estimates for the selected emitter and launch options.

        Raises
        ------
        DispatchError
            An input dtype is unsupported.
        TraceError
            Tracing rejects the Pallas call.
        EmitError
            The kernel uses unsupported shapes, addressing, effects, or primitive
            parameters.
        UnsupportedPrimitiveError
            A staged primitive has no registered lowering rule.
        ValueError
            The explicit group is incompatible with cooperative emission.

        Notes
        -----
        This retraces and emits rather than using the dispatch cache. Device queries may supply
        limits; without a device, limits remain unknown. Declared storage is an estimate, not a
        Metal compiler resource measurement.
        """
        check_dtypes(args)
        return explain_spec(
            trace(self._staged, *(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args)),
            self._params.threadgroup_dims,
            dot_general=self._params.dot_general,
            program=self._params.program,
        )

    def __call__(self, *args):
        """Dispatch the specialized kernel and return its JAX outputs.

        Parameters
        ----------
        *args : array-like
            Input arrays with supported dtypes and shapes matching the Pallas layout.

        Returns
        -------
        jax.Array or tuple of jax.Array
            One array for a single output, otherwise a tuple in declared output order.

        Raises
        ------
        DispatchError
            An input dtype cannot be lowered.
        FileNotFoundError
            The native FFI library is missing and no valid override is configured.
        TraceError
            The call has unsupported structure or unsafe aliases or writes.
        EmitError
            The kernel uses unsupported shapes, addressing, effects, or primitive
            parameters.
        UnsupportedPrimitiveError
            A staged primitive has no registered lowering rule.

        Notes
        -----
        The first use registers the native handler and specializes the kernel. Under ``jax.jit``
        dispatch executes as part of the compiled CPU program; compilation and device failures
        surface through the runtime.
        """
        _register()
        return self._pipelined(*args)

    def _build_pipelined(self):
        """Under jax.vmap, one FFI call over the whole batch, looped by the
        native handler; unvmapped calls dispatch once. jax.ffi's whole-batch
        methods would dispatch the per-shape grid over batched buffers."""

        @jax.custom_batching.custom_vmap
        def pipelined(*args):
            return self._ffi_dispatch(args, vmap_method=None)

        @pipelined.def_vmap
        def _pipelined_vmap_rule(axis_size, in_batched, *args):
            # 'sequential', not None: this rule takes one vmap level, an
            # enclosing one batches the ffi_call itself.
            out = self._ffi_dispatch(
                args,
                vmap_method="sequential",
                in_batched=in_batched,
                axis_size=axis_size,
            )
            return out, jax.tree.map(lambda _: True, out)

        return pipelined

    def _ffi_dispatch(
        self,
        args: tuple[Any, ...],
        *,
        vmap_method: str | None,
        in_batched: list[bool] | None = None,
        axis_size: int | None = None,
    ):
        """Build and invoke the ffi_call. With `axis_size`/`in_batched`
        (batched args carry the batch at axis 0, unbatched ones use stride
        0), the native handler loops over the batch."""
        batched = list(in_batched) if in_batched is not None else [False] * len(args)
        unbatched = [
            jax.ShapeDtypeStruct(a.shape[1:] if b else a.shape, a.dtype)
            for a, b in zip(args, batched, strict=True)
        ]
        spec, _, attributes = self._spec_and_msl(unbatched)
        if axis_size is not None:
            # An aliased input shares its buffer with a batched output, so an
            # unbatched one needs a copy per batch element.
            operands = list(args)
            for i, _ in spec.aliases:
                if not batched[i]:
                    operands[i] = jax.numpy.broadcast_to(args[i], (axis_size, *args[i].shape))
                    batched[i] = True
            args = tuple(operands)
        in_strides = [
            np.dtype(u.dtype).itemsize * math.prod(u.shape) if b else 0
            for u, b in zip(unbatched, batched, strict=True)
        ]
        out_strides = [
            np.dtype(info.dtype).itemsize * math.prod(info.array_shape) for info in spec.outputs
        ]
        lead = () if axis_size is None else (int(axis_size),)
        out_structs = [
            jax.ShapeDtypeStruct(lead + tuple(info.array_shape), info.dtype)
            for info in spec.outputs
        ]
        result_shape_dtypes = out_structs[0] if len(out_structs) == 1 else out_structs
        outs = jax.ffi.ffi_call(
            _TARGET_NAME,
            result_shape_dtypes=result_shape_dtypes,
            vmap_method=vmap_method,
            # XLA may donate the input buffer for aliased pairs.
            input_output_aliases=dict(spec.aliases) or None,
        )(
            *args,
            **attributes,
            batch_size=1 if axis_size is None else int(axis_size),
            elem_strides=np.asarray(in_strides + out_strides, dtype=np.int64),
        )
        # ffi_call returns a list for several outputs; pallas_call a tuple.
        return tuple(outs) if len(out_structs) > 1 else outs


def launch_attributes(spec: KernelSpec, msl_source: str, params: CompilerParams) -> dict[str, Any]:
    """The `palladium_dispatch` attributes describing one launch of `msl_source`.

    The CPU FFI handler and the `metal` PJRT compiler read the same attributes,
    plus `batch_size` and `elem_strides`, which the caller adds.
    """
    grid, threadgroup = launch_geometry(spec, msl_source, params.threadgroup_dims)
    return {
        # Native caches key on the digest instead of the source itself.
        "source_id": hashlib.sha256(msl_source.encode()).hexdigest(),
        "msl_source": msl_source,
        "function_name": spec.name,
        **{f"grid_{axis}": size for axis, size in zip("xyz", grid)},
        # A (0, 0, 0) threadgroup lets the runtime choose.
        **{f"threadgroup_{axis}": size for axis, size in zip("xyz", threadgroup or (0, 0, 0))},
        "math_mode": _runtime.MATH_MODES[params.math_mode],
    }


def pallas_lowering(ctx, *in_nodes, interpret: Any = False, **params):
    """Lower one `pallas_call` for the `metal` PJRT platform.

    Emits the same `palladium_dispatch` custom call as `metal_call`; the
    plugin's compiler turns it into a launch inside the program's command
    buffers. With `interpret=True`, JAX's own rule lowers the interpreter.
    """
    if interpret:
        from jax._src.interpreters import mlir  # JAX's default rule has no public name
        from jax.experimental import pallas as pl

        default = mlir._lowerings[pl.pallas_call_p].rule
        return default(ctx, *in_nodes, interpret=interpret, **params)
    from palladium.trace import spec_from_params

    options = lowering_options(params, "metal")
    spec = spec_from_params(params)
    check_dtypes(ctx.avals_in)
    check_threadgroup(options.threadgroup_dims)
    msl_source = emit_msl(spec, dot_general=options.dot_general, program=options.program)
    rule = jax.ffi.ffi_lowering(
        _TARGET_NAME,
        operand_output_aliases=dict(spec.aliases) or None,
    )
    return rule(
        ctx,
        *in_nodes,
        **launch_attributes(spec, msl_source, options),
        batch_size=1,
        elem_strides=np.zeros(len(ctx.avals_in) + len(ctx.avals_out), dtype=np.int64),
    )


def metal_call(kernel: Callable, **pallas_kwargs) -> FfiCallable:
    """Create a Pallas callable that dispatches Metal kernels from the JAX CPU backend.

    Parameters
    ----------
    kernel : callable
        Pallas kernel operating on Refs, with outputs written in place.
    **pallas_kwargs : dict
        Arguments for ``pl.pallas_call``, including required ``out_shape``, optional
        grid, BlockSpecs, scratch, aliases, and ``compiler_params=CompilerParams(...)``.
        Do not pass ``interpret``; use the returned ``interpret`` callable.

    Returns
    -------
    FfiCallable
        Callable accepting NumPy or JAX arrays and returning a JAX array, or a tuple for
        multiple outputs. Supports ``jax.jit`` and ``jax.vmap``.

    Raises
    ------
    TypeError
        ``compiler_params`` is not a Palladium ``CompilerParams`` instance.

    Notes
    -----
    Tracing, dtype validation, and Metal compilation are deferred until use. This path
    requires the native FFI library and a Metal device, but not the PJRT plugin. One
    vmap level batches dispatches in one native call; outer levels run sequentially.
    Gradients require an explicit backward implementation via ``with_vjp``. FAST math is the
    default; use SAFE for compensated arithmetic and strict NaN or signed-zero behavior.

    See Also
    --------
    FfiCallable.explain : Inspect launch geometry and storage before dispatch.
    with_vjp : Attach an explicit reverse-mode derivative.
    """
    import jax.experimental.pallas as pl

    return FfiCallable(
        pl.pallas_call(kernel, **pallas_kwargs),
        pl.pallas_call(kernel, **pallas_kwargs, interpret=True),
        pallas_kwargs.get("compiler_params"),
    )
