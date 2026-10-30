"""`metal_call`: a Pallas kernel dispatched to Metal from the CPU platform.

The kernel is registered as a jax.ffi target backed by the runtime's C API
(`native/ffi/`, `native/runtime/`), so it composes with jax.jit and jax.vmap
and needs no PJRT plugin.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from palladium import _runtime
from palladium.diagnostics import KernelDiagnostics, explain_spec, log_compile
from palladium.emit import emit_msl
from palladium.emit.core import CTYPES
from palladium.errors import DispatchError
from palladium.launch import CompilerParams, check_threadgroup, launch_geometry
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


def check_dtypes(args: tuple) -> None:
    """Reject unsupported input dtypes before they reach MSL emission.

    Parameters
    ----------
    args : tuple of array-like
        Objects exposing a ``dtype`` attribute, including shape-only examples.

    Raises
    ------
    DispatchError
        An argument is not float32, float16, bfloat16, int32, uint32, or bool.

    Notes
    -----
    An empty argument tuple passes. This checks inputs only; individual primitives may
    support fewer dtypes.
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
    kernel : callable
        Pallas kernel operating on input, output, and optional scratch Refs.
    pallas_kwargs : dict of str to object
        Keywords for ``pl.pallas_call``, excluding ``interpret``. Prefer ``metal_call``
        as the constructor.

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

    def __init__(self, kernel: Callable, pallas_kwargs: dict[str, Any]):
        import jax.experimental.pallas as pl

        params = pallas_kwargs.get("compiler_params")
        if params is None:
            params = CompilerParams()
        elif not isinstance(params, CompilerParams):
            raise TypeError(
                f"metal_call takes palladium.CompilerParams, not {type(params).__name__}"
            )
        self._params = params
        self._staged = pl.pallas_call(kernel, **pallas_kwargs)
        self.interpret = pl.pallas_call(kernel, **pallas_kwargs, interpret=True)
        # Bounded LRU of (spec, msl, digest) per input shape signature.
        self._cache: OrderedDict[tuple, tuple[KernelSpec, str, str]] = OrderedDict()
        # Serialize cache misses: concurrent first calls compile once.
        self._lock = threading.Lock()
        self._pipelined = self._build_pipelined()

    @staticmethod
    def _shapes(args) -> list[jax.ShapeDtypeStruct]:
        return [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args]

    def _spec_and_msl(self, args: tuple[Any, ...]) -> tuple[KernelSpec, str, str]:
        """Trace and emit for these argument shapes, cached per signature.

        Returns the spec, the MSL text, and its SHA-256 digest, which native
        caches key on instead of the source itself.
        """
        key = tuple((a.shape, np.dtype(a.dtype).str) for a in args)
        # Lookup and LRU promotion under one lock: a concurrent miss can
        # evict this entry between the two.
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._cache.move_to_end(key)
        if entry is None:
            with self._lock:
                entry = self._cache.get(key)
                if entry is None:
                    check_dtypes(tuple(args))
                    spec = trace(self._staged, *self._shapes(args))
                    check_threadgroup(spec, self._params.threadgroup_dims)
                    log_compile(
                        spec,
                        self._params.threadgroup_dims,
                        self._params.dot_general,
                        self._params.program,
                    )
                    msl = emit_msl(
                        spec,
                        dot_general=self._params.dot_general,
                        program=self._params.program,
                    )
                    entry = (spec, msl, hashlib.sha256(msl.encode()).hexdigest())
                    self._cache[key] = entry
                    if len(self._cache) > _CACHE_SIZE:
                        self._cache.popitem(last=False)
        return entry

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
            trace(self._staged, *self._shapes(args)),
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
        import jax.custom_batching

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
        batched = in_batched if in_batched is not None else [False] * len(args)
        unbatched = [
            jax.ShapeDtypeStruct(a.shape[1:] if b else a.shape, a.dtype)
            for a, b in zip(args, batched, strict=True)
        ]
        spec, msl_source, source_id = self._spec_and_msl(tuple(unbatched))
        grid, threadgroup = launch_geometry(spec, msl_source, self._params.threadgroup_dims)
        # MRLaunchDesc takes 3 grid dims; a (0, 0, 0) threadgroup lets the
        # runtime choose.
        threadgroup = threadgroup or (0, 0, 0)
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
            source_id=source_id,
            msl_source=msl_source,
            function_name=spec.name,
            grid_x=int(grid[0]),
            grid_y=int(grid[1]),
            grid_z=int(grid[2]),
            threadgroup_x=int(threadgroup[0]),
            threadgroup_y=int(threadgroup[1]),
            threadgroup_z=int(threadgroup[2]),
            math_mode=_runtime.MATH_MODES[self._params.math_mode],
            batch_size=1 if axis_size is None else int(axis_size),
            elem_strides=np.asarray(in_strides + out_strides, dtype=np.int64),
        )
        # ffi_call returns a list for several outputs; pallas_call a tuple.
        return tuple(outs) if len(out_structs) > 1 else outs


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
    requires the native FFI library and a Metal device, but does not require jax-mps. One
    vmap level batches dispatches in one native call; outer levels run sequentially.
    Gradients require an explicit backward implementation via ``with_vjp``. FAST math is the
    default; use SAFE for compensated arithmetic and strict NaN or signed-zero behavior.

    See Also
    --------
    FfiCallable.explain : Inspect launch geometry and storage before dispatch.
    with_vjp : Attach an explicit reverse-mode derivative.
    """
    return FfiCallable(kernel, pallas_kwargs)
