"""`metal_call`: a Pallas kernel dispatched to Metal from the CPU platform.

The kernel is registered as a jax.ffi target backed by metal-runtime's C
API (`native/ffi/`), so it composes with jax.jit and jax.vmap and needs no
PJRT plugin. `PALLADIUM_FFI_LIBRARY` overrides the handler path for an
out-of-tree build.
"""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import importlib.resources
import math
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np

from palladium.diagnostics import (
    KernelDiagnostics,
    check_threadgroup,
    explain_spec,
    log_compile,
    normalize_threadgroup,
    simdgroup_width,
)
from palladium.emit import emit_msl
from palladium.emit.core import CTYPES, DOT_GENERAL_POLICIES
from palladium.errors import DispatchError
from palladium.trace import KernelSpec, trace

__all__ = ["FfiCallable", "metal_call"]


_TARGET_NAME = "palladium_dispatch"
_LIBRARY_NAME = "libpalladium_ffi.dylib"

# MRMathMode ordinals from metal-runtime's c_api.h; metal_runtime.MathMode
# is a StrEnum.
_MATH_MODE_ORDINALS = {"safe": 0, "relaxed": 1, "fast": 2}


def _library_path() -> Path:
    override = os.environ.get("PALLADIUM_FFI_LIBRARY")
    if override:
        return Path(override)
    # An editable install spans two roots (source tree and CMake build
    # dir); joinpath resolves to whichever root has the file.
    candidate = importlib.resources.files("palladium").joinpath(_LIBRARY_NAME)
    if not candidate.is_file():
        raise FileNotFoundError(
            f"palladium's jax.ffi handler ({_LIBRARY_NAME}) isn't built or "
            "installed. `uv sync`/`pip install` builds it automatically via "
            "CMakeLists.txt; if this is a from-source checkout, re-run that. "
            "Otherwise set PALLADIUM_FFI_LIBRARY to point at an existing build."
        )
    return Path(str(candidate))


_REGISTER_LOCK = threading.Lock()
_registered = False


def _register() -> None:
    """Load the native handler and register it, once per process. Locked
    so concurrent first calls do not both run the registration body."""
    global _registered
    if _registered:
        return
    with _REGISTER_LOCK:
        if _registered:
            return
        handle = ctypes.CDLL(str(_library_path()))
        capsule = jax.ffi.pycapsule(handle.palladium_dispatch)
        jax.ffi.register_ffi_target(_TARGET_NAME, capsule, platform="cpu")
        _registered = True


@dataclasses.dataclass(frozen=True)
class CallOptions:
    """The Metal-side keywords every call path accepts, split off the
    `pl.pallas_call` keywords once at construction."""

    math_mode: Any
    threadgroup: tuple[int, ...] | None
    cache_size: int
    dot_general: str
    vmap_method: str | None

    @classmethod
    def split(
        cls,
        pallas_kwargs: dict[str, Any],
        *,
        vmap_method: str | None = None,
    ) -> CallOptions:
        """Pop the Metal-side keywords out of `pallas_kwargs` (mutated)."""
        from metal_runtime import MathMode

        dot_general = pallas_kwargs.pop("dot_general", "auto")
        if dot_general not in DOT_GENERAL_POLICIES:
            raise ValueError("dot_general must be 'auto', 'default', or 'tensorops'")
        return cls(
            math_mode=pallas_kwargs.pop("math_mode", MathMode.FAST),
            threadgroup=normalize_threadgroup(pallas_kwargs.pop("threadgroup", None)),
            cache_size=pallas_kwargs.pop("cache_size", 256),
            dot_general=dot_general,
            vmap_method=pallas_kwargs.pop("vmap_method", vmap_method),
        )


def check_dtypes(args: tuple) -> None:
    """Reject unsupported dtypes before tracing; they otherwise surface
    as a KeyError inside emit."""
    for i, a in enumerate(args):
        dtype = getattr(a, "dtype", None)
        name = np.dtype(dtype if dtype is not None else np.asarray(a).dtype).name
        if name not in CTYPES:
            hint = (
                "; float64 usually means jax_enable_x64 is on, disable it or cast to float32"
                if name == "float64"
                else ""
            )
            raise DispatchError(
                f"argument {i} has dtype {name}, which palladium cannot "
                f"lower (supported: {', '.join(CTYPES)}){hint}"
            )


# Whole-batch methods conflict with the shape-specialized grid. pipelined
# batches in one FFI call; nested vmap levels run sequentially.
_SAFE_VMAP_METHODS = (None, "sequential", "sequential_unrolled", "pipelined")


class FfiCallable:
    """A palladium kernel registered as a jax.ffi target, dispatched inside
    XLA's execution so it composes under `jax.jit`. Tracing and MSL emission
    are cached per input shape/dtype. Not differentiable by itself (`ffi_call`
    has no JVP/transpose rule); pair with a backward kernel via `with_vjp`.

    Attributes
    ----------
    interpret : callable
        The same pallas_call with `interpret=True`: the CPU oracle.
    """

    def __init__(self, kernel: Callable, pallas_kwargs: dict[str, Any], options: CallOptions):
        if options.vmap_method not in _SAFE_VMAP_METHODS:
            raise ValueError(
                f"vmap_method {options.vmap_method!r} is not supported: the launch "
                "grid is baked per unbatched shape, so whole-batch methods "
                "would dispatch it over batched buffers. Use 'pipelined' "
                "(the default: one FFI call, the native handler loops the "
                "batch), 'sequential' or 'sequential_unrolled' (one "
                "dispatch per batch element), or put the batch dimension "
                "in the Pallas grid instead."
            )
        import jax.experimental.pallas as pl

        self._staged = pl.pallas_call(kernel, **pallas_kwargs)
        self.interpret = pl.pallas_call(kernel, **pallas_kwargs, interpret=True)
        self._options = options
        # Bounded LRU of (spec, msl, digest) per input shape signature.
        self._cache: OrderedDict[tuple, tuple[KernelSpec, str, str]] = OrderedDict()
        # Serialize cache misses: concurrent first calls compile once.
        self._lock = threading.Lock()
        self._math_mode_ordinal = _MATH_MODE_ORDINALS[options.math_mode]
        self._vmap_method = options.vmap_method
        # None is wrapped too, so its batching error names Palladium's options.
        self._pipelined = (
            self._build_pipelined() if options.vmap_method in ("pipelined", None) else None
        )

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
                    check_threadgroup(spec, self._options.threadgroup)
                    log_compile(spec, self._options.threadgroup, self._options.dot_general)
                    msl = emit_msl(spec, dot_general=self._options.dot_general)
                    entry = (spec, msl, hashlib.sha256(msl.encode()).hexdigest())
                    self._cache[key] = entry
                    size = self._options.cache_size
                    while size and len(self._cache) > size:
                        self._cache.popitem(last=False)
        return entry

    def explain(self, *args) -> KernelDiagnostics:
        """Report launch geometry and emitted MSL size for these inputs.
        Emits MSL; compiles and dispatches nothing.

        Parameters
        ----------
        *args
            Arrays or `jax.ShapeDtypeStruct`s fixing input shapes; no
            data is read.
        """
        check_dtypes(args)
        return explain_spec(
            trace(self._staged, *self._shapes(args)),
            self._options.threadgroup,
            dot_general=self._options.dot_general,
        )

    def __call__(self, *args):
        """Dispatch via jax.ffi; traceable and jittable."""
        _register()
        if self._pipelined is not None:
            return self._pipelined(*args)
        return self._ffi_dispatch(args, vmap_method=self._vmap_method)

    def _build_pipelined(self):
        """The 'pipelined' batching rule: under jax.vmap, one FFI call over
        the whole batch; unvmapped calls dispatch once. With
        `vmap_method=None` the rule rejects the batch."""
        import jax.custom_batching

        batching_disabled = self._vmap_method is None

        @jax.custom_batching.custom_vmap
        def pipelined(*args):
            return self._ffi_dispatch(args, vmap_method=None)

        @pipelined.def_vmap
        def _pipelined_vmap_rule(axis_size, in_batched, *args):
            if batching_disabled:
                raise ValueError(
                    "jax.vmap over this kernel needs a vmap_method, and this "
                    "one was built with vmap_method=None. Pass "
                    "metal_call(..., vmap_method='pipelined') for one FFI "
                    "call over the whole batch, or 'sequential' for one "
                    "dispatch per element. jax.ffi's own whole-batch methods "
                    "(expand_dims, broadcast_all) are not usable here: the "
                    "launch grid is baked per unbatched shape."
                )
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
        # MRLaunchDesc takes 3 grid dims; (0, 0, 0) threadgroup lets the
        # runtime choose.
        grid = (tuple(spec.grid) + (1, 1, 1))[:3]
        from palladium.emit.tensorops import cooperative_launch, emits_cooperative

        if emits_cooperative(msl_source):
            required, grid = cooperative_launch(spec.grid, simdgroup_width())
            provided = (
                (self._options.threadgroup + (1, 1, 1))[:3]
                if self._options.threadgroup is not None
                else None
            )
            if provided is not None and provided != required:
                raise ValueError(
                    f"cooperative kernel requires threadgroup={required}, got {self._options.threadgroup}"
                )
            threadgroup = required
        else:
            tg = self._options.threadgroup or (0,)
            threadgroup = (tuple(tg) + (1, 1, 1))[:3] if tg != (0,) else (0, 0, 0)
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
            math_mode=self._math_mode_ordinal,
            batch_size=1 if axis_size is None else int(axis_size),
            elem_strides=np.asarray(in_strides + out_strides, dtype=np.int64),
        )
        # ffi_call returns a list for several outputs; pallas_call a tuple.
        return tuple(outs) if len(out_structs) > 1 else outs


def metal_call(kernel: Callable, **pallas_kwargs) -> FfiCallable:
    """`pl.pallas_call`, dispatched to the Apple GPU from the CPU platform.

    Parameters
    ----------
    kernel : callable
        A Pallas kernel function (operates on Refs).
    **pallas_kwargs
        The usual `pl.pallas_call` keywords (out_shape, grid, in_specs,
        out_specs, ...), plus `math_mode` (`metal_runtime.MathMode`, FAST
        by default; SAFE for compensated arithmetic), `threadgroup` (int
        or tuple; None lets the runtime choose), `cache_size` (traced
        shapes kept, 256 by default), `dot_general` ("auto", "default", or
        "tensorops"), and `vmap_method` ("pipelined" by default: one FFI
        call per batch; "sequential" or "sequential_unrolled" dispatch per
        element; None rejects vmap).

    Notes
    -----
    FAST math reorders float arithmetic and approximates transcendentals,
    so results are not bit-equal to the `interpret` oracle: ~1e-6 relative
    for f32 elementwise work, up to ~1e-4 through exp/log-heavy kernels and
    reductions. Use SAFE for IEEE ordering, and always for compensated
    arithmetic (FAST deletes the error terms).

    Returns
    -------
    FfiCallable
        Composable with `jax.jit` and `jax.vmap`; NumPy inputs are accepted
        and outputs are JAX arrays. `.interpret` is the CPU oracle.
    """
    return FfiCallable(
        kernel, pallas_kwargs, CallOptions.split(pallas_kwargs, vmap_method="pipelined")
    )
