"""`metal_call`: a Pallas kernel dispatched to Metal from the CPU platform.

The kernel is registered as a jax.ffi target backed by metal-runtime's C
API (`native/ffi/`), so it composes with jax.jit and jax.vmap and needs no
PJRT plugin. `PALLADIUM_FFI_LIBRARY` overrides the handler path for an
out-of-tree build.
"""

from __future__ import annotations

import ctypes
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

from palladium.diagnostics import KernelDiagnostics, explain_spec, log_compile
from palladium.emit import emit_msl
from palladium.emit.core import CTYPES
from palladium.errors import DispatchError
from palladium.launch import CompilerParams, check_threadgroup, launch_geometry
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
    """Load the native handler and register it, once per process."""
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


# Traced shapes kept per callable.
_CACHE_SIZE = 256


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
                    check_threadgroup(spec, self._params.threadgroup)
                    log_compile(spec, self._params.threadgroup, self._params.dot_general)
                    msl = emit_msl(spec, dot_general=self._params.dot_general)
                    entry = (spec, msl, hashlib.sha256(msl.encode()).hexdigest())
                    self._cache[key] = entry
                    if len(self._cache) > _CACHE_SIZE:
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
            self._params.threadgroup,
            dot_general=self._params.dot_general,
        )

    def __call__(self, *args):
        """Dispatch via jax.ffi; traceable and jittable."""
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
        grid, threadgroup = launch_geometry(spec, msl_source, self._params.threadgroup)
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
            math_mode=_MATH_MODE_ORDINALS[self._params.math_mode],
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
        out_specs, ...). Metal-side options travel as
        `compiler_params=palladium.CompilerParams(...)`, as on mps.

    Returns
    -------
    FfiCallable
        Composable with `jax.jit` and `jax.vmap` (one FFI call per batch);
        NumPy inputs are accepted and outputs are JAX arrays. `.interpret`
        is the CPU oracle.

    Notes
    -----
    FAST math reorders float arithmetic and approximates transcendentals,
    so results are not bit-equal to the `interpret` oracle: ~1e-6 relative
    for f32 elementwise work, up to ~1e-4 through exp/log-heavy kernels and
    reductions. Use SAFE for IEEE ordering, and always for compensated
    arithmetic (FAST deletes the error terms).
    """
    return FfiCallable(kernel, pallas_kwargs)
