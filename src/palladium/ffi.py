"""jax.ffi bridge: registers palladium kernels as a real JAX primitive,
composable with jax.jit, through metal-runtime's C API (`native/ffi/`).

The native handler (`libpalladium_ffi.dylib`) is built by scikit-build-core
as part of the normal package build (`uv sync`/`pip install`), from the
repo root's `CMakeLists.txt`; nothing to build separately. Found via
`importlib.resources`, same pattern as `metal_runtime.c_api`, so this
works for both an editable dev install (dylib in the CMake build dir)
and a real wheel install (dylib inside the installed package).
`PALLADIUM_FFI_LIBRARY` overrides the path for an out-of-tree build.
"""

from __future__ import annotations

import ctypes
import importlib.resources
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np

from palladium.diagnostics import KernelDiagnostics, explain_spec, log_compile
from palladium.emit import emit_msl, is_simdgroup_cooperative
from palladium.emit.core import SIMDGROUP_WIDTH
from palladium.emit.gemm import gemm_groups
from palladium.trace import KernelSpec, trace

__all__ = ["FfiCallable", "metal_call_jit"]

_TARGET_NAME = "palladium_dispatch"
_LIBRARY_NAME = "libpalladium_ffi.dylib"

# MRMathMode ordinals from native/ffi's c_api.h; metal_runtime.MathMode is a
# StrEnum ("safe"/"relaxed"/"fast"), not the C API's int, so this is the one
# place that needs to know both.
_MATH_MODE_ORDINALS = {"safe": 0, "relaxed": 1, "fast": 2}


def _library_path() -> Path:
    override = os.environ.get("PALLADIUM_FFI_LIBRARY")
    if override:
        return Path(override)
    # `palladium`'s __path__ can span multiple roots (editable install:
    # source tree + CMake build dir), so join-and-check resolves to
    # whichever root actually has the file, same as
    # metal_runtime.c_api's own include_dir()/library_dir().
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
    """Loads the native handler and registers it, once per process.

    Locked, not `functools.cache`: concurrent first calls must not both
    run the registration body.
    """
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


# jax.ffi.ffi_call's batching methods that re-invoke the target once per
# batch element, which is the only shape palladium can honor: the grid
# and MSL source are baked as FFI attributes for the unbatched shape, so
# the whole-batch methods (expand_dims, broadcast_all) would dispatch
# that grid over batched buffers and return wrong results silently.
_SAFE_VMAP_METHODS = (None, "sequential", "sequential_unrolled")


class FfiCallable:
    """A palladium kernel registered as a jax.ffi target: jax.jit-composable.

    Unlike `MetalCallable` (NumPy in, NumPy out, eager), dispatch happens
    inside XLA's execution via `native/ffi/palladium_ffi.cpp`'s generic
    handler, so a call composes inside `jax.jit` next to `jnp` ops.
    Tracing and MSL emission are cached per input shape/dtype, same as
    `MetalCallable` caches `BoundKernel`s.

    Not differentiable by itself: `ffi_call` has no JVP/transpose rule.
    Pair a forward and a backward kernel through `jax.custom_vjp`; see
    `examples/07_custom_vjp.py` for the worked recipe.

    Attributes
    ----------
    interpret : callable
        The same pallas_call with `interpret=True`: the CPU oracle.
    """

    def __init__(
        self,
        kernel: Callable,
        pallas_kwargs: dict[str, Any],
        math_mode: Any,
        vmap_method: str | None = None,
    ) -> None:
        import jax.experimental.pallas as pl

        if vmap_method not in _SAFE_VMAP_METHODS:
            raise ValueError(
                f"vmap_method {vmap_method!r} is not supported: the launch "
                "grid is baked per unbatched shape, so whole-batch methods "
                "would dispatch it over batched buffers. Use 'sequential' "
                "or 'sequential_unrolled' (one dispatch per batch element), "
                "or put the batch dimension in the Pallas grid instead."
            )
        self._staged = pl.pallas_call(kernel, **pallas_kwargs)
        self.interpret = pl.pallas_call(kernel, **pallas_kwargs, interpret=True)
        value = math_mode.value if hasattr(math_mode, "value") else math_mode
        self._math_mode = _MATH_MODE_ORDINALS[value]
        self._vmap_method = vmap_method
        self._cache: dict[tuple, tuple[KernelSpec, str, bool, int]] = {}
        # Guards trace/emit on a cache miss, mirroring MetalCallable.
        self._lock = threading.Lock()

    def explain(self, *args) -> KernelDiagnostics:
        """Report how the kernel executes for these inputs, mirroring
        `MetalCallable.explain`. Emits MSL; compiles and dispatches
        nothing.

        Parameters
        ----------
        *args
            Arrays or `jax.ShapeDtypeStruct`s fixing input shapes; no
            data is read.
        """
        shapes = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args]
        return explain_spec(trace(self._staged, *shapes))

    def _spec_and_msl(self, args: tuple) -> tuple[KernelSpec, str, bool, int]:
        key = tuple((a.shape, np.dtype(a.dtype).str) for a in args)
        entry = self._cache.get(key)
        if entry is None:
            with self._lock:
                entry = self._cache.get(key)
                if entry is None:
                    shapes = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args]
                    spec = trace(self._staged, *shapes)
                    cooperative = is_simdgroup_cooperative(spec)
                    groups = gemm_groups(spec) if cooperative else 1
                    log_compile(spec)
                    entry = (
                        spec,
                        emit_msl(spec, cooperative=cooperative),
                        cooperative,
                        groups,
                    )
                    self._cache[key] = entry
        return entry

    def __call__(self, *args):
        """Dispatch via jax.ffi; traceable and jittable."""
        _register()
        spec, msl_source, cooperative, groups = self._spec_and_msl(args)
        # MRLaunchDesc always wants 3 grid dims; palladium grids are 1-3D.
        grid = tuple(spec.grid) + (1, 1, 1)
        if cooperative:
            # One SIMD-group per instance (several per threadgroup when
            # the GEMM lowering applies); must match emit_msl's instance
            # indexing, same as dispatch.BoundKernel.
            grid = (grid[0] * SIMDGROUP_WIDTH, *grid[1:])
            threadgroup = (SIMDGROUP_WIDTH * groups, 1, 1)
        else:
            threadgroup = (0, 0, 0)  # runtime chooses
        out_structs = [
            jax.ShapeDtypeStruct(info.array_shape, info.dtype) for info in spec.outputs
        ]
        result_shapes = out_structs[0] if len(out_structs) == 1 else out_structs
        return jax.ffi.ffi_call(
            _TARGET_NAME, result_shapes, vmap_method=self._vmap_method
        )(
            *args,
            msl_source=msl_source,
            function_name=spec.name,
            grid_x=int(grid[0]),
            grid_y=int(grid[1]),
            grid_z=int(grid[2]),
            threadgroup_x=int(threadgroup[0]),
            threadgroup_y=int(threadgroup[1]),
            threadgroup_z=int(threadgroup[2]),
            math_mode=self._math_mode,
        )


def metal_call_jit(kernel: Callable, **pallas_kwargs) -> FfiCallable:
    """`pl.pallas_call`, dispatched to the Apple GPU, composable with `jax.jit`.

    Parameters
    ----------
    kernel : callable
        A Pallas kernel function (operates on Refs).
    **pallas_kwargs
        The usual `pl.pallas_call` keywords (out_shape, grid, in_specs,
        out_specs, ...), plus `math_mode` (`metal_runtime.MathMode`,
        FAST by default; SAFE for df32-prelude kernels) and
        `vmap_method` ('sequential' or 'sequential_unrolled'; None, the
        default, rejects `jax.vmap`). The sequential methods dispatch
        once per batch element, each paying the fixed dispatch floor, so
        a batch dimension in the Pallas grid is the fast path; vmap is
        the convenience.

    Returns
    -------
    FfiCallable
        Traceable, jittable; composes with surrounding `jnp` code.

    Examples
    --------
    >>> add_one = metal_call_jit(kernel, out_shape=...)  # doctest: +SKIP
    >>> jax.jit(lambda x: jnp.sum(add_one(x) ** 2))(x)  # doctest: +SKIP
    """
    from metal_runtime import MathMode

    math_mode = pallas_kwargs.pop("math_mode", MathMode.FAST)
    vmap_method = pallas_kwargs.pop("vmap_method", None)
    return FfiCallable(kernel, pallas_kwargs, math_mode, vmap_method)
