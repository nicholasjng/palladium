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
import functools
import importlib.resources
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np

from palladium.emit import emit_msl, is_simdgroup_cooperative
from palladium.emit.core import SIMDGROUP_WIDTH
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


@functools.cache
def _register() -> None:
    """Loads the native handler and registers it, once per process."""
    handle = ctypes.CDLL(str(_library_path()))
    capsule = jax.ffi.pycapsule(handle.palladium_dispatch)
    jax.ffi.register_ffi_target(_TARGET_NAME, capsule, platform="cpu")


class FfiCallable:
    """A palladium kernel registered as a jax.ffi target: jax.jit-composable.

    Unlike `MetalCallable` (NumPy in, NumPy out, eager), dispatch happens
    inside XLA's execution via `native/ffi/palladium_ffi.cpp`'s generic
    handler, so a call composes inside `jax.jit` next to `jnp` ops.
    Tracing and MSL emission are cached per input shape/dtype, same as
    `MetalCallable` caches `BoundKernel`s.

    Not differentiable: `ffi_call` has no JVP/transpose rule by default
    (a `custom_vjp` wrapper is the path if ever needed).

    Attributes
    ----------
    interpret : callable
        The same pallas_call with `interpret=True`: the CPU oracle.
    """

    def __init__(
        self, kernel: Callable, pallas_kwargs: dict[str, Any], math_mode: Any
    ) -> None:
        import jax.experimental.pallas as pl

        self._staged = pl.pallas_call(kernel, **pallas_kwargs)
        self.interpret = pl.pallas_call(kernel, **pallas_kwargs, interpret=True)
        value = math_mode.value if hasattr(math_mode, "value") else math_mode
        self._math_mode = _MATH_MODE_ORDINALS[value]
        self._cache: dict[tuple, tuple[KernelSpec, str, bool]] = {}

    def _spec_and_msl(self, args: tuple) -> tuple[KernelSpec, str, bool]:
        key = tuple((a.shape, np.dtype(a.dtype).str) for a in args)
        entry = self._cache.get(key)
        if entry is None:
            shapes = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args]
            spec = trace(self._staged, *shapes)
            cooperative = is_simdgroup_cooperative(spec)
            entry = (spec, emit_msl(spec, cooperative=cooperative), cooperative)
            self._cache[key] = entry
        return entry

    def __call__(self, *args):
        """Dispatch via jax.ffi; traceable and jittable."""
        _register()
        spec, msl_source, cooperative = self._spec_and_msl(args)
        # MRLaunchDesc always wants 3 grid dims; palladium grids are 1-3D.
        grid = tuple(spec.grid) + (1, 1, 1)
        if cooperative:
            # One SIMD-group (one threadgroup) per instance; must match
            # emit_msl's instance indexing, same as dispatch.BoundKernel.
            grid = (grid[0] * SIMDGROUP_WIDTH, *grid[1:])
            threadgroup = (SIMDGROUP_WIDTH, 1, 1)
        else:
            threadgroup = (0, 0, 0)  # runtime chooses
        out_structs = [
            jax.ShapeDtypeStruct(info.array_shape, info.dtype) for info in spec.outputs
        ]
        result_shapes = out_structs[0] if len(out_structs) == 1 else out_structs
        return jax.ffi.ffi_call(_TARGET_NAME, result_shapes)(
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
        FAST by default; SAFE for df32-prelude kernels).

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
    return FfiCallable(kernel, pallas_kwargs, math_mode)
