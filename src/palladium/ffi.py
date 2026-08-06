"""jax.ffi bridge: registers palladium kernels as a real JAX primitive,
composable with jax.jit, through metal-runtime's C API (`native/ffi/`).

`PALLADIUM_FFI_LIBRARY` overrides the path for an out-of-tree build.
"""

from __future__ import annotations

import ctypes
import importlib.resources
import math
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metal_runtime import MathMode

import jax
import numpy as np

from palladium.diagnostics import (
    KernelDiagnostics,
    explain_spec,
    log_compile,
    normalize_threadgroup,
)
from palladium.emit import emit_msl
from palladium.errors import EmitError
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


# Batching methods palladium can honor. The grid is baked per unbatched
# shape, so jax.ffi's own whole-batch methods (expand_dims, broadcast_all)
# would silently dispatch it over batched buffers; they stay rejected.
# The sequential methods re-invoke the target once per element, each a
# full blocking dispatch. 'pipelined' (palladium's own, via custom_vmap)
# makes one FFI call whose native handler loops over the batch with
# several dispatches in flight. One vmap level only.
_SAFE_VMAP_METHODS = (None, "sequential", "sequential_unrolled", "pipelined")


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
        math_mode: MathMode | str,
        vmap_method: str | None = None,
        threadgroup: int | tuple[int, ...] | None = None,
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
        # MathMode is a StrEnum, so members index the dict as their value.
        self._math_mode = _MATH_MODE_ORDINALS[math_mode]
        self._vmap_method = vmap_method
        self._threadgroup = normalize_threadgroup(threadgroup)
        self._cache: dict[tuple, tuple[KernelSpec, str]] = {}
        # Guards trace/emit on a cache miss, mirroring MetalCallable.
        self._lock = threading.Lock()
        self._pipelined = (
            self._build_pipelined() if vmap_method == "pipelined" else None
        )

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
        return explain_spec(trace(self._staged, *shapes), self._threadgroup)

    def _spec_and_msl(self, args: tuple[Any, ...]) -> tuple[KernelSpec, str]:
        key = tuple((a.shape, np.dtype(a.dtype).str) for a in args)
        entry = self._cache.get(key)
        if entry is None:
            with self._lock:
                entry = self._cache.get(key)
                if entry is None:
                    shapes = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args]
                    spec = trace(self._staged, *shapes)
                    log_compile(spec)
                    if spec.uses_threadgroup and self._threadgroup is None:
                        raise EmitError(
                            f"kernel {spec.name!r} declares threadgroup_memory "
                            "scratch, so it must be dispatched with an explicit "
                            "threadgroup= size; metal_call_jit otherwise launches "
                            "with a runtime-chosen one (commonly far larger than "
                            "the declared extent), and a thread_index() past that "
                            "extent writes out of bounds with no error. Pass "
                            "metal_call_jit(..., threadgroup=N). Mirrors the same "
                            "check in palladium.bind for the eager path."
                        )
                    entry = (spec, emit_msl(spec))
                    self._cache[key] = entry
        return entry

    def __call__(self, *args):
        """Dispatch via jax.ffi; traceable and jittable."""
        _register()
        if self._pipelined is not None:
            return self._pipelined(*args)
        return self._ffi_dispatch(args, vmap_method=self._vmap_method)

    def _build_pipelined(self):
        """The 'pipelined' batching rule: under jax.vmap, one FFI call
        handling the whole batch natively; unvmapped calls take the
        ordinary single-dispatch path."""
        import jax.custom_batching

        @jax.custom_batching.custom_vmap
        def pipelined(*args):
            return self._ffi_dispatch(args, vmap_method=None)

        @pipelined.def_vmap
        def _pipelined_vmap_rule(axis_size, in_batched, *args):
            out = self._ffi_dispatch(
                args, vmap_method=None, in_batched=in_batched, axis_size=axis_size
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
        (batched args carry the batch at axis 0, unbatched ones ride
        along at stride 0), the native handler loops over the batch;
        without them, one plain dispatch."""
        batched = in_batched if in_batched is not None else [False] * len(args)
        unbatched = [
            jax.ShapeDtypeStruct(a.shape[1:] if b else a.shape, a.dtype)
            for a, b in zip(args, batched, strict=True)
        ]
        spec, msl_source = self._spec_and_msl(tuple(unbatched))
        # MRLaunchDesc always wants 3 grid dims; palladium grids are 1-3D.
        grid = tuple(spec.grid) + (1, 1, 1)
        # (0, 0, 0) lets the runtime choose (c_api.cpp); a cooperative
        # kernel never reaches here with None, per the check above.
        tg = self._threadgroup or (0,)
        threadgroup = (tuple(tg) + (1, 1, 1))[:3] if tg != (0,) else (0, 0, 0)
        in_strides = [
            np.dtype(u.dtype).itemsize * math.prod(u.shape) if b else 0
            for u, b in zip(unbatched, batched, strict=True)
        ]
        out_strides = [
            np.dtype(info.dtype).itemsize * math.prod(info.array_shape)
            for info in spec.outputs
        ]
        lead = () if axis_size is None else (int(axis_size),)
        out_structs = [
            jax.ShapeDtypeStruct(lead + tuple(info.array_shape), info.dtype)
            for info in spec.outputs
        ]
        result_shape_dtypes = out_structs[0] if len(out_structs) == 1 else out_structs
        return jax.ffi.ffi_call(
            _TARGET_NAME,
            result_shape_dtypes=result_shape_dtypes,
            vmap_method=vmap_method,
            # In-place pairs from pallas input_output_aliases,
            # XLA may then donate the input buffer.
            input_output_aliases=dict(spec.aliases) or None,
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
            batch_size=1 if axis_size is None else int(axis_size),
            elem_strides=np.asarray(in_strides + out_strides, dtype=np.int64),
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
        `vmap_method` ('pipelined', 'sequential', or 'sequential_unrolled';
        None, the default, rejects `jax.vmap`).
        'pipelined' handles the whole batch in one FFI call and is the
        fastest vmap path; a batch dimension in the Pallas grid still
        beats it (one dispatch total). `threadgroup` (int or tuple; None
        lets the runtime choose) is required for kernels using
        `palladium.threadgroup_memory`.

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
    threadgroup = pallas_kwargs.pop("threadgroup", None)
    return FfiCallable(kernel, pallas_kwargs, math_mode, vmap_method, threadgroup)
