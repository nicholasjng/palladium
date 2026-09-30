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
from typing import Any

import jax
import numpy as np

from palladium._callable import CallOptions, PallasCallable
from palladium.diagnostics import simdgroup_width

__all__ = ["FfiCallable", "metal_call_jit"]


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


# Whole-batch methods conflict with the shape-specialized grid. pipelined
# batches in one FFI call; nested vmap levels run sequentially.
_SAFE_VMAP_METHODS = (None, "sequential", "sequential_unrolled", "pipelined")


class FfiCallable(PallasCallable):
    """A palladium kernel registered as a jax.ffi target, dispatched inside
    XLA's execution so it composes under `jax.jit`. Tracing and MSL emission
    are cached per input shape/dtype. Not differentiable by itself (`ffi_call`
    has no JVP/transpose rule); pair with a backward kernel via `with_vjp`."""

    execution_path = "cpu-ffi-to-metal"

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
        super().__init__(kernel, pallas_kwargs, options)
        self._math_mode_ordinal = _MATH_MODE_ORDINALS[options.math_mode]
        self._vmap_method = options.vmap_method
        # None is wrapped too, so its batching error names Palladium's options.
        self._pipelined = (
            self._build_pipelined() if options.vmap_method in ("pipelined", None) else None
        )

    def pin(self, *args) -> Callable[[], Any]:
        """Not available: under `jax.jit` XLA owns the operand buffers. Use
        `metal_call(...).pin`, or keep inputs as device arrays."""
        raise NotImplementedError(
            "FfiCallable.pin is not available: under jax.jit, XLA owns "
            "operand buffers, so there is nothing for palladium to pin "
            "across calls. Use palladium.metal_call(...).pin for the eager "
            "path, or keep inputs as device arrays and let jit reuse them."
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
                    "metal_call_jit(..., vmap_method='pipelined') for one FFI "
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
                (self._threadgroup + (1, 1, 1))[:3] if self._threadgroup is not None else None
            )
            if provided is not None and provided != required:
                raise ValueError(
                    f"cooperative kernel requires threadgroup={required}, got {self._threadgroup}"
                )
            threadgroup = required
        else:
            tg = self._threadgroup or (0,)
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


def metal_call_jit(kernel: Callable, **pallas_kwargs) -> FfiCallable:
    """`pl.pallas_call`, dispatched to the Apple GPU, composable with `jax.jit`.

    Parameters
    ----------
    kernel : callable
        A Pallas kernel function (operates on Refs).
    **pallas_kwargs
        The usual `pl.pallas_call` keywords (out_shape, grid, in_specs,
        out_specs, ...), plus `math_mode` (`metal_runtime.MathMode`, FAST
        by default; SAFE for compensated arithmetic), `threadgroup` (int
        or tuple; None lets the runtime choose), `cache_size` (256 by
        default), `dot_general` ("auto", "default", or "tensorops"; see
        `metal_call`), and `vmap_method`: 'pipelined' (the default) runs
        the whole batch in one FFI call, 'sequential' and
        'sequential_unrolled' dispatch per element, None rejects
        `jax.vmap`. A batch dimension in the Pallas grid is one dispatch
        total and beats all of them.

    Returns
    -------
    FfiCallable
        Traceable and jittable.
    """
    return FfiCallable(
        kernel, pallas_kwargs, CallOptions.split(pallas_kwargs, vmap_method="pipelined")
    )
