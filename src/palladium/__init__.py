"""palladium: Pallas kernels on Apple GPU, via metal-runtime.

Pipeline: trace (Pallas -> KernelSpec) -> emit (KernelSpec -> MSL text)
-> bind (MSL -> callable, via metal-runtime). `metal_call` composes the
three behind a `pl.pallas_call`-shaped entry point; `debug_msl` exposes
the intermediate text.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np

from palladium._callable import CallOptions, PallasCallable, unwrap
from palladium.diagnostics import KernelDiagnostics, explain_spec
from palladium.dispatch import BoundKernel, bind
from palladium.emit import emit_jaxpr, emit_msl, rule
from palladium.errors import (
    DispatchError,
    EmitError,
    PalladiumError,
    StackOverflowError,
    TraceError,
    UnsupportedPrimitiveError,
)
from palladium.ffi import FfiCallable, metal_call_jit
from palladium.mps import MPS_CUSTOM_CALL_TARGET, MpsDispatchDescriptor
from palladium.pallas_backend import CompilerParams, install as _install_pallas_backend
from palladium.threadgroup import (
    barrier,
    thread_index,
    threadgroup_memory,
    threads_per_threadgroup,
)
from palladium.trace import BlockInfo, KernelSpec, ScratchInfo, trace
from palladium.verify import VerificationError
from palladium.vjp import with_auxiliary_vjp, with_reference_vjp, with_vjp

__all__ = [
    "MPS_CUSTOM_CALL_TARGET",
    "BlockInfo",
    "BoundKernel",
    "CompilerParams",
    "DispatchError",
    "EmitError",
    "FfiCallable",
    "KernelDiagnostics",
    "KernelSpec",
    "MetalCallable",
    "MpsDispatchDescriptor",
    "PalladiumError",
    "ScratchInfo",
    "StackOverflowError",
    "TraceError",
    "UnsupportedPrimitiveError",
    "VerificationError",
    "barrier",
    "bind",
    "debug_msl",
    "emit_jaxpr",
    "emit_msl",
    "metal_call",
    "metal_call_jit",
    "rule",
    "thread_index",
    "threadgroup_memory",
    "threads_per_threadgroup",
    "trace",
    "with_auxiliary_vjp",
    "with_reference_vjp",
    "with_vjp",
]

__version__ = "0.2.0"

# Plain pl.pallas_call lowered for the mps platform runs through Palladium.
_install_pallas_backend()

CacheKey = tuple[tuple[tuple[int, ...], str], ...]


class MetalCallable(PallasCallable):
    """The palladium pipeline behind a `pl.pallas_call`-shaped call.

    Retraces per input shape/dtype; identical shapes hit `cache`,
    identical source hits metal-runtime's library cache below that.

    Attributes
    ----------
    interpret : callable
        The same pallas_call with `interpret=True`: the CPU oracle.
    cache : dict
        Maps input-shape signatures to compiled `BoundKernel`s;
        `cache[key].msl_source` is the emitted text for that shape.
    """

    execution_path = "metal"

    def __init__(self, kernel: Callable, pallas_kwargs: dict[str, Any], options: CallOptions):
        super().__init__(kernel, pallas_kwargs, options)
        # LRU of compiled Metal pipelines, one per input-shape signature.
        self.cache: OrderedDict[CacheKey, BoundKernel] = OrderedDict()

    def pin(self, *args) -> Callable[[], np.ndarray | tuple[np.ndarray, ...]]:
        """Upload the inputs once; return a zero-argument callable that
        re-dispatches on the pinned device buffers. For repeated calls on
        unchanging inputs; later mutation of the arrays is not observed."""
        arrays = [np.asarray(a) for a in args]
        return self._bound(arrays).pinned(*arrays)

    def iterate(self, *args, steps: int, feedback=None) -> np.ndarray | tuple[np.ndarray, ...]:
        """Run `steps` dispatches with the state resident on the device.

        Each step's outputs refill the inputs for the next step (output j
        into input j by default, or the given (output, input) pairs);
        inputs never fed back stay fixed. One command buffer carries the
        whole loop, so nothing round-trips through NumPy between steps.
        Returns the outputs of the last step.
        """
        arrays = [np.asarray(a) for a in args]
        return self._bound(arrays).iterate(*arrays, steps=steps, feedback=feedback)

    def __call__(self, *args) -> np.ndarray | tuple[np.ndarray, ...]:
        """Run the kernel on the GPU; NumPy in, NumPy out."""
        arrays = [np.asarray(a) for a in args]
        return self._bound(arrays)(*arrays)

    def _bound(self, arrays: list[np.ndarray]) -> BoundKernel:
        """The compiled kernel for these argument shapes, from the cache."""
        key: CacheKey = tuple((a.shape, a.dtype.str) for a in arrays)
        with self._lock:
            bound = self.cache.get(key)
            if bound is not None:
                self.cache.move_to_end(key)
        if bound is not None:
            return bound
        spec, msl, _ = self._spec_and_msl(tuple(arrays))
        with self._lock:
            bound = self.cache.get(key)
            if bound is None:
                bound = bind(
                    spec,
                    msl,
                    math_mode=self._options.math_mode,
                    threadgroup=self._options.threadgroup,
                    dot_general=self._options.dot_general,
                )
                self.cache[key] = bound
                size = self._options.cache_size
                while size and len(self.cache) > size:
                    self.cache.popitem(last=False)
        return bound


_unwrap = unwrap


def debug_msl(kernel: Callable, *example_args, **pallas_kwargs) -> str:
    """Trace `kernel` through pallas_call and return the emitted MSL.

    Parameters
    ----------
    kernel : callable
        A Pallas kernel function (operates on Refs).
    *example_args
        Arrays or `jax.ShapeDtypeStruct`s fixing input shapes; no data is
        read and nothing is compiled or dispatched.
    **pallas_kwargs
        The usual `pl.pallas_call` keywords (out_shape, grid, ...).

    Returns
    -------
    str
        The MSL source `metal_call` would compile for these shapes.

    Examples
    --------
    >>> print(palladium.debug_msl(k, x, out_shape=...))  # doctest: +SKIP
    """
    import jax.experimental.pallas as pl

    dot_general = pallas_kwargs.pop("dot_general", "auto")
    spec = trace(pl.pallas_call(kernel, **pallas_kwargs), *example_args)
    return emit_msl(spec, dot_general=dot_general)


def metal_call(kernel: Callable, **pallas_kwargs) -> MetalCallable:
    """`pl.pallas_call`, but the kernel runs on the Apple GPU.

    Parameters
    ----------
    kernel : callable
        A Pallas kernel function (operates on Refs).
    **pallas_kwargs
        The usual `pl.pallas_call` keywords (out_shape, grid, in_specs,
        out_specs, ...), plus Metal-side extras: `math_mode`
        (metal_runtime.MathMode, FAST by default), `threadgroup`, and
        `dot_general="default"` to force the primitive matmul path, or
        `dot_general="tensorops"` to force cooperative TensorOps. The default
        automatically selects TensorOps for tiled matmuls and attention.

    Notes
    -----
    FAST math reorders float arithmetic and approximates transcendentals,
    so results are not bit-equal to the `interpret` oracle: ~1e-6 relative
    for f32 elementwise work, up to ~1e-4 through exp/log-heavy kernels and
    reductions. Use SAFE for IEEE ordering, and always for compensated
    arithmetic (FAST deletes the error terms).

    Tiled matmuls, batched matmuls, and supported online-softmax attention
    kernels use TensorOps automatically. Untiled dots use the general primitive
    path. Pass `dot_general="default"` to force that path for all dots, or
    `dot_general="tensorops"` to require TensorOps.

    Returns
    -------
    MetalCallable
        NumPy-in/NumPy-out callable with `.interpret` (the CPU oracle)
        and `.cache` (per-shape compiled kernels).
    """
    return MetalCallable(kernel, pallas_kwargs, CallOptions.split(pallas_kwargs))
