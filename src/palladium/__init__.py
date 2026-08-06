"""palladium: Pallas kernels on Apple GPU, via metal-runtime.

Pipeline: trace (Pallas -> KernelSpec) -> emit (KernelSpec -> MSL text)
-> bind (MSL -> callable, via metal-runtime). `metal_call` composes the
three behind a `pl.pallas_call`-shaped entry point.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from palladium.dispatch import BoundKernel, bind
from palladium.emit import emit_jaxpr, emit_msl, rule
from palladium.trace import BlockInfo, KernelSpec, trace

__all__ = [
    "BlockInfo",
    "BoundKernel",
    "KernelSpec",
    "MetalCallable",
    "bind",
    "emit_jaxpr",
    "emit_msl",
    "metal_call",
    "rule",
    "trace",
]

__version__ = "0.1.0"

CacheKey = tuple[tuple[tuple[int, ...], str], ...]


class MetalCallable:
    """The palladium pipeline behind a `pl.pallas_call`-shaped call.

    Takes and returns NumPy arrays; retraces per input shape/dtype and
    caches the compiled kernel per shape in `.cache`. `.interpret` is the
    same pallas_call with interpret=True — the CPU correctness oracle.
    """

    def __init__(
        self,
        kernel: Callable,
        pallas_kwargs: dict[str, Any],
        math_mode: Any,
        threadgroup: int | tuple[int, ...] | None,
    ) -> None:
        import jax.experimental.pallas as pl

        self._staged = pl.pallas_call(kernel, **pallas_kwargs)
        self._math_mode = math_mode
        self._threadgroup = threadgroup
        self.interpret = pl.pallas_call(kernel, **pallas_kwargs, interpret=True)
        self.cache: dict[CacheKey, BoundKernel] = {}

    def __call__(self, *args) -> np.ndarray | tuple[np.ndarray, ...]:
        arrays = [np.asarray(a) for a in args]
        key: CacheKey = tuple((a.shape, a.dtype.str) for a in arrays)
        bound = self.cache.get(key)
        if bound is None:
            spec = trace(self._staged, *arrays)
            bound = bind(
                spec,
                emit_msl(spec),
                math_mode=self._math_mode,
                threadgroup=self._threadgroup,
            )
            self.cache[key] = bound
        return bound(*arrays)


def metal_call(kernel: Callable, **pallas_kwargs) -> MetalCallable:
    """`pl.pallas_call`, but the kernel runs on the Apple GPU.

    Takes the same keywords as `pl.pallas_call` (out_shape, grid, in_specs,
    out_specs, ...) plus `math_mode=` / `threadgroup=` for the Metal side.
    """
    from metal_runtime import MathMode

    math_mode = pallas_kwargs.pop("math_mode", MathMode.FAST)
    threadgroup = pallas_kwargs.pop("threadgroup", None)
    return MetalCallable(kernel, pallas_kwargs, math_mode, threadgroup)
