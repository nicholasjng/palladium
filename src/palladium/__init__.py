"""palladium: Pallas kernels on Apple GPU, via metal-runtime.

Pipeline: trace (Pallas -> KernelSpec) -> emit (KernelSpec -> MSL text)
-> bind (MSL -> callable, via metal-runtime). `metal_call` composes the
three behind a `pl.pallas_call`-shaped entry point; `debug_msl` exposes
the intermediate text.
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
    "debug_msl",
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
        """Run the kernel on the GPU; NumPy in, NumPy out."""
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


def debug_msl(kernel: Callable, *example_args, **pallas_kwargs) -> str:
    """Trace `kernel` through pallas_call and return the emitted MSL.

    Per-operand pointer lines at the top of the body carry the BlockSpec
    offsets; the rest is the kernel jaxpr, statement by statement.

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

    spec = trace(pl.pallas_call(kernel, **pallas_kwargs), *example_args)
    return emit_msl(spec)


def metal_call(kernel: Callable, **pallas_kwargs) -> MetalCallable:
    """`pl.pallas_call`, but the kernel runs on the Apple GPU.

    Parameters
    ----------
    kernel : callable
        A Pallas kernel function (operates on Refs).
    **pallas_kwargs
        The usual `pl.pallas_call` keywords (out_shape, grid, in_specs,
        out_specs, ...), plus two Metal-side extras: `math_mode`
        (metal_runtime.MathMode, FAST by default) and `threadgroup`
        (explicit threadgroup size; None lets the runtime choose).

    Returns
    -------
    MetalCallable
        NumPy-in/NumPy-out callable with `.interpret` (the CPU oracle)
        and `.cache` (per-shape compiled kernels).
    """
    from metal_runtime import MathMode

    math_mode = pallas_kwargs.pop("math_mode", MathMode.FAST)
    threadgroup = pallas_kwargs.pop("threadgroup", None)
    return MetalCallable(kernel, pallas_kwargs, math_mode, threadgroup)
