"""palladium: Pallas kernels on Apple GPU, via metal-runtime.

Pipeline: trace (Pallas -> KernelSpec) -> emit (KernelSpec -> MSL text)
-> bind (MSL -> callable, via metal-runtime). `metal_call` composes the
three behind a `pl.pallas_call`-shaped entry point; `debug_msl` exposes
the intermediate text.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np

from palladium.diagnostics import KernelDiagnostics, explain_spec, log_compile
from palladium.dispatch import BoundKernel, bind
from palladium.emit import emit_jaxpr, emit_msl, rule
from palladium.emit.core import CTYPES as _CTYPES
from palladium.errors import (
    DispatchError,
    EmitError,
    PalladiumError,
    StackOverflowError,
    TraceError,
    UnsupportedPrimitiveError,
)
from palladium.ffi import FfiCallable, metal_call_jit
from palladium.threadgroup import (
    barrier,
    thread_index,
    threadgroup_memory,
    threads_per_threadgroup,
)
from palladium.trace import BlockInfo, KernelSpec, ScratchInfo, trace
from palladium.verify import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    VerificationError,
    verify_against,
)

__all__ = [
    "BlockInfo",
    "BoundKernel",
    "DispatchError",
    "EmitError",
    "FfiCallable",
    "KernelDiagnostics",
    "KernelSpec",
    "MetalCallable",
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
]

__version__ = "0.2.0"

CacheKey = tuple[tuple[tuple[int, ...], str], ...]


def _check_dtypes(args: tuple) -> None:
    """Reject unsupported element types before tracing, with the fix in
    the message; without this they surface as a KeyError deep in emit.

    bfloat16 is supported on both paths: NumPy will not export
    ml_dtypes extension dtypes over DLPack, so the eager path ships the
    bytes as uint16 and relabels the buffer (see dispatch._to_native, a
    lossless reinterpretation); the FFI path passes raw pointers.
    """
    for i, a in enumerate(args):
        dtype = getattr(a, "dtype", None)
        name = np.dtype(dtype if dtype is not None else np.asarray(a).dtype).name
        if name not in _CTYPES:
            hint = (
                "; float64 usually means jax_enable_x64 is on, disable it "
                "or cast to float32"
                if name == "float64"
                else ""
            )
            raise DispatchError(
                f"argument {i} has dtype {name}, which palladium cannot "
                f"lower (supported: {', '.join(_CTYPES)}){hint}"
            )


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
        cache_size: int = 256,
    ) -> None:
        import jax.experimental.pallas as pl

        self._staged = pl.pallas_call(kernel, **pallas_kwargs)
        self._math_mode = math_mode
        self._threadgroup = threadgroup
        self.interpret = pl.pallas_call(kernel, **pallas_kwargs, interpret=True)
        # Bounded like the jax.ffi path's: each entry holds a compiled
        # Metal pipeline, so an unbounded cache in a long-lived process
        # tracing many shapes retains every one. Insertion-ordered and
        # re-inserted on hit, so popping the first item evicts the LRU.
        self.cache: OrderedDict[CacheKey, BoundKernel] = OrderedDict()
        self._cache_size = cache_size
        # Guards trace/emit/compile on a cache miss: concurrent first
        # calls on the same shape must compile exactly once.
        self._lock = threading.Lock()

    def explain(self, *args) -> KernelDiagnostics:
        """Report how the kernel executes for these inputs: launch
        geometry and emitted MSL size. Emits MSL to measure it; compiles
        and dispatches nothing.

        Parameters
        ----------
        *args
            Arrays or `jax.ShapeDtypeStruct`s fixing input shapes; no
            data is read.
        """
        _check_dtypes(args)
        return explain_spec(trace(self._staged, *args), self._threadgroup)

    def verify(
        self,
        *args,
        reference=None,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
    ):
        """Run on the GPU and diff against a reference; return the output.

        The project's validation doctrine in one call. By default the
        reference is this kernel's own `interpret=True` oracle, so a
        round trip is `f.verify(x)` instead of a hand-rolled
        `assert_allclose(f(x), f.interpret(x))`.

        Parameters
        ----------
        *args
            Kernel inputs, same as `__call__`.
        reference : callable, optional
            Alternative reference taking the same arguments. Required
            for kernels using `palladium.threadgroup_memory`: interpret
            models each program instance as a threadgroup of one, so it
            computes a different thing and comparing to it is
            meaningless. `verify` refuses rather than pass silently.
        rtol, atol : float, optional
            Tolerances. The defaults allow for FAST math, which reorders
            float arithmetic and approximates transcendentals.

        Returns
        -------
        The GPU output, so `verify` can replace a call site directly.

        Raises
        ------
        VerificationError
            On disagreement, naming the worst element and its index.
        """
        return _unwrap(
            verify_against(
                self.__call__,
                None if reference is not None else self.interpret,
                trace(self._staged, *args).uses_threadgroup,
                args,
                reference,
                rtol,
                atol,
            )
        )

    def pin(self, *args) -> Callable[[], np.ndarray | tuple[np.ndarray, ...]]:
        """Upload the inputs once; return a zero-argument callable that
        re-dispatches on the pinned device buffers (see
        `BoundKernel.pinned`). Use for repeated calls on unchanging
        inputs. Later mutation of the passed arrays is not observed."""
        arrays = [np.asarray(a) for a in args]
        self(*arrays)  # populate the shape cache (trace/emit/compile)
        key: CacheKey = tuple((a.shape, a.dtype.str) for a in arrays)
        return self.cache[key].pinned(*arrays)

    def __call__(self, *args) -> np.ndarray | tuple[np.ndarray, ...]:
        """Run the kernel on the GPU; NumPy in, NumPy out."""
        arrays = [np.asarray(a) for a in args]
        key: CacheKey = tuple((a.shape, a.dtype.str) for a in arrays)
        bound = self.cache.get(key)
        if bound is not None:
            self.cache.move_to_end(key)
        if bound is None:
            with self._lock:
                bound = self.cache.get(key)
                if bound is None:
                    _check_dtypes(tuple(arrays))
                    spec = trace(self._staged, *arrays)
                    log_compile(spec, self._threadgroup)
                    bound = bind(
                        spec,
                        emit_msl(spec),
                        math_mode=self._math_mode,
                        threadgroup=self._threadgroup,
                    )
                    self.cache[key] = bound
                    while self._cache_size and len(self.cache) > self._cache_size:
                        self.cache.popitem(last=False)
        return bound(*arrays)


def _unwrap(outs):
    """Match `__call__`'s convention: a bare array for single-output."""
    return outs[0] if len(outs) == 1 else outs


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

    Notes
    -----
    The default FAST math mode reorders float arithmetic and uses
    approximate transcendentals, so results are not bit-equal to the
    `interpret` oracle: expect ~1e-6 relative deviation for f32
    elementwise work, up to ~1e-4 through exp/log-heavy kernels and
    reductions (whose combine order also differs). Use SAFE for IEEE
    ordering, and always for compensated arithmetic (FAST deletes the
    error terms).

    Returns
    -------
    MetalCallable
        NumPy-in/NumPy-out callable with `.interpret` (the CPU oracle)
        and `.cache` (per-shape compiled kernels).
    """
    from metal_runtime import MathMode

    math_mode = pallas_kwargs.pop("math_mode", MathMode.FAST)
    threadgroup = pallas_kwargs.pop("threadgroup", None)
    cache_size = pallas_kwargs.pop("cache_size", 256)
    return MetalCallable(kernel, pallas_kwargs, math_mode, threadgroup, cache_size)
