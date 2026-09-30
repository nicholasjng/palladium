"""What a Palladium callable is beyond dispatch: option parsing, the
per-shape cache of traced specs and emitted MSL, diagnostics, verification,
and VJP attachment.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from palladium.diagnostics import (
    KernelDiagnostics,
    check_threadgroup,
    explain_spec,
    log_compile,
    normalize_threadgroup,
)
from palladium.emit import emit_msl
from palladium.emit.core import CTYPES
from palladium.errors import DispatchError
from palladium.trace import KernelSpec, trace
from palladium.verify import DEFAULT_ATOL, DEFAULT_RTOL, verify_against
from palladium.vjp import with_auxiliary_vjp, with_reference_vjp, with_vjp

__all__ = ["CallOptions", "PallasCallable", "check_dtypes", "unwrap"]

DOT_GENERAL_POLICIES = ("auto", "default", "tensorops")


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


def unwrap(outs):
    """Match `__call__`'s convention: a bare array for single-output."""
    return outs[0] if len(outs) == 1 else outs


class PallasCallable:
    """Trace-and-emit half of a callable; subclasses dispatch.

    Attributes
    ----------
    interpret : callable
        The same pallas_call with `interpret=True`: the CPU oracle.
    """

    #: Reported by `explain`; subclasses name their dispatch route.
    execution_path = "metal"

    def __init__(
        self, kernel: Callable, pallas_kwargs: dict[str, Any], options: CallOptions
    ) -> None:
        import jax.experimental.pallas as pl

        self._staged = pl.pallas_call(kernel, **pallas_kwargs)
        self.interpret = pl.pallas_call(kernel, **pallas_kwargs, interpret=True)
        self._options = options
        # Bounded LRU of (spec, msl, digest) per input shape signature.
        self._cache: OrderedDict[tuple, tuple[KernelSpec, str, str]] = OrderedDict()
        # Serialize cache misses: concurrent first calls compile once.
        self._lock = threading.Lock()

    @property
    def _math_mode(self):
        return self._options.math_mode

    @property
    def _threadgroup(self):
        return self._options.threadgroup

    @property
    def _dot_general(self):
        return self._options.dot_general

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
                    log_compile(
                        spec,
                        self._options.threadgroup,
                        execution_path=self.execution_path,
                        dot_general=self._options.dot_general,
                    )
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
        return dataclasses.replace(
            explain_spec(
                trace(self._staged, *self._shapes(args)),
                self._options.threadgroup,
                dot_general=self._options.dot_general,
            ),
            execution_path=self.execution_path,
        )

    def verify(
        self,
        *args,
        reference=None,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
    ):
        """Run on the GPU and diff against a reference; return the output.

        The reference defaults to this kernel's `interpret=True` oracle.
        Cooperative kernels (threadgroup_memory, thread_index, barrier)
        need an explicit `reference=`: interpret models each instance as
        a threadgroup of one, so `verify` refuses the oracle for them.
        Raises `VerificationError` on disagreement, naming the worst
        element and its index.
        """
        return unwrap(
            verify_against(
                self.__call__,
                self.interpret,
                self._spec_and_msl(tuple(args))[0].uses_threadgroup,
                args,
                reference,
                rtol,
                atol,
            )
        )

    def __call__(self, *args):  # pragma: no cover - subclasses dispatch
        raise NotImplementedError

    def with_reference_vjp(self, reference: Callable) -> Callable:
        """`palladium.with_reference_vjp(self, reference)`."""
        return with_reference_vjp(self, reference)

    def with_vjp(self, backward_call: Callable) -> Callable:
        """`palladium.with_vjp(self, backward_call)`."""
        return with_vjp(self, backward_call)

    def with_auxiliary_vjp(self, backward_call: Callable, output_count: int) -> Callable:
        """`palladium.with_auxiliary_vjp(self, backward_call, output_count)`."""
        return with_auxiliary_vjp(self, backward_call, output_count)
