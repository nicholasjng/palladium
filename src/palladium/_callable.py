"""What the three call paths share: option parsing, the per-shape cache of
traced specs and emitted MSL, diagnostics, verification, and VJP attachment.

`metal_call` (eager), `metal_call_jit` (CPU FFI), and `mps_call_jit`
(jax-mps custom call) differ only in how a compiled kernel is dispatched;
everything up to the MSL text is this module.
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
    fallback: str

    @classmethod
    def split(
        cls,
        pallas_kwargs: dict[str, Any],
        *,
        vmap_method: str | None = None,
        fallback: str = "interpret",
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
            fallback=pallas_kwargs.pop("fallback", fallback),
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


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


class PallasCallable:
    """Base of the three call paths.

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

    # Backwards-compatible views of the options.
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
        # Keep lookup and LRU promotion together; another specialization
        # can evict this entry while a concurrent call is touching it.
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
        Kernels using `palladium.threadgroup_memory` need an explicit
        `reference=`: interpret models each instance as a threadgroup of
        one, so it computes something else, and `verify` refuses rather
        than pass silently. Raises `VerificationError` on disagreement,
        naming the worst element and its index.
        """
        return unwrap(
            verify_against(
                self.__call__,
                None if reference is not None else self.interpret,
                self._spec_and_msl(tuple(args))[0].uses_threadgroup,
                args,
                reference,
                rtol,
                atol,
            )
        )

    def __call__(self, *args):  # pragma: no cover - subclasses dispatch
        raise NotImplementedError

    # -- gradients ------------------------------------------------------
    # None of the paths derives a derivative from emitted MSL; these pair
    # the forward call with a backward implementation.

    def with_reference_vjp(self, reference: Callable) -> Callable:
        """Attach a correctness-first VJP computed by JAX from `reference`.

        The primal stays this call; the pullback is JAX's VJP of
        `reference`, which must have the same inputs, outputs, and
        differentiable semantics. Useful for putting a fused forward solve
        into a training loop while a Pallas adjoint kernel is developed; it
        is not a performance solution.
        """

        @jax.custom_vjp
        def differentiated(*args):
            return self(*args)

        def forward(*args):
            return self(*args), args

        def backward(residual, cotangents):
            _, pullback = jax.vjp(reference, *residual)
            return pullback(cotangents)

        differentiated.defvjp(forward, backward)
        return differentiated

    def with_vjp(self, backward_call: Callable) -> Callable:
        """Attach a Pallas custom VJP to this forward call.

        `backward_call` receives the forward primals followed by one
        cotangent per forward output and returns one cotangent per primal,
        in order. It can itself be a palladium call on the same path.
        """

        @jax.custom_vjp
        def differentiated(*args):
            return self(*args)

        def forward(*args):
            return self(*args), args

        def backward(residual, cotangents):
            input_cotangents = _as_tuple(backward_call(*residual, *_as_tuple(cotangents)))
            if len(input_cotangents) != len(residual):
                raise TypeError(
                    "Palladium VJP returned "
                    f"{len(input_cotangents)} input cotangents for "
                    f"{len(residual)} primals"
                )
            return input_cotangents

        differentiated.defvjp(forward, backward)
        return differentiated

    def with_auxiliary_vjp(self, backward_call: Callable, output_count: int) -> Callable:
        """Attach a VJP while retaining trailing forward outputs as residuals.

        The first `output_count` outputs are the public primal result; the
        trailing outputs (checkpoints) are saved and passed to
        `backward_call` after the primals and before the output cotangents.
        """
        if output_count < 1:
            raise ValueError("output_count must be positive")

        def split_outputs(raw_outputs):
            values = _as_tuple(raw_outputs)
            if output_count >= len(values):
                raise ValueError(
                    "with_auxiliary_vjp requires at least one trailing auxiliary output"
                )
            return unwrap(values[:output_count]), values[output_count:]

        @jax.custom_vjp
        def differentiated(*args):
            public, _ = split_outputs(self(*args))
            return public

        def forward(*args):
            public, auxiliaries = split_outputs(self(*args))
            return public, (*args, *auxiliaries)

        def backward(residual, cotangents):
            # The custom-VJP protocol validates the returned pytree against
            # the primal arguments; the auxiliary count is backward_call's ABI.
            return _as_tuple(backward_call(*residual, *_as_tuple(cotangents)))

        differentiated.defvjp(forward, backward)
        return differentiated
