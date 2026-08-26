"""Exception hierarchy: everything palladium raises derives from
PalladiumError, split by pipeline stage (trace, emit, dispatch).

TraceError also subclasses ValueError, DispatchError TypeError.

Errors worth branching on carry the deciding value as an attribute, so
handlers need not match on message text:
`UnsupportedPrimitiveError.primitive` and
`StackOverflowError.stack_bytes`/`.limit`.
"""

from __future__ import annotations

__all__ = [
    "DispatchError",
    "EmitError",
    "PalladiumError",
    "StackOverflowError",
    "TraceError",
    "UnsupportedPrimitiveError",
]


class PalladiumError(Exception):
    """Base class for all palladium errors."""


class TraceError(PalladiumError, ValueError):
    """The pallas_call cannot be traced into a KernelSpec (invalid or
    unsupported structure: zero or multiple pallas_calls, scalar
    prefetch, unknown block dim types)."""


class EmitError(PalladiumError):
    """The emitter cannot lower this kernel (unsupported or invalid
    input: shapes, dtypes, primitive parameters, memory layout)."""


class UnsupportedPrimitiveError(EmitError, NotImplementedError):
    """The kernel stages a primitive with no registered lowering rule.

    Distinct from EmitError proper: the primitive itself is missing, not
    an unsupported case of an existing rule. Extensible via `rule`.

    Attributes
    ----------
    primitive : str or None
        Name of the missing primitive, for callers branching on it.
    """

    def __init__(self, *args: object, primitive: str | None = None) -> None:
        super().__init__(*args)
        self.primitive = primitive


class StackOverflowError(EmitError):
    """Per-instance thread-local storage exceeds the per-thread stack.

    Every loaded block and intermediate lives in thread-local memory, so
    a kernel's live arrays are bounded by a few KB. Raised either
    pre-flight from `emit_msl`'s own accounting or by translating
    Metal's opaque pipeline-creation failure.

    Attributes
    ----------
    stack_bytes : int or None
        Estimated per-thread bytes the kernel declares, when known.
    limit : int or None
        Budget it was measured against, when known.
    """

    def __init__(
        self,
        *args: object,
        stack_bytes: int | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(*args)
        self.stack_bytes = stack_bytes
        self.limit = limit


class DispatchError(PalladiumError, TypeError):
    """A compiled kernel was called with arguments that do not match its
    traced spec (wrong count, shape, or dtype)."""
