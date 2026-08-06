"""Exception hierarchy: everything palladium raises derives from
PalladiumError, split by pipeline stage (trace, emit, dispatch).

TraceError doubles as ValueError and DispatchError as TypeError so
call sites that predate the hierarchy keep their behavior.
"""

from __future__ import annotations

__all__ = [
    "DispatchError",
    "EmitError",
    "PalladiumError",
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
    """


class DispatchError(PalladiumError, TypeError):
    """A compiled kernel was called with arguments that do not match its
    traced spec (wrong count, shape, or dtype)."""
