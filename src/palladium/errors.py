"""Exception hierarchy: everything palladium raises derives from
PalladiumError, split by stage (trace, emit, dispatch). TraceError also
subclasses ValueError, DispatchError TypeError.
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
    """The kernel stages a primitive with no registered lowering rule (the
    primitive is missing, not an unsupported case of an existing rule).
    Extensible via `rule`.

    Attributes
    ----------
    primitive : str or None
        Name of the missing primitive, for callers branching on it.
    """

    def __init__(self, *args: object, primitive: str | None = None) -> None:
        super().__init__(*args)
        self.primitive = primitive


class DispatchError(PalladiumError, TypeError):
    """A `metal_call` argument has a dtype palladium cannot lower."""
