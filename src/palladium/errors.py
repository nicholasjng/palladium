"""Palladium-defined validation errors, grouped by trace, emit, and dispatch stage.

JAX, option validation, Metal compilation, and runtime execution may raise
additional exception types.
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
    """Base of Palladium's tracing, emission, and dispatch validation failures."""


class TraceError(PalladiumError, ValueError):
    """A Pallas call cannot be normalized into a kernel specification: unsupported call
    structure, effects, block dimensions, invalid aliases, or an obvious write race."""


class EmitError(PalladiumError):
    """A kernel construct cannot be emitted as valid Metal source: an unsupported
    primitive case, shape, addressing mode, or resource limit."""


class UnsupportedPrimitiveError(EmitError, NotImplementedError):
    """A staged JAX primitive has no registered MSL lowering.

    Unsupported parameter combinations of an existing rule raise EmitError instead.
    Emitter extensions can register a rule with ``palladium.emit.core.rule``.

    Attributes
    ----------
    primitive : str or None
        The missing primitive's name, to inspect without parsing the message.
    """

    def __init__(self, *args: object, primitive: str | None = None) -> None:
        super().__init__(*args)
        self.primitive = primitive


class DispatchError(PalladiumError, TypeError):
    """An input dtype is unsupported by the CPU-to-Metal callable. Metal compilation
    and dispatch failures are reported by the runtime instead."""
