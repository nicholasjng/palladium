"""Palladium-defined validation errors, grouped by trace, emit, and dispatch stage.

JAX, option validation, Metal compilation, and runtime execution may raise
additional exception types. TraceError is also a ValueError and DispatchError
is also a TypeError.
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
    """Represent a Palladium tracing, emission, or dispatch validation failure.

    Parameters
    ----------
    *args : object
        Exception message and optional context forwarded to the base exception.

    Notes
    -----
    Catch this base class to handle Palladium-defined errors. JAX validation, native
    compilation, and runtime libraries may raise their own exceptions.
    """


class TraceError(PalladiumError, ValueError):
    """Report a Pallas call that cannot be normalized into a kernel specification.

    Parameters
    ----------
    *args : object
        Exception message and optional context forwarded to the base exception.

    Notes
    -----
    Also a ValueError. Covers unsupported call structure, effects, block dimensions, invalid
    aliases, and obvious parallel write races.
    """


class EmitError(PalladiumError):
    """Report a kernel construct that cannot be emitted as valid Metal source.

    Parameters
    ----------
    *args : object
        Exception message and optional context forwarded to the base exception.

    Notes
    -----
    Covers unsupported primitive cases, shapes, addressing, and resource validation. Missing
    primitive rules use UnsupportedPrimitiveError.
    """


class UnsupportedPrimitiveError(EmitError, NotImplementedError):
    """Report a staged JAX primitive with no registered MSL lowering.

    Parameters
    ----------
    *args : object
        Exception message and optional context.
    primitive : str or None, optional
        Missing primitive name, default None.

    Attributes
    ----------
    primitive : str or None
        Name for callers to inspect without parsing the message.

    Notes
    -----
    Also an EmitError and NotImplementedError. Unsupported parameter combinations of an
    existing rule raise EmitError instead. Emitter extensions can register a new rule with
    ``palladium.emit.core.rule``.
    """

    def __init__(self, *args: object, primitive: str | None = None) -> None:
        super().__init__(*args)
        self.primitive = primitive


class DispatchError(PalladiumError, TypeError):
    """Report an input dtype unsupported by the CPU-to-Metal callable.

    Parameters
    ----------
    *args : object
        Exception message and optional context forwarded to the base exception.

    Notes
    -----
    Also a TypeError. Native Metal compilation and dispatch failures are reported by the
    runtime rather than this dtype-validation exception.
    """
