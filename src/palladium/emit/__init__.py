"""jaxpr -> MSL emission.

One-thread-per-instance rules (`rules`), lowered over shared machinery (`core`).
"""

from palladium.emit import rules as _rules  # noqa: F401  (registers RULES)
from palladium.emit.core import (
    RULES,
    Cursor,
    CVal,
    EmitError,
    Environment,
    emit_jaxpr,
    emit_msl,
    rule,
)

__all__ = [
    "RULES",
    "CVal",
    "Cursor",
    "EmitError",
    "Environment",
    "emit_jaxpr",
    "emit_msl",
    "rule",
]
