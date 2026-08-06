"""jaxpr -> MSL emission.

Two execution models share one machinery (`core`): the default
one-thread-per-instance rules (`rules`) and the simdgroup-cooperative
rules (`coop`), selected per kernel by `is_simdgroup_cooperative`.
"""

from palladium.emit import rules as _rules  # noqa: F401  (registers RULES)
from palladium.emit.coop import (
    COOP_RULES,
    coop_rows,
    coop_rule,
    emit_jaxpr_coop,
    is_simdgroup_cooperative,
)
from palladium.emit.core import (
    RULES,
    CVal,
    EmitError,
    EmitState,
    emit_jaxpr,
    emit_msl,
    rule,
)

__all__ = [
    "COOP_RULES",
    "RULES",
    "CVal",
    "EmitError",
    "EmitState",
    "coop_rows",
    "coop_rule",
    "emit_jaxpr",
    "emit_jaxpr_coop",
    "emit_msl",
    "is_simdgroup_cooperative",
    "rule",
]
