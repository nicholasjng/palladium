"""jaxpr -> MSL emission: `core` (lowering IR), `rules` (per-primitive
lowerings), `addressing`, `tensorops`, and `kernel` (assembly)."""

from palladium.emit.core import EmitStats
from palladium.emit.kernel import emit_msl, emit_msl_stats

__all__ = ["EmitStats", "emit_msl", "emit_msl_stats"]
