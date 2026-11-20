"""Jaxpr effect helpers: per-equation Ref read/write queries, and the gate
against effects palladium cannot perform on the GPU.

Ref state effects aggregate through control flow: a `swap` inside a `scan`
body surfaces in the outer equation's effects.
"""

from __future__ import annotations

from jax._src.state.types import AccumEffect, RefEffect, WriteEffect  # no public names
from jax.extend.core import Effect, Jaxpr, JaxprEqn, Var

__all__ = ["eqn_reads_ref", "eqn_writes_ref", "foreign_effects"]


def foreign_effects(jaxpr: Jaxpr) -> list[Effect]:
    """The effects of `jaxpr`, nested control flow included, that are not Ref reads or
    writes and so cannot run in a GPU kernel, sorted by type name and string."""
    return sorted(
        (e for e in jaxpr.effects if not isinstance(e, RefEffect)),
        key=lambda e: (type(e).__name__, str(e)),
    )


def eqn_reads_ref(eqn: JaxprEqn, ref: Var) -> bool:
    """Whether `eqn` may read `ref`, directly or inside nested control flow."""
    return any(
        isinstance(e, RefEffect)
        and not isinstance(e, (WriteEffect, AccumEffect))
        and e.input is ref
        for e in eqn.effects
    )


def eqn_writes_ref(eqn: JaxprEqn, ref: Var) -> bool:
    """Whether `eqn` may write or accumulate into `ref`, directly or inside nested
    control flow."""
    return any(isinstance(e, (WriteEffect, AccumEffect)) and e.input is ref for e in eqn.effects)
