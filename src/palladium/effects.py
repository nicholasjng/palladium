"""Jaxpr effect helpers: per-Ref read/write sets, and the gate against
effects palladium cannot perform on the GPU.

State effects (`Read`/`Write`/`Accum`, each keyed to a Ref var)
aggregate per jaxpr and through control flow: a `swap` inside a `scan`
body surfaces in the outer equation's effects.
"""

from __future__ import annotations

from jax._src.effects import Effect
from jax._src.state.types import AccumEffect, RefEffect, WriteEffect
from jax.extend.core import Jaxpr, JaxprEqn, Var

__all__ = [
    "eqn_reads_ref",
    "eqn_writes_ref",
    "foreign_effects",
    "read_refs",
    "written_refs",
]


def foreign_effects(jaxpr: Jaxpr) -> list[Effect]:
    """Effects in `jaxpr` that are not Ref state effects, in a
    deterministic order."""
    return sorted(
        (e for e in jaxpr.effects if not isinstance(e, RefEffect)),
        key=lambda e: (type(e).__name__, str(e)),
    )


def written_refs(jaxpr: Jaxpr) -> set[Var]:
    """Ref vars `jaxpr` may write (swap or accumulate), at any depth."""
    return {
        e.input
        for e in jaxpr.effects
        if isinstance(e, (WriteEffect, AccumEffect)) and isinstance(e.input, Var)
    }


def read_refs(jaxpr: Jaxpr) -> set[Var]:
    """Ref vars `jaxpr` may read, at any depth."""
    return {
        e.input
        for e in jaxpr.effects
        if isinstance(e, RefEffect)
        and not isinstance(e, (WriteEffect, AccumEffect))
        and isinstance(e.input, Var)
    }


def eqn_reads_ref(eqn: JaxprEqn, ref: Var) -> bool:
    """Whether `eqn` may read `ref`, including inside sub-jaxprs."""
    return any(
        isinstance(e, RefEffect)
        and not isinstance(e, (WriteEffect, AccumEffect))
        and e.input is ref
        for e in eqn.effects
    )


def eqn_writes_ref(eqn: JaxprEqn, ref: Var) -> bool:
    """Whether `eqn` may write `ref`, including inside sub-jaxprs."""
    return any(
        isinstance(e, (WriteEffect, AccumEffect)) and e.input is ref
        for e in eqn.effects
    )
