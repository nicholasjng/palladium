"""Jaxpr effect helpers: per-equation Ref read/write queries, and the gate
against effects palladium cannot perform on the GPU.

Ref state effects aggregate through control flow: a `swap` inside a `scan`
body surfaces in the outer equation's effects.
"""

from __future__ import annotations

from jax._src.effects import Effect
from jax._src.state.types import AccumEffect, RefEffect, WriteEffect
from jax.extend.core import Jaxpr, JaxprEqn, Var

__all__ = ["eqn_reads_ref", "eqn_writes_ref", "foreign_effects"]


def foreign_effects(jaxpr: Jaxpr) -> list[Effect]:
    """Find effects that cannot execute inside a Palladium GPU kernel.

    Parameters
    ----------
    jaxpr : Jaxpr
        Kernel body whose aggregate effect set includes effects from nested control
        flow.

    Returns
    -------
    list of Effect
        Non-Ref effects sorted by type name and string representation; empty for a pure
        Ref-state kernel.

    Notes
    -----
    Ref effects are allowed by this gate; primitive and addressing support are validated
    during emission.
    """
    return sorted(
        (e for e in jaxpr.effects if not isinstance(e, RefEffect)),
        key=lambda e: (type(e).__name__, str(e)),
    )


def eqn_reads_ref(eqn: JaxprEqn, ref: Var) -> bool:
    """Determine whether an equation may read a particular Ref.

    Parameters
    ----------
    eqn : JaxprEqn
        Staged equation to inspect or lower.
    ref : Var
        Ref variable matched by identity against the equation effect set.

    Returns
    -------
    bool
        True when an aggregate Ref effect identifies a read, including nested control
        flow.

    Notes
    -----
    Write and accumulation effects are excluded from the read query. The query uses staged
    effects rather than inspecting generated source.
    """
    return any(
        isinstance(e, RefEffect)
        and not isinstance(e, (WriteEffect, AccumEffect))
        and e.input is ref
        for e in eqn.effects
    )


def eqn_writes_ref(eqn: JaxprEqn, ref: Var) -> bool:
    """Determine whether an equation may write a particular Ref.

    Parameters
    ----------
    eqn : JaxprEqn
        Staged equation to inspect or lower.
    ref : Var
        Ref variable matched by identity against the equation effect set.

    Returns
    -------
    bool
        True when an aggregate Ref effect identifies a write, including nested control
        flow.

    Notes
    -----
    Both write and accumulation effects count as writes. The query uses staged effects
    rather than inspecting generated source.
    """
    return any(isinstance(e, (WriteEffect, AccumEffect)) and e.input is ref for e in eqn.effects)
