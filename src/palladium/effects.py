"""Jaxpr effect helpers: per-equation Ref read/write queries, and the gate
against effects palladium cannot perform on the GPU.

Ref state effects aggregate through control flow: a `swap` inside a `scan`
body surfaces in the outer equation's effects. Cooperative primitives carry
a `GpuNativeEffect` so JAX's DCE keeps them; the gate lets those through.
"""

from __future__ import annotations

from jax._src.effects import Effect
from jax._src.state.types import AccumEffect, RefEffect, WriteEffect
from jax.extend.core import Jaxpr, JaxprEqn, Var

__all__ = [
    "GpuNativeEffect",
    "eqn_reads_ref",
    "eqn_writes_ref",
    "foreign_effects",
]


class GpuNativeEffect(Effect):
    """Base for effects palladium lowers to GPU instructions. Subclass it for
    a primitive that must survive DCE but has no Ref effect to declare
    (`barrier()`, thread-position builtins); any other non-Ref effect is
    rejected at trace time."""


def foreign_effects(jaxpr: Jaxpr) -> list[Effect]:
    """Effects in `jaxpr` that palladium cannot perform on the GPU (anything
    but Ref state effects and `GpuNativeEffect`), in a deterministic order."""
    return sorted(
        (e for e in jaxpr.effects if not isinstance(e, (RefEffect, GpuNativeEffect))),
        key=lambda e: (type(e).__name__, str(e)),
    )


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
    return any(isinstance(e, (WriteEffect, AccumEffect)) and e.input is ref for e in eqn.effects)
