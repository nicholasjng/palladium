"""Jaxpr effect helpers: per-equation Ref read/write queries, and the gate
against effects palladium cannot perform on the GPU.

State effects (`Read`/`Write`/`Accum`, each keyed to a Ref var)
aggregate per jaxpr and through control flow: a `swap` inside a `scan`
body surfaces in the outer equation's effects.

Palladium's cooperative primitives carry effects for a different
reason: a zero-output primitive with no declared effect is dead code
that JAX's DCE deletes before the emitter sees it. Those subclass
`GpuNativeEffect`, which the gate lets through.
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
    """Base for effects palladium lowers to real GPU instructions.

    Subclass this for a primitive that must survive DCE but has no Ref
    read/write to declare (`barrier()`, the thread-position builtins).
    Any other non-Ref effect (host callbacks, debug prints) is rejected
    at trace time.
    """


def foreign_effects(jaxpr: Jaxpr) -> list[Effect]:
    """Effects in `jaxpr` that palladium cannot perform on the GPU, in a
    deterministic order.

    Ref state effects lower to loads and stores and `GpuNativeEffect`
    subclasses to their own rules; everything else is host-side.
    """
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
