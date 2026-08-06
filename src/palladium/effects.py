"""Jaxpr effect helpers: per-Ref read/write sets, and the gate against
effects palladium cannot perform on the GPU.

State effects (`Read`/`Write`/`Accum`, each keyed to a Ref var)
aggregate per jaxpr and through control flow: a `swap` inside a `scan`
body surfaces in the outer equation's effects.

Palladium's own cooperative primitives also carry effects, for a
different reason: a zero-output primitive with no declared effect is
dead code that JAX's DCE deletes before the emitter ever sees it. Those
subclass `GpuNativeEffect` so the gate lets them through -- they are
things the GPU *can* do, unlike a host callback.
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
    "read_refs",
    "written_refs",
]


class GpuNativeEffect(Effect):
    """Base for effects palladium lowers to real GPU instructions.

    Subclass this for a primitive that must survive DCE but has no Ref
    read/write to declare -- `palladium.barrier()` and the
    thread-position builtins. `foreign_effects` lets these through; any
    other non-Ref effect (host callbacks, debug prints) is still
    rejected at trace time.
    """


def foreign_effects(jaxpr: Jaxpr) -> list[Effect]:
    """Effects in `jaxpr` that palladium cannot perform on the GPU, in a
    deterministic order.

    Ref state effects are lowered as loads and stores; `GpuNativeEffect`
    subclasses are lowered by their own rules. Everything else is
    host-side and has no GPU equivalent.
    """
    return sorted(
        (e for e in jaxpr.effects if not isinstance(e, (RefEffect, GpuNativeEffect))),
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
