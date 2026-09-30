"""Gradients for Palladium calls. No path derives a derivative from emitted
MSL, so a forward call is paired with a backward implementation through
`jax.custom_vjp`. Forward and backward are ordinary JAX callables: a plain
`pl.pallas_call`, `metal_call`, or any JAX function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

__all__ = ["with_vjp"]


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def with_vjp(forward: Callable, backward: Callable, *, residuals: int = 0) -> Callable:
    """Pair `forward` with an explicit backward callable.

    The last `residuals` outputs of `forward` (checkpoints, say) are kept
    for the backward pass and not returned. `backward` receives the forward
    primals, then those residuals, then one cotangent per returned output,
    and returns one cotangent per primal, in order.
    """
    if residuals < 0:
        raise ValueError("residuals must be nonnegative")

    def split(raw_outputs):
        values = _as_tuple(raw_outputs)
        if not residuals:
            return raw_outputs, ()
        if residuals >= len(values):
            raise ValueError(
                f"forward returned {len(values)} outputs, leaving none after {residuals} residuals"
            )
        public = values[:-residuals]
        return (public[0] if len(public) == 1 else public), values[-residuals:]

    @jax.custom_vjp
    def differentiated(*args):
        return split(forward(*args))[0]

    def fwd(*args):
        public, saved = split(forward(*args))
        return public, (*args, *saved)

    def bwd(saved, cotangents):
        input_cotangents = _as_tuple(backward(*saved, *_as_tuple(cotangents)))
        if len(input_cotangents) != len(saved) - residuals:
            raise TypeError(
                f"backward returned {len(input_cotangents)} cotangents for "
                f"{len(saved) - residuals} primals"
            )
        return input_cotangents

    differentiated.defvjp(fwd, bwd)
    return differentiated
