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
    """Preserve tuple outputs and wrap any other value as a single output; do not flatten pytrees."""
    return value if isinstance(value, tuple) else (value,)


def with_vjp(forward: Callable, backward: Callable, *, residuals: int = 0) -> Callable:
    """Attach an explicit reverse-mode derivative to a forward callable.

    Parameters
    ----------
    forward : callable
        JAX-compatible forward function accepting positional primals and returning one
        array or a tuple of outputs.
    backward : callable
        Receives the original primals, saved residual outputs, then one cotangent per
        public output. Returns one cotangent per primal in the same order.
    residuals : int, optional
        Number of trailing forward outputs to save for the backward pass and hide from
        the public result. Defaults to zero.

    Returns
    -------
    callable
        Function decorated with ``jax.custom_vjp``. With residuals, one remaining public
        output is unwrapped from its tuple.

    Raises
    ------
    ValueError
        ``residuals`` is negative, or at call time leaves no public forward outputs.
    TypeError
        The backward callable returns the wrong number of input cotangents.

    Notes
    -----
    Forward and backward may be Pallas, FFI, or ordinary JAX callables. This supplies
    reverse mode only; JAX forward-mode differentiation through ``custom_vjp`` is
    unsupported. The calling convention expects arrays or flat tuples of outputs; arbitrary
    output pytrees are not flattened.
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
