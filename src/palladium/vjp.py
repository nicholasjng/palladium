"""Gradients for Palladium calls: none of the paths derives a derivative
from emitted MSL, so a forward call is paired with a backward
implementation through `jax.custom_vjp`.

The forward and backward callables are ordinary JAX callables: a plain
`pl.pallas_call` (lowered by Palladium on mps), `metal_call_jit`, or any
JAX function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

__all__ = ["with_auxiliary_vjp", "with_reference_vjp", "with_vjp"]


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _unwrap(outs):
    return outs[0] if len(outs) == 1 else outs


def with_reference_vjp(forward: Callable, reference: Callable) -> Callable:
    """Pair `forward` with the VJP JAX derives from `reference`.

    The primal stays `forward`; the pullback is JAX's VJP of `reference`,
    which must have the same inputs, outputs, and differentiable
    semantics. Useful for putting a fused forward solve into a training
    loop while a Pallas adjoint kernel is developed; it is not a
    performance solution.
    """

    @jax.custom_vjp
    def differentiated(*args):
        return forward(*args)

    def fwd(*args):
        return forward(*args), args

    def bwd(residual, cotangents):
        _, pullback = jax.vjp(reference, *residual)
        return pullback(cotangents)

    differentiated.defvjp(fwd, bwd)
    return differentiated


def with_vjp(forward: Callable, backward: Callable) -> Callable:
    """Pair `forward` with an explicit backward callable.

    `backward` receives the forward primals followed by one cotangent per
    forward output and returns one cotangent per primal, in order.
    """

    @jax.custom_vjp
    def differentiated(*args):
        return forward(*args)

    def fwd(*args):
        return forward(*args), args

    def bwd(residual, cotangents):
        input_cotangents = _as_tuple(backward(*residual, *_as_tuple(cotangents)))
        if len(input_cotangents) != len(residual):
            raise TypeError(
                "Palladium VJP returned "
                f"{len(input_cotangents)} input cotangents for {len(residual)} primals"
            )
        return input_cotangents

    differentiated.defvjp(fwd, bwd)
    return differentiated


def with_auxiliary_vjp(forward: Callable, backward: Callable, output_count: int) -> Callable:
    """Pair `forward` with `backward` while keeping trailing forward outputs
    as residuals.

    The first `output_count` outputs are the public primal result; the
    trailing outputs (checkpoints) are saved and passed to `backward` after
    the primals and before the output cotangents. One fused forward
    dispatch can so return its final state and an internal checkpoint
    buffer consumed by one fused backward dispatch.
    """
    if output_count < 1:
        raise ValueError("output_count must be positive")

    def split_outputs(raw_outputs):
        values = _as_tuple(raw_outputs)
        if output_count >= len(values):
            raise ValueError("with_auxiliary_vjp requires at least one trailing auxiliary output")
        return _unwrap(values[:output_count]), values[output_count:]

    @jax.custom_vjp
    def differentiated(*args):
        public, _ = split_outputs(forward(*args))
        return public

    def fwd(*args):
        public, auxiliaries = split_outputs(forward(*args))
        return public, (*args, *auxiliaries)

    def bwd(residual, cotangents):
        # The custom-VJP protocol validates the returned pytree against the
        # primal arguments; the auxiliary count is backward's ABI.
        return _as_tuple(backward(*residual, *_as_tuple(cotangents)))

    differentiated.defvjp(fwd, bwd)
    return differentiated
