"""`with_vjp`, run on the Pallas interpreter."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium


def _add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def _add_vjp_kernel(x_ref, y_ref, cotangent_ref, x_gradient_ref, y_gradient_ref):
    del x_ref, y_ref
    x_gradient_ref[...] = cotangent_ref[...]
    y_gradient_ref[...] = cotangent_ref[...]


def _add_with_residual_kernel(x_ref, y_ref, output_ref, residual_ref):
    output_ref[...] = x_ref[...] + y_ref[...]
    residual_ref[...] = x_ref[...]


def _add_with_residual_vjp_kernel(
    x_ref, y_ref, residual_ref, cotangent_ref, x_gradient_ref, y_gradient_ref
):
    del x_ref, y_ref, residual_ref
    x_gradient_ref[...] = cotangent_ref[...]
    y_gradient_ref[...] = cotangent_ref[...]


def _interpreted(kernel, out_shape):
    return pl.pallas_call(kernel, out_shape=out_shape, interpret=True)


def test_vjp_accepts_a_pallas_backward_kernel():
    forward = _interpreted(_add_kernel, jax.ShapeDtypeStruct((8,), jnp.float32))
    backward = _interpreted(
        _add_vjp_kernel,
        (jax.ShapeDtypeStruct((8,), jnp.float32), jax.ShapeDtypeStruct((8,), jnp.float32)),
    )
    call = palladium.with_vjp(forward, backward)
    x = jnp.arange(8, dtype=jnp.float32)
    y = jnp.ones(8, dtype=jnp.float32)
    loss = lambda a, b: jnp.sum(call(a, b) ** 2)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jax.grad(loss, argnums=(0, 1)))(x, y)),
        np.asarray((2 * (x + y), 2 * (x + y))),
    )


def test_vjp_saves_trailing_residual_outputs_for_the_backward_kernel():
    forward = _interpreted(
        _add_with_residual_kernel,
        (jax.ShapeDtypeStruct((8,), jnp.float32), jax.ShapeDtypeStruct((8,), jnp.float32)),
    )
    backward = _interpreted(
        _add_with_residual_vjp_kernel,
        (jax.ShapeDtypeStruct((8,), jnp.float32), jax.ShapeDtypeStruct((8,), jnp.float32)),
    )
    call = palladium.with_vjp(forward, backward, residuals=1)
    x = jnp.arange(8, dtype=jnp.float32)
    y = jnp.ones(8, dtype=jnp.float32)
    assert np.asarray(call(x, y)).shape == (8,)
    loss = lambda a, b: jnp.sum(call(a, b) ** 2)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jax.grad(loss, argnums=(0, 1)))(x, y)),
        np.asarray((2 * (x + y), 2 * (x + y))),
    )
    with pytest.raises(ValueError, match="leaving none"):
        palladium.with_vjp(forward, backward, residuals=2)(x, y)
