"""Stretch 16: `transpose` -> a materialized, permuted copy.

Verified against a real jaxpr before implementing (see
`_rule_transpose`'s docstring): `a.T` inside a dot product stages as its
own `transpose` equation, so `dot_general` never needs widening for a
dense layer's backward pass, `transpose` is the actual missing piece.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import palladium

pytestmark = pytest.mark.exercise


def test_transpose_matches_numpy(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...].T

    x = rng.standard_normal((4, 6), dtype=np.float32)
    f = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((6, 4), jnp.float32)
    )
    got = f(x)
    np.testing.assert_array_equal(got, x.T)


def test_square_transpose_matches_numpy(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...].T

    x = rng.standard_normal((5, 5), dtype=np.float32)
    f = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((5, 5), jnp.float32)
    )
    got = f(x)
    np.testing.assert_array_equal(got, x.T)


def test_lhs_transposed_matmul_matches_numpy(rng):
    """dL/dW = x.T @ dy: the actual backward-pass shape for a dense
    layer's weight gradient."""

    def kernel(x_ref, dy_ref, o_ref):
        o_ref[...] = jnp.dot(x_ref[...].T, dy_ref[...])

    x = rng.standard_normal((8, 4), dtype=np.float32)
    dy = rng.standard_normal((8, 6), dtype=np.float32)
    f = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((4, 6), jnp.float32)
    )
    got = f(x, dy)
    np.testing.assert_allclose(got, x.T @ dy, rtol=1e-4, atol=1e-4)


def test_rhs_transposed_matmul_matches_numpy(rng):
    """dL/dx = dy @ W.T: the actual backward-pass shape for a dense
    layer's input gradient."""

    def kernel(dy_ref, w_ref, o_ref):
        o_ref[...] = jnp.dot(dy_ref[...], w_ref[...].T)

    dy = rng.standard_normal((8, 6), dtype=np.float32)
    w = rng.standard_normal((4, 6), dtype=np.float32)
    f = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((8, 4), jnp.float32)
    )
    got = f(dy, w)
    np.testing.assert_allclose(got, dy @ w.T, rtol=1e-4, atol=1e-4)
