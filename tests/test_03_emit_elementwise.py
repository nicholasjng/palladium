"""The ELEMENTWISE table and `_rule_elementwise` (emit/rules.py).

Primitives these kernels stage (check with test_01's jaxpr printing if a
name surprises you): add, sub, mul, div, max, min, exp, sin, cos, sqrt,
tanh, neg, integer_pow, convert_element_type.

Tolerances are float32-honest: kernels compile with MathMode.FAST, so
transcendentals need not be bit-identical to the CPU oracle.
"""

import jax
import jax.numpy as jnp
import numpy as np

import palladium


def _check_against_oracle(kernel, args, rtol=1e-5, atol=1e-6, **shapes):
    f = palladium.metal_call(kernel, **shapes)
    got = f(*args)
    want = np.asarray(f.interpret(*args))
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


def test_saxpy_with_literal(rng):
    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = 2.5 * x_ref[...] + y_ref[...]

    x = rng.standard_normal(512, dtype=np.float32)
    y = rng.standard_normal(512, dtype=np.float32)
    _check_against_oracle(
        kernel, (x, y), out_shape=jax.ShapeDtypeStruct((512,), jnp.float32)
    )


def test_binary_zoo(rng):
    def kernel(x_ref, y_ref, o_ref):
        x, y = x_ref[...], y_ref[...]
        o_ref[...] = jnp.maximum(x, y) * (x - y) / (jnp.minimum(x, y) - 4.0)

    x = rng.standard_normal(256, dtype=np.float32)
    y = rng.standard_normal(256, dtype=np.float32)
    _check_against_oracle(
        kernel, (x, y), out_shape=jax.ShapeDtypeStruct((256,), jnp.float32)
    )


def test_unary_zoo(rng):
    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.exp(-x * x) + jnp.sin(x) * jnp.cos(x) + jnp.tanh(x)

    x = rng.standard_normal(256, dtype=np.float32)
    _check_against_oracle(
        kernel, (x,), out_shape=jax.ShapeDtypeStruct((256,), jnp.float32), rtol=1e-4
    )


def test_integer_pow_and_sqrt(rng):
    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.sqrt(x**2 + 1.0) - x**3

    x = rng.standard_normal(128, dtype=np.float32)
    _check_against_oracle(
        kernel, (x,), out_shape=jax.ShapeDtypeStruct((128,), jnp.float32), rtol=1e-4
    )


def test_lotka_volterra_rhs(rng):
    """The capstone's right-hand side, one evaluation, no loop yet."""

    def kernel(x_ref, y_ref, o1_ref, o2_ref):
        x, y = x_ref[...], y_ref[...]
        o1_ref[...] = 1.1 * x - 0.4 * x * y
        o2_ref[...] = 0.1 * x * y - 0.4 * y

    x = rng.uniform(0.5, 2.0, 256).astype(np.float32)
    y = rng.uniform(0.5, 2.0, 256).astype(np.float32)
    f = palladium.metal_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((256,), jnp.float32),
            jax.ShapeDtypeStruct((256,), jnp.float32),
        ),
    )
    got1, got2 = f(x, y)
    want1, want2 = f.interpret(x, y)
    np.testing.assert_allclose(got1, np.asarray(want1), rtol=1e-5)
    np.testing.assert_allclose(got2, np.asarray(want2), rtol=1e-5)


def test_row_vector_broadcasts_against_matrix_matches_numpy(rng):
    """A dense layer's bias add: `x @ w + b`, `b` shape (32,) broadcasting
    against a (4, 32) matmul result. Verified against a real jaxpr before
    fixing: `b` stages `broadcast_in_dim` to (1, 32), then `add` between
    (4, 32) and (1, 32), not same-shape operands throughout, so
    `_rule_elementwise` has to broadcast per operand, not assume every
    operand matches `dst`'s shape."""

    def kernel(x_ref, w_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(x_ref[...], w_ref[...]) + b_ref[...]

    x = rng.standard_normal((4, 1), dtype=np.float32)
    w = rng.standard_normal((1, 32), dtype=np.float32)
    b = rng.standard_normal((32,), dtype=np.float32)
    f = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32)
    )
    got = f(x, w, b)
    np.testing.assert_allclose(got, x @ w + b, rtol=1e-4, atol=1e-4)


def test_column_vector_broadcasts_against_matrix_matches_numpy(rng):
    """Broadcasting along the other axis: (4, 1) against (4, 32)."""

    def kernel(x_ref, col_ref, o_ref):
        o_ref[...] = x_ref[...] + col_ref[...]

    x = rng.standard_normal((4, 32), dtype=np.float32)
    col = rng.standard_normal((4, 1), dtype=np.float32)
    f = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32)
    )
    got = f(x, col)
    np.testing.assert_allclose(got, x + col, rtol=1e-5, atol=1e-6)
