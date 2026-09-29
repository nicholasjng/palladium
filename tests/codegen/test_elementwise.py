"""The ELEMENTWISE table and `_rule_elementwise`.

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
    _check_against_oracle(kernel, (x, y), out_shape=jax.ShapeDtypeStruct((512,), jnp.float32))


def test_autodiff_cotangent_accumulation(rng):
    def kernel(x_ref, out_ref):
        out_ref[...] = jax.grad(lambda x: jnp.sum(x * x + jnp.sin(x)))(x_ref[...])

    x = rng.standard_normal(32, dtype=np.float32)
    _check_against_oracle(kernel, (x,), out_shape=jax.ShapeDtypeStruct((32,), jnp.float32))


def test_binary_zoo(rng):
    def kernel(x_ref, y_ref, o_ref):
        x, y = x_ref[...], y_ref[...]
        o_ref[...] = jnp.maximum(x, y) * (x - y) / (jnp.minimum(x, y) - 4.0)

    x = rng.standard_normal(256, dtype=np.float32)
    y = rng.standard_normal(256, dtype=np.float32)
    _check_against_oracle(kernel, (x, y), out_shape=jax.ShapeDtypeStruct((256,), jnp.float32))


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
    """One evaluation of the Lotka-Volterra right-hand side."""

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
    """A (32,) bias stages `broadcast_in_dim` to (1, 32) before the add, so the elementwise rule must broadcast per operand."""

    def kernel(x_ref, w_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(x_ref[...], w_ref[...]) + b_ref[...]

    x = rng.standard_normal((4, 1), dtype=np.float32)
    w = rng.standard_normal((1, 32), dtype=np.float32)
    b = rng.standard_normal((32,), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32))
    got = f(x, w, b)
    np.testing.assert_allclose(got, x @ w + b, rtol=1e-4, atol=1e-4)


def test_column_vector_broadcasts_against_matrix_matches_numpy(rng):
    """Broadcasting along the other axis: (4, 1) against (4, 32)."""

    def kernel(x_ref, col_ref, o_ref):
        o_ref[...] = x_ref[...] + col_ref[...]

    x = rng.standard_normal((4, 32), dtype=np.float32)
    col = rng.standard_normal((4, 1), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32))
    got = f(x, col)
    np.testing.assert_allclose(got, x + col, rtol=1e-5, atol=1e-6)


def test_transcendental_zoo_matches_the_oracle(rng):
    """Ops that map onto MSL builtins directly."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = (
            jax.lax.rsqrt(x * x + 1.0)
            + jax.nn.sigmoid(x)
            + jnp.exp2(x)
            + jnp.square(x)
            + jnp.floor(x)
            + jnp.ceil(x)
            + jnp.arctan2(x, 1.5)
            + jnp.tan(x * 0.5)
            + jnp.arcsin(jnp.tanh(x))
            + jnp.arccos(jnp.tanh(x))
            + jnp.arctan(x)
            + jnp.sinh(x)
            + jnp.cosh(x)
            + jnp.arcsinh(x)
            + jnp.arctanh(0.5 * jnp.tanh(x))
            + jnp.isfinite(x).astype(x.dtype)
        )

    x = rng.standard_normal(256, dtype=np.float32)
    _check_against_oracle(
        kernel, (x,), out_shape=jax.ShapeDtypeStruct((256,), jnp.float32), rtol=1e-4
    )


def test_rounding_modes_match_the_oracle():
    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.round(x) * 4.0 + jax.lax.round(x)

    x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 0.49, -0.51], dtype=np.float32)
    _check_against_oracle(kernel, (x,), out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))


def test_helper_backed_ops_match_the_oracle(rng):
    """erf, erf_inv, expm1, and log1p have no MSL builtin; they are
    emitted once as helper functions above the kernel."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        u = 0.9 * jnp.tanh(x)
        o_ref[...] = jax.lax.erf(x) + jax.lax.erf_inv(u) + jnp.expm1(0.1 * x) + jnp.log1p(x * x)

    x = rng.standard_normal(256, dtype=np.float32)
    _check_against_oracle(
        kernel, (x,), out_shape=jax.ShapeDtypeStruct((256,), jnp.float32), rtol=1e-4, atol=1e-5
    )
    msl = palladium.debug_msl(
        kernel,
        jax.ShapeDtypeStruct((256,), jnp.float32),
        out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
    )
    assert msl.count("inline float pd_erf(") == 1
    assert msl.index("inline float pd_erf(") < msl.index("kernel void")


def test_expm1_and_log1p_keep_precision_near_zero():
    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.expm1(x) + jnp.log1p(x)

    x = np.array([1e-8, -1e-8, 1e-5, -1e-5, 1e-3], dtype=np.float32)
    import metal_runtime as mr

    call = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((5,), jnp.float32), math_mode=mr.MathMode.SAFE
    )
    want = np.expm1(x.astype(np.float64)) + np.log1p(x.astype(np.float64))
    np.testing.assert_allclose(call(x), want.astype(np.float32), rtol=1e-6)
