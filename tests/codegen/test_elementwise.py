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


def test_autodiff_cotangent_accumulation(rng):
    """Compare in-kernel autodiff of a reused input with the interpreter, exercising add_any
    lowering."""

    def kernel(x_ref, out_ref):
        out_ref[...] = jax.grad(lambda x: jnp.sum(x * x + jnp.sin(x)))(x_ref[...])

    x = rng.standard_normal(32, dtype=np.float32)
    _check_against_oracle(kernel, (x,), out_shape=jax.ShapeDtypeStruct((32,), jnp.float32))


def test_binary_zoo(rng):
    """Compare composed min, max, subtraction, multiplication, and division with the interpreter."""

    def kernel(x_ref, y_ref, o_ref):
        x, y = x_ref[...], y_ref[...]
        o_ref[...] = jnp.maximum(x, y) * (x - y) / (jnp.minimum(x, y) - 4.0)

    x = rng.standard_normal(256, dtype=np.float32)
    y = rng.standard_normal(256, dtype=np.float32)
    _check_against_oracle(kernel, (x, y), out_shape=jax.ShapeDtypeStruct((256,), jnp.float32))


def test_integer_pow_and_sqrt(rng):
    """Compare integer powers and square root in a floating expression with the interpreter."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.sqrt(x**2 + 1.0) - x**3

    x = rng.standard_normal(128, dtype=np.float32)
    _check_against_oracle(
        kernel, (x,), out_shape=jax.ShapeDtypeStruct((128,), jnp.float32), rtol=1e-4
    )


def test_row_vector_broadcasts_against_matrix_matches_numpy(rng):
    """Check that a row bias broadcasts across all rows of a matrix-product result."""

    def kernel(x_ref, w_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(x_ref[...], w_ref[...]) + b_ref[...]

    x = rng.standard_normal((4, 1), dtype=np.float32)
    w = rng.standard_normal((1, 32), dtype=np.float32)
    b = rng.standard_normal((32,), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32))
    got = f(x, w, b)
    np.testing.assert_allclose(got, x @ w + b, rtol=1e-4, atol=1e-4)


def test_column_vector_broadcasts_against_matrix_matches_numpy(rng):
    """Check that a column vector broadcasts across all columns of a matrix."""

    def kernel(x_ref, col_ref, o_ref):
        o_ref[...] = x_ref[...] + col_ref[...]

    x = rng.standard_normal((4, 32), dtype=np.float32)
    col = rng.standard_normal((4, 1), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32))
    got = f(x, col)
    np.testing.assert_allclose(got, x + col, rtol=1e-5, atol=1e-6)


def test_transcendental_zoo_matches_the_oracle(rng):
    """Compare composed MSL-backed transcendental, rounding, and finite-check operations with the
    interpreter."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = (
            jax.lax.rsqrt(x * x + 1.0)
            + jnp.exp(-x * x)
            + jnp.sin(x) * jnp.cos(x)
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
    """Check nearest-even and away-from-zero rounding at positive and negative half-integer ties."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.round(x) * 4.0 + jax.lax.round(x)

    x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 0.49, -0.51], dtype=np.float32)
    _check_against_oracle(kernel, (x,), out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))


def test_helper_backed_ops_match_the_oracle(rng):
    """Check helper-backed numerical results and verify that the erf helper is emitted once before
    the kernel."""

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
    """Compare expm1 plus log1p near zero with a float64 reference under SAFE math."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.expm1(x) + jnp.log1p(x)

    x = np.array([1e-8, -1e-8, 1e-5, -1e-5, 1e-3], dtype=np.float32)
    import metal_runtime as mr

    call = palladium.metal_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((5,), jnp.float32),
        compiler_params=palladium.CompilerParams(math_mode=mr.MathMode.SAFE),
    )
    want = np.expm1(x.astype(np.float64)) + np.log1p(x.astype(np.float64))
    np.testing.assert_allclose(call(x), want.astype(np.float32), rtol=1e-6)
