"""Indexed ref access (`x_ref[i, j]`) in `_rule_get`/`_rule_swap`.

Non-Slice indices squeeze that dim into a pointer offset; Slice indices keep
the dim, with a static Python int or a dynamic Var (`pl.dslice`) as start.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

import palladium


def test_dynamic_scalar_index(rng):
    """Check that program-id-indexed row reads reconstruct the original NumPy input array."""

    def kernel(x_ref, o_ref):
        i = pl.program_id(0)
        o_ref[0, :] = x_ref[i, :]

    f = palladium.metal_call(
        kernel,
        grid=(3,),
        in_specs=[pl.BlockSpec((3, 4), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((1, 4), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((3, 4), jnp.float32),
    )
    x = rng.standard_normal((3, 4), dtype=np.float32)
    np.testing.assert_allclose(f(x), x, rtol=1e-6)


def test_static_literal_index(rng):
    """Compare the difference of two statically indexed rows with NumPy."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[1, :] - x_ref[0, :]

    f = palladium.metal_call(
        kernel,
        out_specs=pl.BlockSpec((4,), lambda: (0,)),
        out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
    )
    x = rng.standard_normal((3, 4), dtype=np.float32)
    np.testing.assert_allclose(f(x), x[1] - x[0], rtol=1e-6)


def test_fully_scalar_indexed_get_and_swap(rng):
    """Compare scalar-indexed loads and stores, including in-range neighbor reads, with the
    interpreter."""

    def kernel(x_ref, o_ref):
        i, j = pl.program_id(0), pl.program_id(1)
        n = 4
        # Clamp with `where` so only comparisons and select_n are staged.
        j_left = jnp.where(j > 0, j - 1, 0)
        j_right = jnp.where(j < n - 1, j + 1, n - 1)
        left = jnp.where(j > 0, x_ref[i, j_left], 0.0)
        right = jnp.where(j < n - 1, x_ref[i, j_right], 0.0)
        o_ref[i, j] = 2.0 * x_ref[i, j] - left - right

    f = palladium.metal_call(
        kernel,
        grid=(4, 4),
        in_specs=[pl.BlockSpec((4, 4), lambda i, j: (0, 0))],
        out_specs=pl.BlockSpec((4, 4), lambda i, j: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((4, 4), jnp.float32),
    )
    x = rng.standard_normal((4, 4), dtype=np.float32)
    got = f(x)
    want = np.asarray(f.interpret(x))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_periodic_wraparound_index(rng):
    """Compare periodic neighbor reads with NumPy rolls to check wraparound index arithmetic."""

    def kernel(x_ref, o_ref):
        i, j = pl.program_id(0), pl.program_id(1)
        n = 4
        ip = jnp.where(i == n - 1, 0, i + 1)
        im = jnp.where(i != 0, i - 1, n - 1)
        o_ref[i, j] = x_ref[ip, j] + x_ref[im, j]

    f = palladium.metal_call(
        kernel,
        grid=(4, 4),
        in_specs=[pl.BlockSpec((4, 4), lambda i, j: (0, 0))],
        out_specs=pl.BlockSpec((4, 4), lambda i, j: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((4, 4), jnp.float32),
    )
    x = rng.standard_normal((4, 4), dtype=np.float32)
    got = f(x)
    want = np.roll(x, -1, 0) + np.roll(x, 1, 0)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_dynamic_partial_leading_slice(rng):
    """Compare dynamic-start, fixed-size leading-axis slices with the interpreter."""

    def kernel(x_ref, o_ref):
        i = pl.program_id(0)
        o_ref[...] = x_ref[pl.dslice(i, 2), :]

    f = palladium.metal_call(
        kernel,
        grid=(3,),
        in_specs=[pl.BlockSpec((5, 4), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((2, 4), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((6, 4), jnp.float32),
    )
    x = rng.standard_normal((5, 4), dtype=np.float32)
    got = f(x)
    want = np.asarray(f.interpret(x))
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_noncontiguous_kept_dims():
    """Check that a partial trailing-axis slice addresses noncontiguous source storage correctly."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[:, :, 0:1]

    f = palladium.metal_call(
        kernel,
        out_specs=pl.BlockSpec((4, 3, 1), lambda: (0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((4, 3, 1), jnp.float32),
    )
    x = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    np.testing.assert_array_equal(f(x), x[:, :, 0:1])
