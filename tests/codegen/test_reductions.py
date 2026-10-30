"""`reduce_sum`/`reduce_max` -> nested loops, one per kept dim.

`axes` may be any subset of dims; `jnp.mean` is `reduce_sum` plus a `div`.
`lax.reduce` with any combiner keeps one accumulator per operand.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

import palladium


def test_sum_to_scalar_matches_numpy(rng):
    """Compare a full matrix sum with NumPy, including its rank-zero result."""

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...])

    x = rng.standard_normal((4, 6), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((), jnp.float32))
    got = f(x)
    np.testing.assert_allclose(got, np.sum(x), rtol=1e-4, atol=1e-4)


def test_sum_along_last_axis_matches_numpy(rng):
    """Compare row sums with NumPy to check reduction over the trailing axis."""

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], axis=1)

    x = rng.standard_normal((4, 6), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32))
    got = f(x)
    np.testing.assert_allclose(got, np.sum(x, axis=1), rtol=1e-4, atol=1e-4)


def test_sum_along_first_axis_matches_numpy(rng):
    """Compare column sums with NumPy to check reduction over the leading axis."""

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], axis=0)

    x = rng.standard_normal((4, 6), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((6,), jnp.float32))
    got = f(x)
    np.testing.assert_allclose(got, np.sum(x, axis=0), rtol=1e-4, atol=1e-4)


def test_mse_loss_matches_numpy(rng):
    """Compare a scalar mean-squared-error kernel with NumPy."""

    def kernel(pred_ref, target_ref, o_ref):
        diff = pred_ref[...] - target_ref[...]
        o_ref[...] = jnp.mean(diff**2)

    pred = rng.standard_normal((8, 4), dtype=np.float32)
    target = rng.standard_normal((8, 4), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((), jnp.float32))
    got = f(pred, target)
    want = np.mean((pred - target) ** 2)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_softmax_stability_matches_numpy(rng):
    """Compare max-subtracted softmax with NumPy on scores large enough to overflow an unshifted
    exp."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        x = x - jnp.max(x, axis=1, keepdims=True)
        e = jnp.exp(x)
        o_ref[...] = e / jnp.sum(e, axis=1, keepdims=True)

    x = (rng.standard_normal((4, 6), dtype=np.float32)) * 50.0
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((4, 6), jnp.float32))
    got = f(x)
    xs = x - x.max(axis=1, keepdims=True)
    e = np.exp(xs)
    want = e / e.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def _argmax_combiner(acc, element):
    (best, best_index), (value, index) = acc, element
    take = value > best  # ties keep the earlier index, as numpy does
    return jnp.where(take, value, best), jnp.where(take, index, best_index)


@pytest.mark.parametrize("axis", [0, 1])
def test_argmax_pair_reduce_matches_numpy(rng, axis):
    """The index update reads the old maximum, so the accumulators must
    update at once."""

    def kernel(x_ref, value_ref, index_ref):
        x = x_ref[...]
        index = lax.broadcasted_iota(jnp.int32, x.shape, axis)
        value_ref[...], index_ref[...] = lax.reduce(
            (x, index), (-jnp.inf, jnp.int32(0)), _argmax_combiner, (axis,)
        )

    x = rng.integers(0, 3, (4, 6)).astype(np.float32)  # small range: many ties
    kept = x.shape[1 - axis]
    f = palladium.metal_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((kept,), jnp.float32),
            jax.ShapeDtypeStruct((kept,), jnp.int32),
        ),
    )
    value, index = f(x)
    np.testing.assert_array_equal(value, x.max(axis=axis))
    np.testing.assert_array_equal(index, x.argmax(axis=axis))


def test_reduce_of_fused_inputs_matches_numpy(rng):
    """Both operands are elementwise values fused into the reduction."""

    def kernel(x_ref, squares_ref, positives_ref):
        x = x_ref[...]
        squares_ref[...], positives_ref[...] = lax.reduce(
            (x * x, (x > 0).astype(jnp.int32)),
            (jnp.float32(0), jnp.int32(0)),
            lambda a, b: (a[0] + b[0], a[1] + b[1]),
            (1,),
        )

    x = rng.standard_normal((4, 6), dtype=np.float32)
    f = palladium.metal_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((4,), jnp.float32),
            jax.ShapeDtypeStruct((4,), jnp.int32),
        ),
    )
    squares, positives = f(x)
    np.testing.assert_allclose(squares, (x * x).sum(axis=1), rtol=1e-5)
    np.testing.assert_array_equal(positives, (x > 0).sum(axis=1))


def test_three_operand_reduce_over_two_axes_matches_numpy(rng):
    def kernel(x_ref, y_ref, total_ref, low_ref, high_ref):
        total_ref[...], low_ref[...], high_ref[...] = lax.reduce(
            (x_ref[...], y_ref[...], y_ref[...]),
            (jnp.float32(0), jnp.inf, -jnp.inf),
            lambda a, b: (a[0] + b[0], jnp.minimum(a[1], b[1]), jnp.maximum(a[2], b[2])),
            (0, 2),
        )

    x = rng.standard_normal((3, 4, 5), dtype=np.float32)
    y = rng.standard_normal((3, 4, 5), dtype=np.float32)
    out = jax.ShapeDtypeStruct((4,), jnp.float32)
    total, low, high = palladium.metal_call(kernel, out_shape=(out, out, out))(x, y)
    np.testing.assert_allclose(total, x.sum(axis=(0, 2)), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(low, y.min(axis=(0, 2)))
    np.testing.assert_array_equal(high, y.max(axis=(0, 2)))
