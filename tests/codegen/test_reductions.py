"""`reduce_sum`/`reduce_max` -> nested loops, one per kept dim.

`axes` may be any subset of dims; `jnp.mean` is `reduce_sum` plus a `div`.
"""

import jax
import jax.numpy as jnp
import numpy as np

import palladium


def test_sum_to_scalar_matches_numpy(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...])

    x = rng.standard_normal((4, 6), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((), jnp.float32))
    got = f(x)
    np.testing.assert_allclose(got, np.sum(x), rtol=1e-4, atol=1e-4)


def test_sum_along_last_axis_matches_numpy(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], axis=1)

    x = rng.standard_normal((4, 6), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32))
    got = f(x)
    np.testing.assert_allclose(got, np.sum(x, axis=1), rtol=1e-4, atol=1e-4)


def test_sum_along_first_axis_matches_numpy(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], axis=0)

    x = rng.standard_normal((4, 6), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((6,), jnp.float32))
    got = f(x)
    np.testing.assert_allclose(got, np.sum(x, axis=0), rtol=1e-4, atol=1e-4)


def test_mean_matches_numpy(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.mean(x_ref[...])

    x = rng.standard_normal((4, 6), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((), jnp.float32))
    got = f(x)
    np.testing.assert_allclose(got, np.mean(x), rtol=1e-4, atol=1e-4)


def test_mse_loss_matches_numpy(rng):
    def kernel(pred_ref, target_ref, o_ref):
        diff = pred_ref[...] - target_ref[...]
        o_ref[...] = jnp.mean(diff**2)

    pred = rng.standard_normal((8, 4), dtype=np.float32)
    target = rng.standard_normal((8, 4), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((), jnp.float32))
    got = f(pred, target)
    want = np.mean((pred - target) ** 2)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_max_to_scalar_matches_numpy(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.max(x_ref[...])

    x = rng.standard_normal((4, 6), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((), jnp.float32))
    got = f(x)
    np.testing.assert_allclose(got, np.max(x), rtol=1e-6, atol=1e-6)


def test_max_along_last_axis_matches_numpy(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.max(x_ref[...], axis=1)

    x = rng.standard_normal((4, 6), dtype=np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32))
    got = f(x)
    np.testing.assert_allclose(got, np.max(x, axis=1), rtol=1e-6, atol=1e-6)


def test_softmax_stability_matches_numpy(rng):
    """A max-subtracted softmax matches numpy on scores large enough to overflow a raw exp."""

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
