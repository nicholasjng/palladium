"""iota, cumulative reductions, concatenate, and dynamic_slice
(emit/rules/structural.py), checked against the interpreter oracle."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import palladium


def _check(kernel, args, out_shape, rtol=1e-5, atol=1e-6):
    f = palladium.metal_call(kernel, out_shape=out_shape)
    got = f(*args)
    want = np.asarray(f.interpret(*args))
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


def test_iota_and_arange(rng):
    def kernel(x_ref, o_ref):
        rows = jax.lax.broadcasted_iota(jnp.int32, (4, 8), 0)
        cols = jax.lax.broadcasted_iota(jnp.int32, (4, 8), 1)
        o_ref[...] = x_ref[...] * rows.astype(jnp.float32) + cols.astype(jnp.float32)

    x = rng.standard_normal((4, 8), dtype=np.float32)
    _check(kernel, (x,), jax.ShapeDtypeStruct((4, 8), jnp.float32))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumsum_along_either_axis(rng, axis, reverse):
    def kernel(x_ref, o_ref):
        o_ref[...] = jax.lax.cumsum(x_ref[...], axis=axis, reverse=reverse)

    x = rng.standard_normal((6, 10), dtype=np.float32)
    _check(kernel, (x,), jax.ShapeDtypeStruct((6, 10), jnp.float32), rtol=1e-4, atol=1e-5)


def test_cumprod_cummax_cummin(rng):
    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.cumprod(x, axis=1) + jax.lax.cummax(x, axis=1) - jax.lax.cummin(x, axis=0)

    x = rng.uniform(0.5, 1.5, (5, 7)).astype(np.float32)
    _check(kernel, (x,), jax.ShapeDtypeStruct((5, 7), jnp.float32), rtol=1e-4, atol=1e-5)


def test_concatenate_along_both_dimensions(rng):
    def kernel(x_ref, y_ref, o_ref):
        x, y = x_ref[...], y_ref[...]
        top = jnp.concatenate([x, y], axis=1)  # (3, 8)
        o_ref[...] = jnp.concatenate([top, top * 2.0], axis=0)  # (6, 8)

    x = rng.standard_normal((3, 4), dtype=np.float32)
    y = rng.standard_normal((3, 4), dtype=np.float32)
    _check(kernel, (x, y), jax.ShapeDtypeStruct((6, 8), jnp.float32))


def test_dynamic_slice_clamps_like_jax(rng):
    def kernel(x_ref, i_ref, o_ref):
        i = i_ref[0]
        o_ref[...] = jax.lax.dynamic_slice(x_ref[...], (i, i * 2), (2, 3))

    x = rng.standard_normal((5, 6), dtype=np.float32)
    for start in (0, 1, 3, 7):  # 3 and 7 clamp on at least one axis
        _check(
            kernel,
            (x, np.array([start], np.int32)),
            jax.ShapeDtypeStruct((2, 3), jnp.float32),
        )


def test_systematic_resampling_building_blocks(rng):
    """The particle-filter idiom: normalized weights, cumulative sum,
    and a stratified threshold comparison."""

    def kernel(w_ref, u_ref, o_ref):
        w = w_ref[...]
        cdf = jnp.cumsum(w / jnp.sum(w))
        positions = (
            jax.lax.broadcasted_iota(jnp.int32, w.shape, 0).astype(jnp.float32) + u_ref[0]
        ) / w.shape[0]
        o_ref[...] = jnp.sum((positions[:, None] > cdf[None, :]).astype(jnp.int32), axis=1)

    w = rng.uniform(0.0, 1.0, 16).astype(np.float32)
    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((16,), jnp.int32))
    args = (w, np.array([0.37], np.float32))
    np.testing.assert_array_equal(f(*args), np.asarray(f.interpret(*args)))
