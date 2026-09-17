"""Numerical semantics of sign/rem and dynamic-bound Pallas loops."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from metal_runtime import MathMode

import palladium


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32, np.uint32])
def test_sign_and_remainder(dtype):
    def kernel(x, y, sign, remainder):
        sign[...] = jax.lax.sign(x[...])
        remainder[...] = jax.lax.rem(x[...], y[...])

    if np.issubdtype(dtype, np.floating):
        x = np.array([-np.inf, -5.5, -0.0, 0.0, 5.5, np.inf, np.nan, 3.0], dtype)
        y = np.array([2, 2, 2, -2, -2, 2, 2, 0], dtype)
    elif dtype == np.int32:
        x = np.array([-2147483648, -5, -4, 0, 4, 5, 17, 3], dtype)
        y = np.array([-1, 2, -3, 2, -3, 2, 0, -1], dtype)
    else:
        x = np.array([0, 1, 2, 5, 17, 4294967295, 4294967294, 3], dtype)
        y = np.array([2, 2, 3, 2, 0, 4294967295, 4294967295, 1], dtype)
    # SAFE is required for meaningful NaN/signed-zero behavior on Metal.
    call = palladium.metal_call(
        kernel,
        math_mode=MathMode.SAFE,
        out_shape=(jax.ShapeDtypeStruct(x.shape, dtype),) * 2,
    )
    actual = call(x, y)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = (
            jax.lax.sign(jnp.asarray(x)),
            jax.lax.rem(jnp.asarray(x), jnp.asarray(y)),
        )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
    if np.issubdtype(dtype, np.floating):
        np.testing.assert_array_equal(np.signbit(actual[0][2:4]), np.signbit(x[2:4]))


def test_bare_dynamic_fori_loop():
    def kernel(bounds, output):
        # Dynamic lower/upper bounds, an empty loop, and signed floor/modulo.
        lo, hi = bounds[0], bounds[1]
        output[0] = jax.lax.fori_loop(
            lo, hi, lambda i, total: total + i // 3 + i % 3, jnp.int32(0)
        )

    call = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((1,), jnp.int32))
    for lo, hi in [(-7, 9), (2, 7), (4, 4), (7, 2)]:
        actual = call(np.array([lo, hi], np.int32))
        assert int(actual[0]) == sum(i // 3 + i % 3 for i in range(lo, hi))
