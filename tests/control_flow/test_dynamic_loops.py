"""Dynamic-bound fori_loop behavior, including empty/reversed ranges."""

import jax
import jax.numpy as jnp
import numpy as np

import palladium


def test_bare_dynamic_fori_loop():
    def kernel(bounds, output):
        # Dynamic lower/upper bounds, an empty loop, and signed floor/modulo.
        lo, hi = bounds[0], bounds[1]
        output[0] = jax.lax.fori_loop(lo, hi, lambda i, total: total + i // 3 + i % 3, jnp.int32(0))

    call = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((1,), jnp.int32))
    for lo, hi in [(-7, 9), (2, 7), (4, 4), (7, 2)]:
        actual = call(np.array([lo, hi], np.int32))
        assert int(actual[0]) == sum(i // 3 + i % 3 for i in range(lo, hi))
