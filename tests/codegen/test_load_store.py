"""Block load/store: `_rule_get` and `_rule_swap` round-trip a block exactly."""

import jax
import jax.numpy as jnp
import numpy as np

import palladium


def _copy_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]


def test_copy_roundtrip_2d(rng):
    """Verify that loading and storing a two-dimensional block preserves every element exactly."""
    f = palladium.metal_call(_copy_kernel, out_shape=jax.ShapeDtypeStruct((16, 32), jnp.float32))
    x = rng.standard_normal((16, 32), dtype=np.float32)
    np.testing.assert_array_equal(f(x), x)
