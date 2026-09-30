"""Cooperative TensorOps matmuls on the device: K tails, edge tiles, half precision."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium

TM, TN = 16, 32


def _tiles(extent: int, tile: int) -> int:
    return (extent + tile - 1) // tile


@pytest.mark.parametrize(
    "m, n, k, dtype, tol",
    [
        (30, 45, 144, jnp.float32, 3e-3),  # K tail and output edge tiles
        (32, 64, 256, jnp.float16, 2e-2),  # K loop
        (32, 64, 144, jnp.bfloat16, 1e-1),  # K tail
    ],
)
def test_matmul(rng, m, n, k, dtype, tol):
    def kernel(a_ref, b_ref, out_ref):
        out_ref[...] = jnp.matmul(a_ref[...], b_ref[...])

    call = palladium.metal_call(
        kernel,
        dot_general="tensorops",
        grid=(_tiles(m, TM), _tiles(n, TN)),
        in_specs=[
            pl.BlockSpec((TM, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, TN), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((TM, TN), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
    )
    a = jnp.asarray(rng.standard_normal((m, k), dtype=np.float32), dtype)
    b = jnp.asarray(rng.standard_normal((k, n), dtype=np.float32), dtype)
    np.testing.assert_allclose(
        np.asarray(call(a, b), np.float32),
        np.asarray(jnp.matmul(a, b), np.float32),
        rtol=tol,
        atol=tol,
    )
