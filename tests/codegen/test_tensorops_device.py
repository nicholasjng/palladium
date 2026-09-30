"""Cooperative TensorOps kernels on the device: matmul tails, edge tiles, half
precision, and attention tiles small enough that MPP leaves slots unowned."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium
from palladium.workloads.pallas_flash_attention import (
    attention_kernel,
    attention_specs,
    reference_attention,
)

TM = 16


def _tiles(extent: int, tile: int) -> int:
    return (extent + tile - 1) // tile


@pytest.mark.parametrize(
    "m, n, k, tn, dtype, tol",
    [
        (30, 45, 144, 32, jnp.float32, 3e-3),  # K tail and output edge tiles
        (32, 64, 256, 32, jnp.float16, 2e-2),  # K loop
        (32, 64, 144, 32, jnp.bfloat16, 1e-1),  # K tail
        (32, 32, 16, 16, jnp.float32, 3e-3),  # 16x16 output tiles
    ],
)
def test_matmul(rng, m, n, k, tn, dtype, tol):

    def kernel(a_ref, b_ref, out_ref):
        out_ref[...] = jnp.matmul(a_ref[...], b_ref[...])

    call = palladium.metal_call(
        kernel,
        grid=(_tiles(m, TM), _tiles(n, tn)),
        in_specs=[
            pl.BlockSpec((TM, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, tn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((TM, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        compiler_params=palladium.CompilerParams(dot_general="tensorops"),
    )
    a = jnp.asarray(rng.standard_normal((m, k), dtype=np.float32), dtype)
    b = jnp.asarray(rng.standard_normal((k, n), dtype=np.float32), dtype)
    np.testing.assert_allclose(
        np.asarray(call(a, b), np.float32),
        np.asarray(jnp.matmul(a, b), np.float32),
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim, tile_q, tile_k", [(16, 16, 16), (16, 16, 128), (32, 32, 32)])
def test_attention(rng, head_dim, tile_q, tile_k, causal):
    shape = (1, 256, 2, head_dim)
    grid, in_specs, out_specs = attention_specs(1, 256, 2, tile_q, head_dim)
    call = palladium.metal_call(
        attention_kernel(tile_q=tile_q, tile_k=tile_k, head_dim=head_dim, causal=causal),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        compiler_params=palladium.CompilerParams(dot_general="tensorops"),
    )
    q, k, v = (rng.standard_normal(shape, dtype=np.float32) for _ in range(3))
    np.testing.assert_allclose(
        call(q, k, v), reference_attention(q, k, v, causal=causal), rtol=3e-4, atol=3e-4
    )
