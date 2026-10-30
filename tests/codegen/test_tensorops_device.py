"""Cooperative TensorOps kernels on the device: matmul tails, edge tiles, half
precision, attention tiles small enough that MPP leaves slots unowned, and
attention with `simdgroup_matrix`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flash_attention import (
    attention_call,
    attention_kernel,
    attention_specs,
    reference_attention,
)
from jax.experimental import pallas as pl

import palladium

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
    """Compare cooperative matmul with JAX across K tails, output edge tiles, and
    float32/half/bfloat buffers."""

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


def test_matmul_with_a_tanh_epilogue(rng):
    """Compare a cooperative matmul with a fused tanh epilogue against NumPy."""

    def kernel(a_ref, b_ref, out_ref):
        out_ref[...] = jnp.tanh(jnp.matmul(a_ref[...], b_ref[...]))

    call = palladium.metal_call(
        kernel,
        compiler_params=palladium.CompilerParams(dot_general="tensorops"),
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((TM, 32), lambda i, j: (i, 0)),
            pl.BlockSpec((32, 32), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((TM, 32), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((32, 64), jnp.float32),
    )
    a = rng.standard_normal((32, 32), dtype=np.float32) * 0.1
    b = rng.standard_normal((32, 64), dtype=np.float32) * 0.1
    np.testing.assert_allclose(call(a, b), np.tanh(a @ b), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim, tile_q, tile_k", [(16, 16, 16), (16, 16, 128), (32, 32, 32)])
def test_attention(rng, head_dim, tile_q, tile_k, causal):
    """Compare causal and noncausal cooperative attention with NumPy across small tiles and head
    dimensions."""
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


_SIMDGROUP_ATTENTION = [
    (64, 64, 16, 256, True),  # the fastest tiling at 4096 tokens on an M2
    (16, 32, 16, 256, True),
    (48, 32, 64, 256, True),
    (64, 128, 32, 256, True),  # four fragment rows per SIMD group
    (56, 32, 8, 256, True),  # SAM2's head width and 8-key tiles: no padding
    (64, 64, 16, 128, True),  # cross attention with fewer keys (noncausal only)
    (32, 32, 128, 256, False),  # K and V tiles too large to stage: matmul2d
    (64, 16, 16, 256, False),  # query tiles below 32 rows: matmul2d
]


@pytest.mark.parametrize(
    "head_dim, tile_q, tile_k, key_length, simdgroup, causal",
    [
        (*case, causal)
        for case in _SIMDGROUP_ATTENTION
        for causal in (False, True)
        if not causal or case[3] == 256
    ],
)
def test_attention_with_simdgroup_matrix(
    rng, head_dim, tile_q, tile_k, key_length, simdgroup, causal
):
    """Compare attention under `dot_general="simdgroup"` with NumPy over two batches
    and three heads, and check which schedule the tiles select."""
    shape = (2, 256, 3, head_dim)
    kv_shape = (2, key_length, 3, head_dim)
    tiles = {"tile_q": tile_q, "tile_k": tile_k, "causal": causal, "key_length": key_length}
    avals = (jax.ShapeDtypeStruct(shape, jnp.float32),) + (
        jax.ShapeDtypeStruct(kv_shape, jnp.float32),
    ) * 2
    spec = palladium.trace(attention_call(shape, **tiles), *avals)
    assert ("simdgroup_mma" in palladium.emit_msl(spec, dot_general="simdgroup")) == simdgroup
    call = attention_call(
        shape,
        **tiles,
        call=palladium.metal_call,
        compiler_params=palladium.CompilerParams(dot_general="simdgroup"),
    )
    q = rng.standard_normal(shape, dtype=np.float32)
    k, v = (rng.standard_normal(kv_shape, dtype=np.float32) for _ in range(2))
    np.testing.assert_allclose(
        call(q, k, v), reference_attention(q, k, v, causal=causal), rtol=3e-4, atol=3e-4
    )
