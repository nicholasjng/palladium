"""Adapt canonical batched StableHLO products to the existing Pallas GEMM emitter."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from palladium.device import prefers_simdgroup_matrix
from palladium.emit.tensorops import compile_kernel
from palladium.trace import trace


@functools.lru_cache(maxsize=128)
def gemm_source(
    batch: int, m: int, n: int, k: int, dtype: np.dtype
) -> tuple[str, tuple[int, int, int]]:
    """Return source and thread grid; a program is a group of 128 threads.

    Aligned float32 products use SIMD-group matrices where they are faster
    (`prefers_simdgroup_matrix`); Metal 4 TensorOps (`matmul2d`) handles the
    rest: edge tiles, K tails, half/bfloat storage, and GPUs with matrix
    accelerators.
    """
    tm = 64 if m % 64 == 0 else 32 if m % 32 == 0 else 16
    tn = 64 if n % 64 == 0 else 32 if n % 32 == 0 else 16

    def kernel(a, b, out):
        out[...] = jnp.matmul(a[...], b[...])

    # An unbatched product is a rank-2 call, its tiles on x and y. With a
    # leading batch axis of 1, x holds one threadgroup and the tiles go on y
    # and z, which ran a 4096^3 float32 product 1.5-1.8x slower on the M2.
    # The emitter requires the batch first in batched calls.
    if batch == 1:
        grid = (pl.cdiv(m, tm), pl.cdiv(n, tn))
        in_specs = [
            pl.BlockSpec((tm, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, tn), lambda i, j: (0, j)),
        ]
        out_spec = pl.BlockSpec((tm, tn), lambda i, j: (i, j))
        shapes = (m, k), (k, n), (m, n)
        threads = (grid[0] * 128, grid[1], 1)
    else:
        grid = (batch, pl.cdiv(m, tm), pl.cdiv(n, tn))
        in_specs = [
            pl.BlockSpec((1, tm, k), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((1, k, tn), lambda b, i, j: (b, 0, j)),
        ]
        out_spec = pl.BlockSpec((1, tm, tn), lambda b, i, j: (b, i, j))
        shapes = (batch, m, k), (batch, k, n), (batch, m, n)
        threads = (batch * 128, grid[1], grid[2])
    call = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        out_shape=jax.ShapeDtypeStruct(shapes[2], dtype),
    )
    spec = trace(call, *(jax.ShapeDtypeStruct(shape, dtype) for shape in shapes[:2]))
    source, _ = compile_kernel(spec, "gemm", prefer_simdgroup=prefers_simdgroup_matrix())
    return source, threads
