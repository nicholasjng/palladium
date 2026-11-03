"""Adapt canonical batched StableHLO products to the existing Pallas GEMM emitter."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from palladium.emit.tensorops import compile_kernel
from palladium.trace import trace


@functools.lru_cache(maxsize=128)
def gemm_source(
    batch: int, m: int, n: int, k: int, dtype: np.dtype
) -> tuple[str, tuple[int, int, int]]:
    """Return source and thread grid; a program is a group of 128 threads.

    Aligned float32 products use SIMD-group matrices. The existing Metal 4
    TensorOps path handles edge tiles, K tails, and half/bfloat storage.
    """
    tm = 64 if m % 64 == 0 else 32 if m % 32 == 0 else 16
    tn = 64 if n % 64 == 0 else 32 if n % 32 == 0 else 16

    def kernel(a, b, out):
        out[...] = jnp.matmul(a[...], b[...])

    grid = (batch, pl.cdiv(m, tm), pl.cdiv(n, tn))
    call = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, tm, k), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((1, k, tn), lambda b, i, j: (b, 0, j)),
        ],
        out_specs=pl.BlockSpec((1, tm, tn), lambda b, i, j: (b, i, j)),
        out_shape=jax.ShapeDtypeStruct((batch, m, n), dtype),
    )
    spec = trace(
        call, jax.ShapeDtypeStruct((batch, m, k), dtype), jax.ShapeDtypeStruct((batch, k, n), dtype)
    )
    source, _ = compile_kernel(spec, "gemm", prefer_simdgroup=True)
    return source, (batch * 128, grid[1], grid[2])
