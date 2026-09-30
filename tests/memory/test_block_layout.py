"""Grids and blocks: `_rule_program_id` and `block_offset`.

Each program instance's Refs point at its own block, so the emitted pointers
carry per-thread offsets computed from the BlockSpec index map.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

import palladium


def test_program_id_lands_in_the_right_block():
    def kernel(o_ref):
        pid = pl.program_id(0)
        o_ref[...] = jnp.zeros((4,), jnp.float32) + pid

    f = palladium.metal_call(
        kernel,
        grid=(16,),
        out_specs=pl.BlockSpec((4,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((64,), jnp.float32),
    )
    want = np.repeat(np.arange(16, dtype=np.float32), 4)
    np.testing.assert_array_equal(f(), want)


def test_row_blocks_2d(rng):
    """Blocks are rows of a 2D array: (1, 128) of (64, 128), contiguous."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1.0

    f = palladium.metal_call(
        kernel,
        grid=(64,),
        in_specs=[pl.BlockSpec((1, 128), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((1, 128), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((64, 128), jnp.float32),
    )
    x = rng.standard_normal((64, 128), dtype=np.float32)
    np.testing.assert_allclose(f(x), x + 1.0, rtol=1e-6)


def test_2d_grid_with_index_map_arithmetic(rng):
    """A 2D grid mapped onto rows via `i * 16 + j` lowers the index map's arithmetic through the ELEMENTWISE table."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 3.0

    f = palladium.metal_call(
        kernel,
        grid=(8, 16),
        in_specs=[pl.BlockSpec((1, 32), lambda i, j: (i * 16 + j, 0))],
        out_specs=pl.BlockSpec((1, 32), lambda i, j: (i * 16 + j, 0)),
        out_shape=jax.ShapeDtypeStruct((128, 32), jnp.float32),
    )
    x = rng.standard_normal((128, 32), dtype=np.float32)
    got = f(x)
    want = np.asarray(f.interpret(x))
    np.testing.assert_allclose(got, want, rtol=1e-6)
