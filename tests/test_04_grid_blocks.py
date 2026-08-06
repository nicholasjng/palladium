"""Exercise 3: `_rule_program_id` and `_block_offset` in emit.py.

Until now every kernel was one program instance seeing the whole array.
These kernels tile: the grid launches many program instances and each one's
Refs point at *its* block, so the emitted pointers need per-thread offsets
computed from the BlockSpec index map.

Requires exercise 2 (the offset arithmetic and the `+ pid` kernel reuse
elementwise rules).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium
from palladium.emit import EmitError

pytestmark = pytest.mark.exercise


def test_blocked_double_1d(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    f = palladium.metal_call(
        kernel,
        grid=(32,),
        in_specs=[pl.BlockSpec((8,), lambda i: (i,))],
        out_specs=pl.BlockSpec((8,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
    )
    x = rng.standard_normal(256, dtype=np.float32)
    np.testing.assert_allclose(f(x), 2.0 * x, rtol=1e-6)


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
    """Blocks are rows of a 2D array: (1, 128) of (64, 128) — contiguous."""

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
    """A 2D grid mapped onto rows via `i * 16 + j`. If you evaluated the
    index map by recursing with emit_jaxpr (as the exercise hint suggests),
    the mul/add inside it come from your exercise-2 table for free."""

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


def test_noncontiguous_block_is_rejected(rng):
    """(4, 8) blocks of a (32, 128) array are strided in memory. The driver
    refuses them up front — loudly wrong beats silently wrong. Lifting this
    (strided copy loops in get/swap) is the ROADMAP's stretch exercise 6."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    f = palladium.metal_call(
        kernel,
        grid=(8, 16),
        in_specs=[pl.BlockSpec((4, 8), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((4, 8), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((32, 128), jnp.float32),
    )
    with pytest.raises(EmitError, match="not contiguous"):
        f(np.zeros((32, 128), np.float32))
