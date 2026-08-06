"""Stretch 6: indexed ref access (`x_ref[i, j]`) in `_rule_get`/`_rule_swap`.

Non-Slice indices (scalar Var or Literal atoms) squeeze that dim into a
pointer offset; Slice indices keep the dim. Verified against real jaxprs
before implementing (see NEXT.md): a Slice's `start` is a plain Python int
when static, a Var when dynamic (`pl.dslice`); a non-Slice index is always
an ordinary atom, same as any other jaxpr value. Only the first kept dim
may be partial (the same contiguous-block discipline BlockSpec blocks
already follow); non-unit strides are unimplemented.

This does not touch a Ref's own BlockSpec binding (`test_04_grid_blocks.py`'s
`test_noncontiguous_block_is_rejected` still holds): strided or multi-dim
partial *blocks* are a separate, still-unimplemented restriction.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium
from palladium.emit import EmitError

pytestmark = pytest.mark.exercise


def test_dynamic_scalar_index(rng):
    """`x_ref[i, :]`: program_id selects a row, `:` keeps the trailing dim."""

    def kernel(x_ref, o_ref):
        i = pl.program_id(0)
        o_ref[0, :] = x_ref[i, :]

    f = palladium.metal_call(
        kernel,
        grid=(3,),
        in_specs=[pl.BlockSpec((3, 4), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((1, 4), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((3, 4), jnp.float32),
    )
    x = rng.standard_normal((3, 4), dtype=np.float32)
    np.testing.assert_allclose(f(x), x, rtol=1e-6)


def test_static_literal_index(rng):
    """`x_ref[1, :]`: a compile-time-constant row index."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[1, :] - x_ref[0, :]

    f = palladium.metal_call(
        kernel,
        out_specs=pl.BlockSpec((4,), lambda: (0,)),
        out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
    )
    x = rng.standard_normal((3, 4), dtype=np.float32)
    np.testing.assert_allclose(f(x), x[1] - x[0], rtol=1e-6)


def test_fully_scalar_indexed_get_and_swap(rng):
    """The stencil shape example 4 needs: both dims scalar on both the
    load and the store, boundary-guarded neighbour reads via `jnp.where`
    (out-of-bounds indices are never actually loaded, `where` picks the
    branch, but both branches trace, so the index must stay in-range)."""

    def kernel(x_ref, o_ref):
        i, j = pl.program_id(0), pl.program_id(1)
        n = 4
        # Clamp with `where`, not `jnp.minimum`/`maximum`: integer min/max
        # is a separate, still-unimplemented gap (`fmin`/`fmax` are
        # float-only in Metal); `where` only needs already-supported
        # comparisons and select_n.
        j_left = jnp.where(j > 0, j - 1, 0)
        j_right = jnp.where(j < n - 1, j + 1, n - 1)
        left = jnp.where(j > 0, x_ref[i, j_left], 0.0)
        right = jnp.where(j < n - 1, x_ref[i, j_right], 0.0)
        o_ref[i, j] = 2.0 * x_ref[i, j] - left - right

    f = palladium.metal_call(
        kernel,
        grid=(4, 4),
        in_specs=[pl.BlockSpec((4, 4), lambda i, j: (0, 0))],
        out_specs=pl.BlockSpec((4, 4), lambda i, j: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((4, 4), jnp.float32),
    )
    x = rng.standard_normal((4, 4), dtype=np.float32)
    got = f(x)
    want = np.asarray(f.interpret(x))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_periodic_wraparound_index(rng):
    """Neighbour reads with periodic (wraparound) boundaries: `i == n - 1`
    stages `eq`, `i != 0` would stage `ne`. Both were missing from
    ELEMENTWISE (only lt/le/gt/ge existed) until this stretch needed them
    for exactly this pattern (example 4's stencil)."""

    def kernel(x_ref, o_ref):
        i, j = pl.program_id(0), pl.program_id(1)
        n = 4
        ip = jnp.where(i == n - 1, 0, i + 1)
        im = jnp.where(i != 0, i - 1, n - 1)
        o_ref[i, j] = x_ref[ip, j] + x_ref[im, j]

    f = palladium.metal_call(
        kernel,
        grid=(4, 4),
        in_specs=[pl.BlockSpec((4, 4), lambda i, j: (0, 0))],
        out_specs=pl.BlockSpec((4, 4), lambda i, j: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((4, 4), jnp.float32),
    )
    x = rng.standard_normal((4, 4), dtype=np.float32)
    got = f(x)
    want = np.roll(x, -1, 0) + np.roll(x, 1, 0)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_dynamic_partial_leading_slice(rng):
    """`x_ref[pl.dslice(i, 2), :]`: a dynamic-start, statically-sized
    partial slice on the leading dim, `:` kept full on the trailing one."""

    def kernel(x_ref, o_ref):
        i = pl.program_id(0)
        o_ref[...] = x_ref[pl.dslice(i, 2), :]

    f = palladium.metal_call(
        kernel,
        grid=(3,),
        in_specs=[pl.BlockSpec((5, 4), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((2, 4), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((6, 4), jnp.float32),
    )
    x = rng.standard_normal((5, 4), dtype=np.float32)
    got = f(x)
    want = np.asarray(f.interpret(x))
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_strided_slice_rejected():
    """A non-unit stride (`x_ref[::2]`) is a real gap, not silently wrong."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[::2]

    f = palladium.metal_call(
        kernel,
        out_specs=pl.BlockSpec((2,), lambda: (0,)),
        out_shape=jax.ShapeDtypeStruct((2,), jnp.float32),
    )
    with pytest.raises(EmitError, match="stride"):
        f(np.zeros(4, np.float32))


def test_noncontiguous_kept_dims_rejected():
    """A partial Slice that isn't the first kept dim: all rows, all
    columns, then (1 of 2) of a trailing dim, non-contiguous in memory
    (mirrors `_check_contiguous`'s "only the leading dim may be partial"
    for BlockSpec blocks, one level down)."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[:, :, 0:1]

    f = palladium.metal_call(
        kernel,
        out_specs=pl.BlockSpec((4, 3, 1), lambda: (0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((4, 3, 1), jnp.float32),
    )
    with pytest.raises(EmitError, match="first kept dim"):
        f(np.zeros((4, 3, 2), np.float32))
