"""`CompilerParams(program="threadgroup")`: row programs on one threadgroup each.

Every kernel is plain Pallas on a 1D grid of row tiles, checked against the
Pallas interpreter and NumPy through `metal_call`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium
from palladium.errors import EmitError
from palladium.launch import launch_geometry

ROWS = palladium.CompilerParams(program="threadgroup")


def _softmax(x_ref, o_ref):
    x = x_ref[...]
    e = jnp.exp(x - jnp.max(x, axis=1, keepdims=True))
    o_ref[...] = e / jnp.sum(e, axis=1, keepdims=True)


def _layer_norm(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[...]
    mean = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=1, keepdims=True)
    o_ref[...] = (x - mean) * jax.lax.rsqrt(var + 1e-5) * w_ref[...] + b_ref[...]


def _call(kernel, rows, columns, tile, *, vectors=0, out_rows=False, row_inputs=0):
    tile_spec = pl.BlockSpec((tile, columns), lambda i: (i, 0))
    vector_spec = pl.BlockSpec((columns,), lambda i: (0,))
    row_spec = pl.BlockSpec((tile,), lambda i: (i,))
    out_spec = row_spec if out_rows else tile_spec
    out_shape = (rows,) if out_rows else (rows, columns)
    return palladium.metal_call(
        kernel,
        grid=(pl.cdiv(rows, tile),),
        in_specs=[tile_spec] + [vector_spec] * vectors + [row_spec] * row_inputs,
        out_specs=out_spec,
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        compiler_params=ROWS,
    )


def _check(call, *args, reference, atol=1e-5):
    got = np.asarray(call(*args))
    np.testing.assert_allclose(got, np.asarray(call.interpret(*args)), rtol=1e-5, atol=atol / 10)
    np.testing.assert_allclose(got, reference, rtol=1e-4, atol=atol)


def _np_softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    ("rows", "columns", "tile"),
    [
        (50, 96, 8),  # narrow rows, partial last tile
        (9, 1000, 2),  # whole threadgroup, columns not a multiple of the threads per row
        (7, 300, 1),  # one row per program
        (64, 64, 64),  # rows per tile equal to columns
    ],
)
def test_row_softmax_matches_interpreter(metal_device, rng, rows, columns, tile):
    """A max, a sum and a store pass give the softmax for both layouts."""
    x = rng.standard_normal((rows, columns)).astype(np.float32) * 4.0
    _check(_call(_softmax, rows, columns, tile), x, reference=_np_softmax(x))


@pytest.mark.parametrize("columns", [128, 768])
def test_layer_norm_with_column_vectors(metal_device, rng, columns):
    """Weights and biases are (columns,) vectors read per column."""
    rows = 20
    x = rng.standard_normal((rows, columns)).astype(np.float32)
    w = rng.standard_normal((columns,)).astype(np.float32)
    b = rng.standard_normal((columns,)).astype(np.float32)
    mean = x.mean(axis=1, keepdims=True)
    want = (x - mean) / np.sqrt(x.var(axis=1, keepdims=True) + 1e-5) * w + b
    _check(_call(_layer_norm, rows, columns, 4, vectors=2), x, w, b, reference=want)


def test_per_row_output_and_input(metal_device, rng):
    """A (rows,) input scales each row; a (rows,) output holds a logsumexp."""

    def kernel(x_ref, t_ref, o_ref):
        x = x_ref[...] / t_ref[...][:, None]
        top = jnp.max(x, axis=1)
        o_ref[...] = top + jnp.log(jnp.sum(jnp.exp(x - top[:, None]), axis=1))

    x = rng.standard_normal((21, 640)).astype(np.float32)
    t = rng.uniform(0.5, 2.0, (21,)).astype(np.float32)
    scaled = x / t[:, None]
    want = np.log(np.exp(scaled - scaled.max(1, keepdims=True)).sum(1)) + scaled.max(1)
    _check(_call(kernel, 21, 640, 4, out_rows=True, row_inputs=1), x, t, reference=want)


def test_inlines_jax_nn_softmax_and_where(metal_device, rng):
    """`jax.nn.softmax` stages a custom_jvp_call and `jnp.where` a jit."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.where(x > 0, jax.nn.softmax(x, axis=1), 0.0)

    x = rng.standard_normal((12, 160)).astype(np.float32)
    want = np.where(x > 0, _np_softmax(x), 0.0)
    _check(_call(kernel, 12, 160, 4), x, reference=want)


def test_rejects_unknown_program_scope():
    with pytest.raises(ValueError, match="program must be"):
        palladium.CompilerParams(program="warp")


def test_declared_width_sets_the_launch():
    """Wide rows declare their group width; launch geometry follows it."""
    x = jax.ShapeDtypeStruct((8, 2048), jnp.float32)
    call = _call(_softmax, 8, 2048, 2)
    spec, msl, _ = call._spec_and_msl((x,))
    grid, group = launch_geometry(spec, msl, None)
    assert group == (512, 1, 1) and grid == (4 * 512, 1, 1)


# The scans differ only in their combine and identity: one arithmetic, one extremum.
_SCANS = {
    "cumsum": (jax.lax.cumsum, np.cumsum),
    "cummax": (jax.lax.cummax, np.maximum.accumulate),
}


@pytest.mark.parametrize("op", sorted(_SCANS))
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(("rows", "columns", "tile"), [(50, 96, 8), (9, 1000, 2), (3, 4096, 1)])
def test_column_scans(metal_device, rng, op, reverse, rows, columns, tile):
    """Scans across chunks of the row, forward and reverse, in both layouts."""
    jax_scan, np_scan = _SCANS[op]

    def kernel(x_ref, o_ref):
        o_ref[...] = jax_scan(x_ref[...], axis=1, reverse=reverse)

    x = rng.standard_normal((rows, columns)).astype(np.float32)
    flipped = x[:, ::-1] if reverse else x
    want = np_scan(flipped, axis=1)
    want = want[:, ::-1] if reverse else want
    # Sums of up to 4096 unit-scale terms, in another order than NumPy's.
    _check(_call(kernel, rows, columns, tile), x, reference=want, atol=1e-3)


def test_integer_scan_and_reduction_are_exact(metal_device, rng):
    """Whole-group partials keep 32-bit integers exact, beyond float's 2**24."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.cumsum(x, axis=1, dtype=jnp.int32) - jnp.sum(x, axis=1, keepdims=True)

    x = rng.integers(0, 1 << 20, (4, 2048)).astype(np.int32)
    call = palladium.metal_call(
        kernel,
        grid=(4,),
        in_specs=[pl.BlockSpec((1, 2048), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((1, 2048), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((4, 2048), jnp.int32),
        compiler_params=ROWS,
    )
    want = (np.cumsum(x, axis=1) - x.sum(axis=1, keepdims=True)).astype(np.int32)
    np.testing.assert_array_equal(call(x), want)


def test_rejects_scans_along_rows_and_overlong_rows(metal_device):
    def along_rows(x_ref, o_ref):
        o_ref[...] = jnp.cumsum(x_ref[...], axis=0)

    with pytest.raises(EmitError, match="scan along the columns only"):
        _call(along_rows, 8, 64, 8)(np.zeros((8, 64), np.float32))

    def along_columns(x_ref, o_ref):
        o_ref[...] = jnp.cumsum(x_ref[...], axis=1)

    with pytest.raises(EmitError, match="scan rows of at most"):
        _call(along_columns, 1, 20000, 1)(np.zeros((1, 20000), np.float32))


def _tile_call(kernel, rows, columns, tile, outputs, dtype=jnp.float32):
    """Outputs are "rows" (rows, columns), "tile" (programs, columns) and
    "scalar" (programs,), the last two one block per row tile."""
    programs = pl.cdiv(rows, tile)
    specs = {
        "rows": (pl.BlockSpec((tile, columns), lambda i: (i, 0)), (rows, columns)),
        "tile": (pl.BlockSpec((1, columns), lambda i: (i, 0)), (programs, columns)),
        "scalar": (pl.BlockSpec((1,), lambda i: (i,)), (programs,)),
    }
    return palladium.metal_call(
        kernel,
        grid=(programs,),
        in_specs=[pl.BlockSpec((tile, columns), lambda i: (i, 0))],
        out_specs=[specs[o][0] for o in outputs],
        out_shape=[jax.ShapeDtypeStruct(specs[o][1], dtype) for o in outputs],
        compiler_params=ROWS,
    )


@pytest.mark.parametrize(
    ("rows", "columns", "tile"),
    [
        (48, 96, 8),  # narrow: SIMD groups combine after the loop
        (16, 200, 2),  # narrow, fewer rows per tile than SIMD groups
        (12, 1000, 4),  # whole threadgroup per row
        (6, 64, 1),  # one row per tile
    ],
)
def test_column_statistics_per_tile(metal_device, rng, rows, columns, tile):
    """Mean, variance and maximum over the rows of each tile, accumulated
    across the row loop, next to a per-row softmax. The mean comes out of a
    nested jit, so it is bound after the loop."""

    def kernel(x_ref, s_ref, mean_ref, var_ref, max_ref):
        x = x_ref[...]
        e = jnp.exp(x - jnp.max(x, axis=1, keepdims=True))
        s_ref[...] = e / jnp.sum(e, axis=1, keepdims=True)
        mean = jax.jit(lambda x: jnp.mean(x, axis=0, keepdims=True))(x)
        mean_ref[...] = mean
        var_ref[...] = jnp.mean(x * x, axis=0, keepdims=True) - mean * mean
        max_ref[...] = jnp.max(x, axis=0, keepdims=True)

    x = rng.standard_normal((rows, columns)).astype(np.float32)
    tiles = x.reshape(-1, tile, columns)
    want = (_np_softmax(x), tiles.mean(1), tiles.var(1), tiles.max(1))
    call = _tile_call(kernel, rows, columns, tile, ("rows", "tile", "tile", "tile"))
    for got, interpreted, expected in zip(call(x), call.interpret(x), want, strict=True):
        np.testing.assert_allclose(got, interpreted, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(("columns", "tile"), [(96, 8), (1000, 4)])
def test_scalar_statistics_per_tile(metal_device, rng, columns, tile):
    """Whole-tile sums, and sums over the rows of per-row values."""

    def kernel(x_ref, total_ref, peaks_ref):
        x = x_ref[...]
        total_ref[...] = jnp.sum(x).reshape(1)
        peaks_ref[...] = jnp.sum(jnp.max(x, axis=1), keepdims=True)

    rows = 4 * tile
    x = rng.standard_normal((rows, columns)).astype(np.float32)
    tiles = x.reshape(-1, tile, columns)
    call = _tile_call(kernel, rows, columns, tile, ("scalar", "scalar"))
    total, peaks = call(x)
    np.testing.assert_allclose(total, tiles.sum((1, 2)), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(peaks, tiles.max(2).sum(1), rtol=1e-5, atol=1e-5)


def test_integer_row_sums_are_exact(metal_device, rng):
    """SIMD groups combine row accumulators as bits, exactly."""

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], axis=0, keepdims=True)

    x = rng.integers(0, 1 << 26, (64, 128)).astype(np.int32)
    got = _tile_call(kernel, 64, 128, 16, ("tile",), dtype=jnp.int32)(x)
    np.testing.assert_array_equal(got, x.reshape(4, 16, 128).sum(1, dtype=np.int32))


def test_rejects_rows_after_reductions_over_them(metal_device):
    def centered(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = x - jnp.sum(x, axis=0, keepdims=True)

    with pytest.raises(EmitError, match="only per-tile values remain"):
        _call(centered, 16, 64, 4)(np.zeros((16, 64), np.float32))

    def column_sums(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], axis=0, keepdims=True)

    with pytest.raises(EmitError, match="only when tiles divide them"):
        _tile_call(column_sums, 50, 64, 8, ("tile",))(np.zeros((50, 64), np.float32))


def _three_point(x_ref, o_ref):
    padded = jnp.pad(x_ref[...], ((0, 0), (1, 1)))
    o_ref[...] = padded[:, :-2] - 2.0 * padded[:, 1:-1] + padded[:, 2:]


def _roll(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = jnp.roll(x, 3, axis=1) - jnp.roll(jnp.exp(x), -5, axis=1)


def _forward_difference(x_ref, o_ref):
    o_ref[...] = jnp.diff(x_ref[...], axis=1, append=0.0)


def _np_three_point(x):
    padded = np.pad(x, ((0, 0), (1, 1)))
    return padded[:, :-2] - 2 * padded[:, 1:-1] + padded[:, 2:]


_STENCILS = {
    "three_point": (_three_point, _np_three_point),
    "roll": (_roll, lambda x: np.roll(x, 3, axis=1) - np.roll(np.exp(x), -5, axis=1)),
    "difference": (_forward_difference, lambda x: np.diff(x, axis=1, append=0.0)),
}


@pytest.mark.parametrize("stencil", sorted(_STENCILS))
@pytest.mark.parametrize(("rows", "columns", "tile"), [(50, 96, 8), (9, 1000, 2)])
def test_column_stencils(metal_device, rng, stencil, rows, columns, tile):
    """Slices, pads and concatenations along the row read shifted columns."""
    kernel, reference = _STENCILS[stencil]
    x = rng.standard_normal((rows, columns)).astype(np.float32)
    _check(_call(kernel, rows, columns, tile), x, reference=reference(x))


def test_shifted_column_vector_and_reduction(metal_device, rng):
    """A shifted (columns,) vector, and a reduction over a stencil."""

    def kernel(x_ref, w_ref, o_ref):
        x, w = x_ref[...], w_ref[...]
        smooth = x[:, 1:] * w[:-1] + x[:, :-1] * w[1:]
        o_ref[...] = jnp.sum(smooth * smooth, axis=1)

    x = rng.standard_normal((20, 300)).astype(np.float32)
    w = rng.standard_normal((300,)).astype(np.float32)
    smooth = x[:, 1:] * w[:-1] + x[:, :-1] * w[1:]
    want = (smooth * smooth).sum(1)
    _check(_call(kernel, 20, 300, 4, vectors=1, out_rows=True), x, w, reference=want, atol=1e-4)


def test_rejects_shifted_scans_and_row_shifts(metal_device):
    def shifted_scan(x_ref, o_ref):
        o_ref[...] = jnp.roll(jnp.cumsum(x_ref[...], axis=1), 1, axis=1)

    with pytest.raises(EmitError, match="cannot shift a scan"):
        _call(shifted_scan, 8, 64, 4)(np.zeros((8, 64), np.float32))

    def row_shift(x_ref, o_ref):
        o_ref[...] = jnp.roll(x_ref[...], 1, axis=0)

    with pytest.raises(EmitError, match="along the columns only"):
        _call(row_shift, 8, 64, 4)(np.zeros((8, 64), np.float32))
