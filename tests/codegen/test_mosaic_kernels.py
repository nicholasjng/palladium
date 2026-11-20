"""Kernels written the way Mosaic's are: a grid walked in order with scratch
kept between programs, another backend's compiler parameters, and scalar
operands prefetched for the index maps. Checked against the interpreter."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import palladium
from palladium.errors import TraceError


def _accumulate(x_ref, o_ref, acc_ref):
    """Sum the blocks along grid axis 1: zero at its first step, write at its last."""

    @pl.when(pl.program_id(1) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += x_ref[...]

    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)
    def _():
        o_ref[...] = acc_ref[...]


def _accumulating_call(compiler_params):
    return palladium.metal_call(
        _accumulate,
        grid=(4, 4),
        in_specs=[pl.BlockSpec((16, 16), lambda i, k: (i, k))],
        out_specs=pl.BlockSpec((16, 16), lambda i, k: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((64, 16), jnp.float32),
        scratch_shapes=[pltpu.VMEM((16, 16), jnp.float32)],
        **({"compiler_params": compiler_params} if compiler_params else {}),
    )


@pytest.mark.parametrize(
    "compiler_params",
    [
        palladium.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        # Mosaic's own parameters, as a kernel written for it carries them.
        pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        # Without semantics Mosaic runs every axis in order.
        pltpu.CompilerParams(),
    ],
    ids=["palladium", "mosaic", "mosaic-default"],
)
def test_scratch_accumulates_along_an_axis_run_in_order(rng, compiler_params):
    x = rng.standard_normal((64, 64), dtype=np.float32)
    got = _accumulating_call(compiler_params)(x)
    np.testing.assert_allclose(got, x.reshape(64, 4, 16).sum(1), rtol=1e-5)


def test_scratch_read_before_written_needs_dimension_semantics():
    """Programs are parallel threads with scratch of their own: a kernel that
    reads what an earlier program left there would compute something else."""
    spec = jax.ShapeDtypeStruct((64, 64), jnp.float32)
    with pytest.raises(TraceError, match="reads scratch before writing it"):
        palladium.trace(_accumulating_call(None).interpret, spec)


def test_axis_run_in_order_is_a_loop_not_a_grid_dimension():
    """Only the parallel axes take one of Metal's three grid dimensions."""
    params = palladium.CompilerParams(dimension_semantics=("parallel", "arbitrary"))
    example = jax.ShapeDtypeStruct((64, 64), jnp.float32)
    assert _accumulating_call(params).explain(example).grid == (4, 1, 1)
    spec = palladium.trace(_accumulating_call(params).interpret, example)
    assert spec.grid == (4, 4) and spec.sequential == (1,)
    assert "for (uint _seq1 = 0; _seq1 < 4; ++_seq1)" in palladium.emit_msl(spec)


def test_scalar_prefetch_feeds_the_kernel_and_the_index_maps(rng):
    """Each program adds its entry of the prefetched operand to the block that
    entry selects."""

    def kernel(order_ref, x_ref, o_ref):
        o_ref[...] = x_ref[...] + order_ref[pl.program_id(0)].astype(jnp.float32)

    call = palladium.metal_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(4,),
            in_specs=[pl.BlockSpec((16, 64), lambda i, order: (order[i], 0))],
            out_specs=pl.BlockSpec((16, 64), lambda i, order: (i, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct((64, 64), jnp.float32),
    )
    x = rng.standard_normal((64, 64), dtype=np.float32)
    order = np.array([3, 1, 0, 2], np.int32)
    want = np.concatenate([x[16 * j : 16 * j + 16] + np.float32(j) for j in order])
    np.testing.assert_array_equal(call(order, x), want)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16])
def test_narrow_integers_wrap_and_convert(rng, dtype):
    """8- and 16-bit blocks load, store, convert and do wrapping arithmetic."""
    info = np.iinfo(dtype)
    a, b = (rng.integers(info.min, info.max, (8, 16)).astype(dtype) for _ in range(2))

    def kernel(a_ref, b_ref, wrapped_ref, scaled_ref):
        x, y = a_ref[...], b_ref[...]
        wrapped_ref[...] = jnp.where(x < y, x * y + x, (-y) ^ (x & ~y))
        scaled_ref[...] = x.astype(jnp.float32) * 0.5

    call = palladium.metal_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((8, 16), dtype),
            jax.ShapeDtypeStruct((8, 16), jnp.float32),
        ),
    )
    for got, want in zip(call(a, b), call.interpret(a, b), strict=True):
        np.testing.assert_array_equal(got, want)


def test_narrow_integers_reject_rules_without_a_narrow_form():
    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = a_ref[...] // b_ref[...]

    call = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((8, 16), jnp.int8))
    spec = jax.ShapeDtypeStruct((8, 16), jnp.int8)
    with pytest.raises(palladium.EmitError, match="8- and 16-bit integers"):
        call.explain(spec, spec)
