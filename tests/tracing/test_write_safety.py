"""Parallel-safety validation: grid instances run as parallel threads, so
writes that provably collide across a grid axis are rejected at trace
time instead of racing silently on the GPU.

Trace-only, runs without a Metal device.
"""

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import pytest

from palladium.errors import TraceError
from palladium.trace import trace

F32 = jnp.float32


def _trace(kernel, grid, in_shape, out_shape, **kwargs):
    call = pl.pallas_call(
        kernel, grid=grid, out_shape=jax.ShapeDtypeStruct(out_shape, F32), **kwargs
    )
    return trace(call, jax.ShapeDtypeStruct(in_shape, F32))


def test_full_block_write_ignoring_the_grid_axis_is_rejected():
    # Default out spec: whole array, constant index map. Every one of the
    # 4 instances writes all 8 elements: a guaranteed race.
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    with pytest.raises(TraceError, match="grid axis 0"):
        _trace(kernel, (4,), (8,), (8,))


def test_write_indexed_by_program_id_passes():
    def kernel(x_ref, o_ref):
        i = pl.program_id(0)
        o_ref[i] = x_ref[i] * 2.0

    spec = _trace(kernel, (4,), (4,), (4,))
    assert spec.grid == (4,)


def test_write_indexed_through_arithmetic_on_program_id_passes():
    def kernel(x_ref, o_ref):
        i = pl.program_id(0) * 2
        o_ref[i] = x_ref[i]
        o_ref[i + 1] = x_ref[i + 1]

    spec = _trace(kernel, (4,), (8,), (8,))
    assert spec.grid == (4,)


def test_blocked_output_index_map_passes():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    spec = _trace(
        kernel,
        (4,),
        (8,),
        (8,),
        in_specs=[pl.BlockSpec((2,), lambda i: i)],
        out_specs=pl.BlockSpec((2,), lambda i: i),
    )
    assert spec.grid == (4,)


def test_second_grid_axis_ignored_by_map_and_writes_is_rejected():
    def kernel(x_ref, o_ref):
        i = pl.program_id(0)
        o_ref[i] = x_ref[i] * 2.0  # never varies along axis 1

    with pytest.raises(TraceError, match="grid axis 1"):
        _trace(kernel, (4, 3), (4,), (4,))


def test_write_in_control_flow_indexed_by_a_program_id_operand_passes():
    def kernel(x_ref, o_ref):
        i = pl.program_id(0)

        def write(_):
            o_ref[i] = x_ref[i]
            return 0

        jax.lax.cond(i >= 0, write, lambda _: 0, 0)

    spec = _trace(kernel, (4,), (4,), (4,))
    assert spec.grid == (4,)


def test_write_in_a_loop_calling_program_id_passes():
    def kernel(x_ref, o_ref):
        def body(_, carry):
            i = pl.program_id(0)
            o_ref[i] = x_ref[i] + carry
            return carry

        jax.lax.fori_loop(0, 3, body, 0.0)

    spec = _trace(kernel, (4,), (4,), (4,))
    assert spec.grid == (4,)


def test_write_in_a_loop_ignoring_the_grid_axis_is_rejected():
    def kernel(x_ref, o_ref):
        def body(k, carry):
            o_ref[k] = x_ref[k] + carry
            return carry

        jax.lax.fori_loop(0, 8, body, 0.0)

    with pytest.raises(TraceError, match="grid axis 0"):
        _trace(kernel, (4,), (8,), (8,))


def test_unit_extent_axes_are_exempt():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    spec = _trace(kernel, (1,), (8,), (8,))
    assert spec.grid == (1,)
