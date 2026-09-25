"""Codegen contract for the opt-in cooperative TensorOps dot path."""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl

import palladium
from palladium.emit import EmitError


def _dot(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.matmul(a_ref[...], b_ref[...])


def _blocked_dot(m=32, n=64, k=16, tm=16, tn=32):
    call = pl.pallas_call(
        _dot,
        grid=(m // tm, n // tn),
        in_specs=[
            pl.BlockSpec((tm, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, tn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
    )
    spec = palladium.trace(
        call,
        jax.ShapeDtypeStruct((m, k), jnp.float32),
        jax.ShapeDtypeStruct((k, n), jnp.float32),
    )
    return spec


def _blocked_batched_dot(batch=2, m=32, n=64, k=16, tm=16, tn=32):
    call = pl.pallas_call(
        _dot,
        grid=(batch, m // tm, n // tn),
        in_specs=[
            pl.BlockSpec((1, tm, k), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((1, k, tn), lambda b, i, j: (b, 0, j)),
        ],
        out_specs=pl.BlockSpec((1, tm, tn), lambda b, i, j: (b, i, j)),
        out_shape=jax.ShapeDtypeStruct((batch, m, n), jnp.float32),
    )
    spec = palladium.trace(
        call,
        jax.ShapeDtypeStruct((batch, m, k), jnp.float32),
        jax.ShapeDtypeStruct((batch, k, n), jnp.float32),
    )
    return spec


def test_tensorops_dot_uses_one_threadgroup_per_pallas_program():
    msl = palladium.emit_msl(_blocked_dot(), dot_general="tensorops")

    assert "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>" in msl
    assert "uint3 _pid [[threadgroup_position_in_grid]]" in msl
    assert "execution_simdgroups<4>" in msl
    assert "matmul2d_descriptor desc(16, 32, 16" in msl
    assert "arg0 + _pid.x * 256" in msl
    assert "arg1 + _pid.y * 32" in msl
    assert "arg2 + _pid.x * 1024 + _pid.y * 32" in msl
    assert "for (uint" not in msl


def test_tensorops_batched_dot_maps_batch_and_output_tiles_to_threadgroups():
    msl = palladium.emit_msl(_blocked_batched_dot(), dot_general="tensorops")
    ordinary = palladium.emit_msl(_blocked_batched_dot())

    assert "uint3 _pid [[threadgroup_position_in_grid]]" in msl
    assert "matmul2d_descriptor desc(16, 32, 16" in msl
    assert "arg0 + _pid.x * 512 + _pid.y * 256" in msl
    assert "arg1 + _pid.x * 1024 + _pid.z * 32" in msl
    assert "arg2 + _pid.x * 2048 + _pid.y * 1024 + _pid.z * 32" in msl
    assert "MetalPerformancePrimitives" not in ordinary
    assert "thread_position_in_grid" in ordinary


def test_tensorops_batched_dot_explain_scales_batch_axis_for_groups():
    f = palladium.metal_call(
        _dot,
        grid=(2, 2, 2),
        in_specs=[
            pl.BlockSpec((1, 16, 16), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((1, 16, 32), lambda b, i, j: (b, 0, j)),
        ],
        out_specs=pl.BlockSpec((1, 16, 32), lambda b, i, j: (b, i, j)),
        out_shape=jax.ShapeDtypeStruct((2, 32, 64), jnp.float32),
        dot_general="tensorops",
    )
    diag = f.explain(
        jax.ShapeDtypeStruct((2, 32, 16), jnp.float32),
        jax.ShapeDtypeStruct((2, 16, 64), jnp.float32),
    )

    assert diag.grid == (256, 2, 2)
    assert diag.threadgroup == (128, 1, 1)


def test_tensorops_dot_is_opt_in():
    ordinary = palladium.emit_msl(_blocked_dot())
    assert "MetalPerformancePrimitives" not in ordinary
    assert "thread_position_in_grid" in ordinary


def test_tensorops_explain_reports_group_scaled_dispatch():
    f = palladium.metal_call(
        _dot,
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((16, 16), lambda i, j: (i, 0)),
            pl.BlockSpec((16, 32), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((16, 32), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((32, 64), jnp.float32),
        dot_general="tensorops",
    )
    diag = f.explain(
        jax.ShapeDtypeStruct((32, 16), jnp.float32),
        jax.ShapeDtypeStruct((16, 64), jnp.float32),
    )

    assert diag.grid == (256, 2, 1)
    assert diag.threadgroup == (128, 1, 1)
    assert diag.cooperative


def test_tensorops_dot_is_used_by_jittable_metal_runtime_calls():
    call = palladium.metal_call_jit(
        _dot,
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((16, 16), lambda i, j: (i, 0)),
            pl.BlockSpec((16, 32), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((16, 32), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((32, 64), jnp.float32),
        dot_general="tensorops",
    )
    _, msl = call._spec_and_msl(
        (
            jax.ShapeDtypeStruct((32, 16), jnp.float32),
            jax.ShapeDtypeStruct((16, 64), jnp.float32),
        )
    )

    assert "threadgroup_position_in_grid" in msl
    assert "execution_simdgroups<4>" in msl


def test_tensorops_dot_rejects_unblocked_single_program_matmul():
    call = pl.pallas_call(
        _dot,
        out_shape=jax.ShapeDtypeStruct((16, 32), jnp.float32),
    )
    spec = palladium.trace(
        call,
        jax.ShapeDtypeStruct((16, 16), jnp.float32),
        jax.ShapeDtypeStruct((16, 32), jnp.float32),
    )

    with pytest.raises(EmitError, match="full-block matmul"):
        palladium.emit_msl(spec, dot_general="tensorops")
