"""Codegen contract for the opt-in cooperative TensorOps dot path."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.extend.core import Jaxpr

import palladium
from palladium.emit import EmitError


def _dot(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.matmul(a_ref[...], b_ref[...])


def _dot_rhs_transposed(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.matmul(a_ref[...], jnp.swapaxes(b_ref[...], 0, 1))


def _dot_lhs_transposed(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.matmul(jnp.swapaxes(a_ref[...], 0, 1), b_ref[...])


def _relu_dot(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.maximum(jnp.matmul(a_ref[...], b_ref[...]), 0.0)


def _residual_dot(a_ref, b_ref, residual_ref, out_ref):
    out_ref[...] = jnp.matmul(a_ref[...], b_ref[...]) + residual_ref[...]


def _bias_dot(a_ref, b_ref, bias_ref, out_ref):
    out_ref[...] = jnp.matmul(a_ref[...], b_ref[...]) + bias_ref[...]


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


def _blocked_dot_rhs_transposed(m=32, n=64, k=16, tm=16, tn=32):
    call = pl.pallas_call(
        _dot_rhs_transposed,
        grid=(m // tm, n // tn),
        in_specs=[
            pl.BlockSpec((tm, k), lambda i, j: (i, 0)),
            pl.BlockSpec((tn, k), lambda i, j: (j, 0)),
        ],
        out_specs=pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
    )
    return palladium.trace(
        call,
        jax.ShapeDtypeStruct((m, k), jnp.float32),
        jax.ShapeDtypeStruct((n, k), jnp.float32),
    )


def _blocked_dot_lhs_transposed(m=32, n=64, k=16, tm=16, tn=32):
    call = pl.pallas_call(
        _dot_lhs_transposed,
        grid=(m // tm, n // tn),
        in_specs=[
            pl.BlockSpec((k, tm), lambda i, j: (0, i)),
            pl.BlockSpec((k, tn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
    )
    return palladium.trace(
        call,
        jax.ShapeDtypeStruct((k, m), jnp.float32),
        jax.ShapeDtypeStruct((k, n), jnp.float32),
    )


def _blocked_relu_dot():
    call = pl.pallas_call(
        _relu_dot,
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((16, 16), lambda i, j: (i, 0)),
            pl.BlockSpec((16, 32), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((16, 32), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((32, 64), jnp.float32),
    )
    return palladium.trace(
        call,
        jax.ShapeDtypeStruct((32, 16), jnp.float32),
        jax.ShapeDtypeStruct((16, 64), jnp.float32),
    )


def _blocked_residual_dot():
    call = pl.pallas_call(
        _residual_dot,
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((16, 16), lambda i, j: (i, 0)),
            pl.BlockSpec((16, 32), lambda i, j: (0, j)),
            pl.BlockSpec((16, 32), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((16, 32), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((32, 64), jnp.float32),
    )
    return palladium.trace(
        call,
        jax.ShapeDtypeStruct((32, 16), jnp.float32),
        jax.ShapeDtypeStruct((16, 64), jnp.float32),
        jax.ShapeDtypeStruct((32, 64), jnp.float32),
    )


def _blocked_column_bias_dot():
    call = pl.pallas_call(
        _bias_dot,
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((16, 16), lambda i, j: (i, 0)),
            pl.BlockSpec((16, 32), lambda i, j: (0, j)),
            pl.BlockSpec((32,), lambda i, j: (j,)),
        ],
        out_specs=pl.BlockSpec((16, 32), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((32, 64), jnp.float32),
    )
    return palladium.trace(
        call,
        jax.ShapeDtypeStruct((32, 16), jnp.float32),
        jax.ShapeDtypeStruct((16, 64), jnp.float32),
        jax.ShapeDtypeStruct((64,), jnp.float32),
    )


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


def _blocked_batched_relu_dot():
    batch, m, n, k, tm, tn = 2, 32, 64, 16, 16, 32
    call = pl.pallas_call(
        _relu_dot,
        grid=(batch, m // tm, n // tn),
        in_specs=[
            pl.BlockSpec((1, tm, k), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((1, k, tn), lambda b, i, j: (b, 0, j)),
        ],
        out_specs=pl.BlockSpec((1, tm, tn), lambda b, i, j: (b, i, j)),
        out_shape=jax.ShapeDtypeStruct((batch, m, n), jnp.float32),
    )
    return palladium.trace(
        call,
        jax.ShapeDtypeStruct((batch, m, k), jnp.float32),
        jax.ShapeDtypeStruct((batch, k, n), jnp.float32),
    )


def _blocked_batched_residual_dot():
    batch, m, n, k, tm, tn = 2, 32, 64, 16, 16, 32
    call = pl.pallas_call(
        _residual_dot,
        grid=(batch, m // tm, n // tn),
        in_specs=[
            pl.BlockSpec((1, tm, k), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((1, k, tn), lambda b, i, j: (b, 0, j)),
            pl.BlockSpec((1, tm, tn), lambda b, i, j: (b, i, j)),
        ],
        out_specs=pl.BlockSpec((1, tm, tn), lambda b, i, j: (b, i, j)),
        out_shape=jax.ShapeDtypeStruct((batch, m, n), jnp.float32),
    )
    return palladium.trace(
        call,
        jax.ShapeDtypeStruct((batch, m, k), jnp.float32),
        jax.ShapeDtypeStruct((batch, k, n), jnp.float32),
        jax.ShapeDtypeStruct((batch, m, n), jnp.float32),
    )


def test_tensorops_dot_uses_one_threadgroup_per_pallas_program():
    msl = palladium.emit_msl(_blocked_dot())

    assert "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>" in msl
    assert "uint3 _pid [[threadgroup_position_in_grid]]" in msl
    assert "execution_simdgroups<4>" in msl
    assert "matmul2d_descriptor desc(16, 32, 16" in msl
    assert "arg0 + (int)_pid.x * 256" in msl
    assert "arg1 + (int)_pid.y * 32" in msl
    assert "arg2 + (int)_pid.x * 1024 + (int)_pid.y * 32" in msl


def test_tensorops_dot_fuses_relu_epilogue_in_cooperative_tensor():
    msl, stats = palladium.emit.emit_msl_stats(_blocked_relu_dot(), dot_general="tensorops")

    assert "get_destination_cooperative_tensor<decltype(a), decltype(b), float>()" in msl
    assert "op.run(a_k, b_k, cTc);" in msl
    assert "fmax(float(cTc[element" in msl
    assert "cTc.store(c);" in msl
    assert stats.threadgroup_bytes == 0


def test_tensorops_dot_supports_transposed_rhs_tiles():
    msl = palladium.emit_msl(_blocked_dot_rhs_transposed(), dot_general="tensorops")

    assert "matmul2d_descriptor desc(16, 32, 16, false, true, false," in msl
    assert "arg1 + (int)_pid.y * 512" in msl
    assert "dextents<int, 2>(32, 16)" in msl
    assert "array<int, 2>{1, 16}" in msl


def test_tensorops_dot_supports_transposed_lhs_tiles():
    msl = palladium.emit_msl(_blocked_dot_lhs_transposed(), dot_general="tensorops")

    assert "matmul2d_descriptor desc(16, 32, 16, true, false, false," in msl
    assert "arg0 + (int)_pid.x * 16" in msl
    assert "dextents<int, 2>(32, 16)" in msl
    assert "array<int, 2>{1, 32}" in msl


def test_tensorops_dot_fuses_matrix_residual_add():
    msl, stats = palladium.emit.emit_msl_stats(_blocked_residual_dot(), dot_general="tensorops")

    assert "device float* arg2 [[buffer(2)]]" in msl
    assert "device float* arg3 [[buffer(3)]]" in msl
    assert "arg2 + (int)_pid.x * 1024 + (int)_pid.y * 32" in msl
    assert "(arg3 + (int)_pid.x * 1024 + (int)_pid.y * 32)[row * 64 + column]" in msl
    assert "dot_result[element]" in msl
    assert stats.threadgroup_bytes == 16 * 32 * 4


def test_tensorops_dot_fuses_column_bias():
    spec = _blocked_column_bias_dot()
    msl, stats = palladium.emit.emit_msl_stats(spec, dot_general="tensorops")
    eqns = spec.jaxpr.eqns
    reordered = dataclasses.replace(
        spec,
        jaxpr=Jaxpr(
            spec.jaxpr.constvars,
            spec.jaxpr.invars,
            spec.jaxpr.outvars,
            [eqns[3], eqns[0], eqns[1], eqns[2], *eqns[4:]],
            spec.jaxpr.effects,
            spec.jaxpr.debug_info,
            spec.jaxpr.is_high,
            spec.jaxpr.consts,
        ),
    )

    assert "device float* arg2 [[buffer(2)]]" in msl
    assert "device float* arg3 [[buffer(3)]]" in msl
    assert "arg2 + (int)_pid.y * 32" in msl
    assert "element % 32" in msl
    assert "(arg3 + (int)_pid.x * 1024 + (int)_pid.y * 32)[row * 64 + column]" in msl
    assert stats.threadgroup_bytes == 16 * 32 * 4
    assert palladium.emit_msl(reordered, dot_general="tensorops") == msl


def test_tensorops_batched_dot_maps_batch_and_output_tiles_to_threadgroups():
    msl = palladium.emit_msl(_blocked_batched_dot(), dot_general="tensorops")
    ordinary = palladium.emit_msl(_blocked_batched_dot(), dot_general="default")

    assert "uint3 _pid [[threadgroup_position_in_grid]]" in msl
    assert "matmul2d_descriptor desc(16, 32, 16" in msl
    assert "arg0 + (int)_pid.x * 512 + (int)_pid.y * 256" in msl
    assert "arg1 + (int)_pid.x * 1024 + (int)_pid.z * 32" in msl
    assert "arg2 + (int)_pid.x * 2048 + (int)_pid.y * 1024 + (int)_pid.z * 32" in msl
    assert "MetalPerformancePrimitives" not in ordinary
    assert "thread_position_in_grid" in ordinary


def test_tensorops_batched_dot_fuses_max_epilogue():
    msl, stats = palladium.emit.emit_msl_stats(_blocked_batched_relu_dot(), dot_general="tensorops")

    assert "get_destination_cooperative_tensor<decltype(a), decltype(b), float>()" in msl
    assert "arg2 + (int)_pid.x * 2048 + (int)_pid.y * 1024 + (int)_pid.z * 32" in msl
    assert "fmax(" in msl and "float(0.0f)" in msl
    assert stats.threadgroup_bytes == 0


def test_tensorops_batched_dot_fuses_matrix_residual_add():
    msl, stats = palladium.emit.emit_msl_stats(
        _blocked_batched_residual_dot(), dot_general="tensorops"
    )

    assert "device float* arg3 [[buffer(3)]]" in msl
    assert "arg2 + (int)_pid.x * 2048 + (int)_pid.y * 1024 + (int)_pid.z * 32" in msl
    assert (
        "(arg3 + (int)_pid.x * 2048 + (int)_pid.y * 1024 + (int)_pid.z * 32)[row * 64 + column]"
        in msl
    )
    assert stats.threadgroup_bytes == 16 * 32 * 4


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


def test_auto_selects_tensorops_for_tiled_dots_and_default_forces_the_primitive_path():
    auto = palladium.emit_msl(_blocked_dot())
    assert auto == palladium.emit_msl(_blocked_dot(), dot_general="tensorops")
    ordinary = palladium.emit_msl(_blocked_dot(), dot_general="default")
    assert "MetalPerformancePrimitives" not in ordinary
    assert "thread_position_in_grid" in ordinary


def test_auto_falls_back_to_the_primitive_path_for_dots_tensorops_rejects():
    """Two outputs are outside the cooperative matmul's contract; "auto"
    keeps the kernel on the one-thread-per-program emitter, while
    "tensorops" surfaces the rejection."""

    def kernel(a_ref, b_ref, o_ref, p_ref):
        product = jnp.dot(a_ref[...], b_ref[...])
        o_ref[...] = product
        p_ref[...] = product * 2.0

    out_spec = pl.BlockSpec((16, 32), lambda i, j: (i, j))
    call = pl.pallas_call(
        kernel,
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((16, 16), lambda i, j: (i, 0)),
            pl.BlockSpec((16, 32), lambda i, j: (0, j)),
        ],
        out_specs=(out_spec, out_spec),
        out_shape=(jax.ShapeDtypeStruct((32, 64), jnp.float32),) * 2,
    )
    spec = palladium.trace(call, np.zeros((32, 16), np.float32), np.zeros((16, 64), np.float32))
    auto = palladium.emit_msl(spec)
    assert "MetalPerformancePrimitives" not in auto
    assert auto == palladium.emit_msl(spec, dot_general="default")
    with pytest.raises(EmitError):
        palladium.emit_msl(spec, dot_general="tensorops")


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
