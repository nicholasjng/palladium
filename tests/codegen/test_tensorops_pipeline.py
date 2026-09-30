"""Contracts for the cooperative Metal TensorOps frontend."""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl

from palladium.emit.tensorops import (
    ProgramScope,
    assign_layouts,
    compile_kernel,
    import_kernel,
    plan_kernel,
)
from palladium.trace import trace
from palladium.workloads.pallas_flash_attention import make_pallas_flash_attention


def _tensorops_dot(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.matmul(a_ref[...], b_ref[...])


def _tensorops_relu_dot(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.maximum(jnp.matmul(a_ref[...], b_ref[...]), 0.0)


def _tensorops_chained_dot(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.maximum(jnp.matmul(a_ref[...], b_ref[...]), 0.0) + 1.0


def _tensorops_elementwise_chain(input_ref, residual_ref, output_ref):
    output_ref[...] = jnp.maximum(input_ref[...], 0.0) * 2.0 + residual_ref[...]


def _tensorops_broadcast_elementwise(input_ref, row_bias_ref, scalar_bias_ref, output_ref):
    output_ref[...] = (
        jnp.maximum(input_ref[...], 0.0) * 2.0 + row_bias_ref[...] + scalar_bias_ref[()]
    )


def _tensorops_row_reduction_spec(reduction, m=5, n=8, tm=4):
    def kernel(input_ref, output_ref):
        output_ref[...] = reduction(input_ref[...], axis=1)

    call = pl.pallas_call(
        kernel,
        grid=((m + tm - 1) // tm,),
        in_specs=[pl.BlockSpec((tm, n), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((tm,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((m,), jnp.float32),
    )
    return trace(call, jax.ShapeDtypeStruct((m, n), jnp.float32))


def _tensorops_elementwise_spec(m=9, n=17, tm=4, tn=8):
    call = pl.pallas_call(
        _tensorops_elementwise_chain,
        grid=((m + tm - 1) // tm, (n + tn - 1) // tn),
        in_specs=[
            pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
            pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
    )
    args = [
        jax.ShapeDtypeStruct((m, n), jnp.float32),
        jax.ShapeDtypeStruct((m, n), jnp.float32),
    ]
    return trace(call, *args)


def _tensorops_broadcast_elementwise_spec(m=8, n=16, tm=4, tn=8):
    call = pl.pallas_call(
        _tensorops_broadcast_elementwise,
        grid=(m // tm, n // tn),
        in_specs=[
            pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
            pl.BlockSpec((tn,), lambda i, j: (j,)),
            pl.BlockSpec((), lambda i, j: ()),
        ],
        out_specs=pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
    )
    args = [
        jax.ShapeDtypeStruct((m, n), jnp.float32),
        jax.ShapeDtypeStruct((n,), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.float32),
    ]
    return trace(call, *args)


def _tensorops_matmul_spec(
    kernel=_tensorops_dot, m=32, n=64, k=16, tm=16, tn=32, dtype=jnp.float32
):
    in_specs = [
        pl.BlockSpec((tm, k), lambda i, j: (i, 0)),
        pl.BlockSpec((k, tn), lambda i, j: (0, j)),
    ]
    args = [
        jax.ShapeDtypeStruct((m, k), dtype),
        jax.ShapeDtypeStruct((k, n), dtype),
    ]
    call = pl.pallas_call(
        kernel,
        grid=((m + tm - 1) // tm, (n + tn - 1) // tn),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
    )
    return trace(call, *args)


def test_tensorops_import_preserves_scan_regions_and_tensor_shapes():
    shape = (1, 32, 2, 16)
    call = make_pallas_flash_attention(shape, tile_q=16, tile_k=16)
    args = [jax.ShapeDtypeStruct(shape, jnp.float32)] * 3
    spec = trace(call, *args)

    kernel = import_kernel(plan_kernel(spec, scope=ProgramScope.THREADGROUP))

    assert kernel.plan.grid == (1, 2, 2)
    assert kernel.plan.program_id_attribute == "threadgroup_position_in_grid"
    assert tuple(layout.shape for layout in kernel.plan.inputs) == (
        (1, 16, 1, 16),
        (1, 32, 1, 16),
        (1, 32, 1, 16),
    )
    scan = next(op for op in kernel.body.operations if op.name == "scan")
    assert len(scan.regions) == 1
    assert [op.name for op in scan.regions[0].operations].count("dot_general") == 2

    laid_out = assign_layouts(kernel)
    scan = next(op for op in laid_out.body.operations if op.name == "scan")
    dots = [op for op in scan.regions[0].operations if op.name == "dot_general"]
    assert all(op.results[0].layout.distribution.value == "tensorops" for op in dots)
    reductions = [
        op for op in scan.regions[0].operations if op.name in ("reduce_max", "reduce_sum")
    ]
    assert all(op.results[0].layout.distribution.value == "row_strided" for op in reductions)

    compilation = compile_kernel(spec, scope=ProgramScope.THREADGROUP)
    assert compilation.plan.scope is ProgramScope.THREADGROUP
    assert "threadgroup_position_in_grid" in compilation.source
    assert compilation.threadgroup_bytes > 0


def test_tensorops_defaults_to_thread_scope_for_scalar_programs():
    shape = (4,)

    def identity(src_ref, dst_ref):
        dst_ref[...] = src_ref[...]

    import jax.experimental.pallas as pl

    call = pl.pallas_call(
        identity,
        grid=(4,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
    )
    spec = trace(call, jax.ShapeDtypeStruct(shape, jnp.float32))

    plan = plan_kernel(spec)
    assert plan.scope is ProgramScope.THREAD
    assert plan.program_id_attribute == "thread_position_in_grid"
    compilation = compile_kernel(spec)
    assert "thread_position_in_grid" in compilation.source


def test_tensorops_matmul_lowering_emits_from_ir():
    spec = _tensorops_matmul_spec()
    compilation = compile_kernel(spec, scope=ProgramScope.THREADGROUP)

    assert "threadgroup_position_in_grid" in compilation.source
    assert "execution_simdgroups<4>" in compilation.source
    assert "matmul2d_descriptor desc(16, 32, 16" in compilation.source
    assert "device float* arg0 [[buffer(0)]]" in compilation.source
    assert "const device float*" not in compilation.source
    assert compilation.threadgroup_bytes == 0


def test_tensorops_matmul_accumulates_k_in_tensorops_tiles_and_handles_tail():
    compilation = compile_kernel(_tensorops_matmul_spec(k=144), scope=ProgramScope.THREADGROUP)

    assert "matmul2d_descriptor desc(16, 32, 128, false, false, false," in compilation.source
    assert "mode::multiply_accumulate" in compilation.source
    assert "for (int k_start = 0; k_start < 144; k_start += 128)" in compilation.source
    assert "min(128, 144 - k_start)" in compilation.source
    assert "op.run(a_k, b_k, cTc);" in compilation.source
    assert "cTc[init0] = 0.0f;" in compilation.source


def test_tensorops_matmul_masks_partial_output_tiles():
    compilation = compile_kernel(
        _tensorops_matmul_spec(_tensorops_relu_dot, m=30, n=45, k=32),
        scope=ProgramScope.THREADGROUP,
    )

    assert compilation.plan.grid == (2, 2)
    assert "min(16, 30 - (int)_pid.x * 16)" in compilation.source
    assert "min(32, 45 - (int)_pid.y * 32)" in compilation.source
    assert "cTc.store(c_edge);" in compilation.source
    assert "if (row < min(16, 30 - (int)_pid.x * 16)" in compilation.source
    assert "threadgroup float edge_result[512];" in compilation.source
    assert "cTc[element1] = tensorops_epilogue2;" in compilation.source
    assert "(arg2 + (int)_pid.x * 720 + (int)_pid.y * 32)[row * 45 + column]" in (
        compilation.source
    )


@pytest.mark.parametrize(
    ("dtype", "metal_type"),
    ((jnp.float16, "half"), (jnp.bfloat16, "bfloat")),
)
def test_tensorops_matmul_accepts_half_precision_buffer_types(dtype, metal_type):
    compilation = compile_kernel(
        _tensorops_matmul_spec(k=32, dtype=dtype), scope=ProgramScope.THREADGROUP
    )

    assert f"device {metal_type}* arg0 [[buffer(0)]]" in compilation.source
    assert f"device {metal_type}* arg2 [[buffer(2)]]" in compilation.source
    # Products accumulate in float and narrow on the per-element store.
    assert "get_destination_cooperative_tensor<decltype(a), decltype(b), float>()" in (
        compilation.source
    )
    assert "threadgroup float dot_result[512];" in compilation.source
    assert f"= {metal_type}(dot_result[element]);" in compilation.source


def test_tensorops_matmul_lowering_composes_chained_epilogues():
    chained = compile_kernel(
        _tensorops_matmul_spec(_tensorops_chained_dot), scope=ProgramScope.THREADGROUP
    )
    assert "threadgroup float tensorops_value_0[512];" not in chained.source
    assert "cTc[element1] = tensorops_epilogue3;" in chained.source
    assert "fmax" in chained.source
    assert chained.threadgroup_bytes == 0


@pytest.mark.parametrize(
    ("reduction", "initial", "combine"),
    ((jnp.sum, "0.0f", "reduce0 +"), (jnp.max, "-INFINITY", "max(reduce0,")),
)
def test_tensorops_lowers_standalone_cooperative_row_reductions(reduction, initial, combine):
    spec = _tensorops_row_reduction_spec(reduction)

    compilation = compile_kernel(spec, scope=ProgramScope.THREADGROUP)

    assert compilation.plan.grid == (2,)
    assert "threadgroup_position_in_grid" in compilation.source
    assert "simdgroup_index_in_threadgroup" in compilation.source
    assert "simdgroups_per_threadgroup" in compilation.source
    assert "thread_index_in_simdgroup" in compilation.source
    assert "for (uint column = lane; column < 8; column += (threads_per_group.x / simdgroups))" in (
        compilation.source
    )
    assert "simd_sum(reduce0)" in compilation.source or "simd_max(reduce0)" in compilation.source
    assert "if (row < min(4, 5 - (int)_pid.x * 4))" in compilation.source
    assert f"float reduce0 = {initial};" in compilation.source
    assert combine in compilation.source
    assert "arg1[(int)_pid.x * 4 + row] = reduce0;" in compilation.source
    assert compilation.threadgroup_bytes == 0


def test_tensorops_row_reduction_simd_lanes_cover_wide_rows_and_partial_output_rows():
    spec = _tensorops_row_reduction_spec(jnp.sum, m=19, n=257, tm=5)

    compilation = compile_kernel(spec, scope=ProgramScope.THREADGROUP)

    assert (
        "for (uint column = lane; column < 257; column += (threads_per_group.x / simdgroups))"
        in (compilation.source)
    )
    assert "if (row < min(5, 19 - (int)_pid.x * 5))" in compilation.source
    assert "if (lane == 0)" in compilation.source


def test_tensorops_lowers_cooperative_elementwise_chains_from_jaxpr():
    spec = _tensorops_elementwise_spec()

    compilation = compile_kernel(spec, scope=ProgramScope.THREADGROUP)

    assert compilation.plan.grid == (3, 3)
    assert "threadgroup_position_in_grid" in compilation.source
    assert "thread_position_in_grid" not in compilation.source
    assert "for (uint element = tid; element < 32; element += THREADS)" in (compilation.source)
    assert "tensorops_input" in compilation.source
    assert "tensorops_value" in compilation.source
    assert "max(tensorops_input0, 0.0f)" in compilation.source
    assert (
        "((int)_pid.x * 4 + ((element / 8) % 4)) < 9 && ((int)_pid.y * 8 + (element % 8)) < 17"
    ) in compilation.source
    tile_offset = "(((element / 8) % 4) * 17) + (element % 8)"
    assert f"arg0[(int)_pid.x * 68 + (int)_pid.y * 8 + {tile_offset}]" in compilation.source
    assert f"arg2[(int)_pid.x * 68 + (int)_pid.y * 8 + {tile_offset}]" in compilation.source
    assert compilation.threadgroup_bytes == 0


def test_tensorops_lowers_scalar_and_row_vector_elementwise_broadcasts():
    spec = _tensorops_broadcast_elementwise_spec()

    compilation = compile_kernel(spec, scope=ProgramScope.THREADGROUP)

    assert "arg1[(int)_pid.y * 8 + (element % 8)]" in compilation.source
    assert "arg2[0]" in compilation.source
    assert "arg3[(int)_pid.x * 64 + (int)_pid.y * 8" in compilation.source
    assert "broadcast_in_dim" not in compilation.source
