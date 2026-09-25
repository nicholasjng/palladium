"""Codegen contract for the supported Pallas online-softmax pattern."""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl

import palladium
from palladium.emit import EmitError
from palladium.workloads.pallas_flash_attention import (
    attention_kernel,
    attention_specs,
    make_pallas_flash_attention,
)


def _spec(*, causal: bool = False, tile_q: int = 16, tile_k: int = 16):
    shape = (1, 32, 2, 64)
    call = make_pallas_flash_attention(shape, tile_q=tile_q, tile_k=tile_k, causal=causal)
    args = (jax.ShapeDtypeStruct(shape, jnp.float32),) * 3
    return palladium.trace(call, *args)


@pytest.mark.parametrize("causal", [False, True])
def test_tensorops_attention_fuses_both_dots_into_one_threadgroup(causal):
    spec = _spec(causal=causal)
    source, stats = palladium.emit.emit_msl_stats(spec, dot_general="tensorops")

    assert source.count(".run(") == 2
    assert "score_op.run(q_tile, k_tile, score_tile)" in source
    assert "value_op.run(probability_tile, value_tile, output_tile)" in source
    assert "uint3 group [[threadgroup_position_in_grid]]" in source
    assert "tensor<const device float" not in source
    assert "tensor<device float, dextents<int, 2>, tensor_inline>" in source
    assert "group.z * BQ" in source
    assert "group.x;" in source and "group.y;" in source
    assert source.count("threadgroup_barrier(mem_flags::mem_threadgroup)") == 4
    assert ("k_start + col > q_start + row" in source) is causal
    assert stats.thread_bytes == 0
    assert stats.threadgroup_bytes == (16 * 16 + 16 * 64 + 2 * 16) * 4


def test_tensorops_attention_explain_scales_batch_axis_for_cooperative_groups():
    diagnostics = palladium.explain_spec(_spec(), dot_general="tensorops")

    assert diagnostics.grid == (128, 2, 2)
    assert diagnostics.threadgroup == (128, 1, 1)
    assert diagnostics.threadgroup_bytes == (16 * 16 + 16 * 64 + 2 * 16) * 4
    assert diagnostics.cooperative


def test_tensorops_attention_emits_benchmark_tile_configuration():
    shape = (1, 128, 4, 64)
    grid, in_specs, out_specs = attention_specs(shape[0], shape[1], shape[2], tile_q=32)
    call = pl.pallas_call(
        attention_kernel(tile_q=32, tile_k=64),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
    )
    args = (jax.ShapeDtypeStruct(shape, jnp.float32),) * 3
    source = palladium.emit_msl(palladium.trace(call, *args), dot_general="tensorops")

    assert "constexpr int BQ = 32;" in source
    assert "constexpr int BK = 64;" in source
    assert source.count(".run(") == 2


def test_tensorops_attention_rejects_unsupported_tile_shape():
    shape = (1, 32, 2, 64)
    call = make_pallas_flash_attention(shape, tile_q=8, tile_k=16)
    args = (jax.ShapeDtypeStruct(shape, jnp.float32),) * 3
    spec = palladium.trace(call, *args)

    with pytest.raises(EmitError, match="multiples of 16"):
        palladium.emit_msl(spec, dot_general="tensorops")
