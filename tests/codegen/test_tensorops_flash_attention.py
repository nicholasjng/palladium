"""Codegen contract for the supported Pallas online-softmax pattern."""

import dataclasses
import re

import jax
import jax.numpy as jnp
import pytest
from flash_attention import attention_call
from jax.extend.core import Jaxpr

import palladium
from palladium.diagnostics import explain_spec
from palladium.errors import EmitError


def _spec(*, causal: bool = False, tile_q: int = 16, tile_k: int = 16, head_dim: int = 64):
    shape = (1, 32, 2, head_dim)
    call = attention_call(shape, tile_q=tile_q, tile_k=tile_k, causal=causal)
    args = (jax.ShapeDtypeStruct(shape, jnp.float32),) * 3
    return palladium.trace(call, *args)


def _cross_attention_spec(
    query_length: int = 32,
    key_length: int = 64,
    *,
    heads: int = 2,
    head_dim: int = 64,
    tile_q: int = 16,
    tile_k: int = 16,
):
    q_shape = (1, query_length, heads, head_dim)
    kv_shape = (1, key_length, heads, head_dim)
    call = attention_call(q_shape, tile_q=tile_q, tile_k=tile_k, key_length=key_length)
    args = (jax.ShapeDtypeStruct(q_shape, jnp.float32),) + (
        jax.ShapeDtypeStruct(kv_shape, jnp.float32),
    ) * 2
    return palladium.trace(call, *args)


def _replace_scan_body(spec, transform, transform_outer=None):
    eqns = list(spec.jaxpr.eqns)
    scan_index = next(i for i, eqn in enumerate(eqns) if eqn.primitive.name == "scan")
    scan = eqns[scan_index]
    body = scan.params["jaxpr"]
    body = Jaxpr(
        body.constvars,
        body.invars,
        body.outvars,
        transform(list(body.eqns)),
        body.effects,
        body.debug_info,
        body.is_high,
        body.consts,
    )
    eqns[scan_index] = scan.replace(params={**scan.params, "jaxpr": body})
    if transform_outer is not None:
        eqns = transform_outer(eqns)
    outer = Jaxpr(
        spec.jaxpr.constvars,
        spec.jaxpr.invars,
        spec.jaxpr.outvars,
        eqns,
        spec.jaxpr.effects,
        spec.jaxpr.debug_info,
        spec.jaxpr.is_high,
        spec.jaxpr.consts,
    )
    return dataclasses.replace(spec, jaxpr=outer)


@pytest.mark.parametrize("causal", [False, True])
def test_tensorops_attention_fuses_both_dots_into_one_threadgroup(causal):
    spec = _spec(causal=causal)
    source, stats = palladium.emit.emit_msl_stats(spec, dot_general="tensorops")

    assert source.count(".run(") == 2
    assert "score_op.run(q_tile, k_tile, score_tile)" in source
    assert "value_op.run(score_tile, value_tile, acc)" in source
    assert "16, 16, 64, false, true, false);" in source
    assert "16, 64, 16, false, false, false," in source
    assert "uint3 group [[threadgroup_position_in_grid]]" in source
    assert "tensor<const device float" not in source
    assert "tensor<device float, dextents<int, 2>, tensor_inline>" in source
    assert "threadgroup float scores[256];" in source
    assert "array<int, 2>{1, 128}" in source
    assert "array<int, 2>{{" not in source
    assert "(output + q_base + q_start * 128)[row * 128 + column]" in source
    assert "group.z * BQ" in source
    assert "group.x;" in source and "group.y;" in source
    # Rows belong to SIMD groups, lanes to columns; the accumulator is the
    # value matmul's cooperative tensor, addressed by MPP coordinates.
    assert "for (uint row = sg; row < 16; row += 4)" in source
    assert "scores[row * 16 + lane]" in source
    assert " * 0.125f" in source
    assert "simd_max(lane_max)" in source and "simd_sum(lane_sum)" in source
    assert "= exp((s[0] - new_max));" in source
    assert "exp((new_max - new_max))" not in source
    assert "if (lane == 0)" in source and "row_scale[row] = old_scale;" in source
    assert "acc.get_multidimensional_index(" in source
    assert "acc[_i1] = (row_scale[acc_row] * acc[_i1]);" in source
    assert "threadgroup float accumulator" not in source
    assert source.count("threadgroup_barrier(mem_flags::mem_threadgroup)") == 4
    if causal:
        assert "k_start < (((q_start + BQ + BK - 1) / BK) * BK)" in source
    else:
        assert "k_start < 32" in source
    assert ("(k_start + lane) <= (q_start + row)" in source) is causal
    assert stats.thread_bytes == 0
    assert stats.threadgroup_bytes == (16 * 16 + 3 * 16) * 4


def test_tensorops_attention_supports_distinct_query_and_key_lengths():
    source = palladium.emit_msl(_cross_attention_spec(), dot_general="tensorops")

    assert "const uint q_base = (batch * 32 * 2 + head) * D;" in source
    assert "const uint kv_base = (batch * 64 * 2 + head) * D;" in source
    assert "k_start < 64" in source
    assert "key + kv_base + k_start * 128" in source
    assert "(output + q_base + q_start * 128)" in source


def test_tensorops_attention_matches_dataflow_after_independent_equations_reorder():
    spec = _spec()
    source = palladium.emit_msl(spec, dot_general="tensorops")

    def reorder_reads(eqns):
        eqns[2], eqns[3] = eqns[3], eqns[2]
        return eqns

    def reorder_initial_states(eqns):
        eqns[1], eqns[3] = eqns[3], eqns[1]
        return eqns

    reordered = _replace_scan_body(spec, reorder_reads, reorder_initial_states)
    assert palladium.emit_msl(reordered, dot_general="tensorops") == source


def test_tensorops_attention_rejects_same_primitives_with_wrong_probability_wiring():
    spec = _spec()

    def swap_probability_operands(eqns):
        producers = {var: eqn for eqn in eqns for var in eqn.outvars}
        probability_exp = next(
            eqn
            for eqn in eqns
            if eqn.primitive.name == "exp" and len(eqn.outvars[0].aval.shape) == 2
        )
        center = producers[probability_exp.invars[0]]
        eqns[eqns.index(center)] = center.replace(invars=center.invars[::-1])
        return eqns

    malformed = _replace_scan_body(spec, swap_probability_operands)
    with pytest.raises(EmitError, match="softmax probabilities"):
        palladium.emit_msl(malformed, dot_general="tensorops")


def test_tensorops_attention_matches_sam2_token_to_image_shape():
    spec = _cross_attention_spec(
        query_length=16,
        key_length=4096,
        heads=8,
        head_dim=16,
        tile_q=16,
        tile_k=64,
    )
    source = palladium.emit_msl(spec, dot_general="tensorops")

    assert "constexpr int BQ = 16;" in source
    assert "constexpr int BK = 64;" in source
    assert "constexpr int D = 16;" in source
    assert "k_start < 4096" in source
    assert "const uint kv_base = (batch * 4096 * 8 + head) * D;" in source


def test_tensorops_attention_explain_scales_batch_axis_for_cooperative_groups():
    diagnostics = explain_spec(_spec(), dot_general="tensorops")

    assert diagnostics.grid == (128, 2, 2)
    assert diagnostics.threadgroup == (128, 1, 1)
    assert diagnostics.threadgroup_bytes == (16 * 16 + 3 * 16) * 4
    assert diagnostics.cooperative


def test_tensorops_attention_emits_benchmark_tile_configuration():
    shape = (1, 128, 4, 64)
    call = attention_call(shape, tile_q=32, tile_k=64)
    args = (jax.ShapeDtypeStruct(shape, jnp.float32),) * 3
    source = palladium.emit_msl(palladium.trace(call, *args), dot_general="tensorops")

    assert "constexpr int BQ = 32;" in source
    assert "constexpr int BK = 64;" in source
    assert source.count(".run(") == 2


def test_tensorops_attention_emits_large_query_tile_with_bounded_shared_memory():
    shape = (1, 64, 1, 64)
    call = attention_call(shape, tile_q=64, tile_k=32)
    args = (jax.ShapeDtypeStruct(shape, jnp.float32),) * 3
    source, stats = palladium.emit.emit_msl_stats(
        palladium.trace(call, *args), dot_general="tensorops"
    )

    assert "constexpr int BQ = 64;" in source
    assert "constexpr int BK = 32;" in source
    assert stats.threadgroup_bytes == (64 * 32 + 3 * 64) * 4


@pytest.mark.parametrize("head_dim", [16, 32, 48, 64])
def test_tensorops_attention_supports_multiples_of_sixteen_head_dimensions(head_dim):
    source = palladium.emit_msl(_spec(head_dim=head_dim), dot_general="tensorops")

    assert f"constexpr int D = {head_dim};" in source
    assert f"16, 16, {head_dim}, false, true, false);" in source
    assert f"16, {head_dim}, 16, false, false, false," in source
    scale_line = next(line for line in source.splitlines() if "scores[row * 16 + lane] *" in line)
    scale = float(re.search(r"\* ([0-9.e-]+)f\)", scale_line).group(1))
    assert scale == pytest.approx(head_dim**-0.5, rel=1e-6)


def test_tensorops_attention_rejects_unsupported_tile_shape():
    shape = (1, 32, 2, 64)
    call = attention_call(shape, tile_q=8, tile_k=16)
    args = (jax.ShapeDtypeStruct(shape, jnp.float32),) * 3
    spec = palladium.trace(call, *args)

    with pytest.raises(EmitError, match="multiples of 16"):
        palladium.emit_msl(spec, dot_general="tensorops")


def test_tensorops_attention_rejects_unsupported_head_dimension():
    with pytest.raises(EmitError, match="head dimensions must be multiples of 16"):
        palladium.emit_msl(_spec(head_dim=24), dot_general="tensorops")
