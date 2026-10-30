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
    """Check fused score/value TensorOps, online softmax, and causal masking within one cooperative
    program."""
    spec = _spec(causal=causal)
    source, stats = palladium.emit.emit_msl_stats(spec, dot_general="tensorops")

    # The score product reads K transposed; the value product does not.
    assert source.count(".run(") == 2
    assert "16, 16, 64, false, true, false);" in source
    assert "16, 64, 16, false, false, false," in source
    assert "tensor<const device float" not in source
    assert "array<int, 2>{{" not in source
    assert "exp((new_max - new_max))" not in source
    # The accumulator is the value matmul's cooperative tensor, not shared memory.
    assert "acc.get_multidimensional_index(" in source
    assert "threadgroup float accumulator" not in source
    assert source.count("threadgroup_barrier(mem_flags::mem_threadgroup)") == 4
    if causal:
        assert "k_start < (((q_start + BQ + BK - 1) / BK) * BK)" in source
    else:
        assert "k_start < 32" in source
    assert ("(k_start + lane) <= (q_start + row)" in source) is causal
    assert stats.thread_bytes == 0
    assert stats.threadgroup_bytes == (16 * 16 + 3 * 16) * 4


@pytest.mark.parametrize(
    ("query_length", "key_length", "heads", "head_dim", "tile_k"),
    [(32, 64, 2, 64, 16), (16, 4096, 8, 16, 64)],  # the second: SAM2's token-to-image attention
)
def test_tensorops_attention_supports_distinct_query_and_key_lengths(
    query_length, key_length, heads, head_dim, tile_k
):
    """Check that noncausal cooperative attention addresses and loops over keys by their own
    length."""
    spec = _cross_attention_spec(
        query_length, key_length, heads=heads, head_dim=head_dim, tile_q=16, tile_k=tile_k
    )
    source = palladium.emit_msl(spec, dot_general="tensorops")

    assert f"const uint q_base = (batch * {query_length} * {heads} + head) * D;" in source
    assert f"const uint kv_base = (batch * {key_length} * {heads} + head) * D;" in source
    assert f"k_start < {key_length}" in source


def test_tensorops_attention_matches_dataflow_after_independent_equations_reorder():
    """Check that attention recognition follows data dependencies when independent equations are
    reordered."""
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
    """Verify that rewiring probability operands is rejected even when primitive names are
    unchanged."""
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


def test_tensorops_attention_explain_scales_batch_axis_for_cooperative_groups():
    """Check that attention diagnostics report the cooperative group and scaled batch grid axis."""
    diagnostics = explain_spec(_spec(), dot_general="tensorops")

    assert diagnostics.grid == (128, 2, 2)
    assert diagnostics.threadgroup == (128, 1, 1)
    assert diagnostics.threadgroup_bytes == (16 * 16 + 3 * 16) * 4
    assert diagnostics.cooperative


def test_tensorops_attention_emits_large_query_tile_with_bounded_shared_memory():
    """Check that a 64-row query tile keeps shared storage to the score tile and row state."""
    shape = (1, 64, 1, 64)
    call = attention_call(shape, tile_q=64, tile_k=32)
    args = (jax.ShapeDtypeStruct(shape, jnp.float32),) * 3
    _, stats = palladium.emit.emit_msl_stats(palladium.trace(call, *args), dot_general="tensorops")

    assert stats.threadgroup_bytes == (64 * 32 + 3 * 64) * 4


@pytest.mark.parametrize("head_dim", [16, 48])
def test_tensorops_attention_supports_multiples_of_sixteen_head_dimensions(head_dim):
    """Check that the matmul2d descriptors and the softmax scale follow the head dimension."""
    source = palladium.emit_msl(_spec(head_dim=head_dim), dot_general="tensorops")

    assert f"16, 16, {head_dim}, false, true, false);" in source
    assert f"16, {head_dim}, 16, false, false, false," in source
    literals = [float(x) for x in re.findall(r"\b(\d*\.\d+(?:e-?\d+)?)f\b", source)]
    assert any(x == pytest.approx(head_dim**-0.5, rel=1e-6) for x in literals)


@pytest.mark.parametrize(("tile_q", "head_dim"), [(8, 64), (16, 24)])
def test_matmul2d_attention_rejects_sizes_off_multiples_of_sixteen(tile_q, head_dim):
    """Verify that the matmul2d schedule rejects query tiles and head dimensions that are not
    multiples of sixteen."""
    with pytest.raises(EmitError, match="multiples of 16"):
        palladium.emit_msl(_spec(tile_q=tile_q, head_dim=head_dim), dot_general="tensorops")


@pytest.mark.parametrize("causal", [False, True])
def test_simdgroup_attention_keeps_the_softmax_in_registers(causal):
    """Under `dot_general="simdgroup"`, both products use `simdgroup_matrix`, scores
    never touch threadgroup memory, and a key step has only the two barriers around
    staging K and V; 16-row query tiles keep the `matmul2d` schedule."""
    spec = _spec(causal=causal, tile_q=32, tile_k=16)
    msl, stats = palladium.emit.emit_msl_stats(spec, dot_general="simdgroup")
    assert "simdgroup_mma" in msl
    assert "matmul2d" not in msl and "scores" not in msl
    assert msl.count("threadgroup_barrier") == 2
    assert stats.threadgroup_bytes == 2 * 16 * (64 + 4) * 4
    assert ("> q_start" in msl) == causal  # masking only blocks crossing the diagonal
    assert "matmul2d" in palladium.emit_msl(_spec(causal=causal), dot_general="simdgroup")
