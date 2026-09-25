"""MSL emission for fused online-softmax attention."""

from __future__ import annotations

import dataclasses

from palladium.emit.cooperative import (
    CooperativeValue,
    elementwise_expression,
    emit_online_softmax_rows,
)
from palladium.emit.core import Cursor, CVal, Environment
from palladium.trace import KernelSpec

from ._shared import _kernel_source, _TensorView
from .attention_pattern import _match_attention_pattern


def emit_tensorops_attention(spec: KernelSpec, kernel_name: str | None = None) -> tuple[str, int]:
    """Emit the recognized Pallas online-softmax pattern through Cursor."""
    plan = _match_attention_pattern(spec)
    scan = plan.scan
    query_length, key_length = plan.query_length, plan.key_length
    heads, dim = plan.heads, plan.dim
    tile_q, tile_k, causal = plan.tile_q, plan.tile_k, plan.causal
    dots = plan.dots
    score_op, value_op = plan.score_op, plan.value_op
    online_softmax, final_div_eqn = plan.online_softmax, plan.final_div
    name = kernel_name or spec.name
    cursor = Cursor()
    cursor.emit(f"constexpr int BQ = {tile_q};")
    cursor.emit(f"constexpr int BK = {tile_k};")
    cursor.emit(f"constexpr int D = {dim};")
    cursor.emit(
        "const uint THREADS = threads_per_group.x * threads_per_group.y * threads_per_group.z;"
    )
    cursor.emit("const uint batch = group.x;")
    cursor.emit("const uint head = group.y;")
    cursor.emit("const uint q_start = group.z * BQ;")
    cursor.emit(f"const uint q_base = (batch * {query_length} * {heads} + head) * D;")
    cursor.emit(f"const uint kv_base = (batch * {key_length} * {heads} + head) * D;")

    scores = cursor.allocate("float", (tile_q, tile_k), name="scores", space="threadgroup")
    accumulator = cursor.allocate("float", (tile_q, dim), name="accumulator", space="threadgroup")
    row_max = cursor.allocate("float", (tile_q,), name="row_max", space="threadgroup")
    row_sum = cursor.allocate("float", (tile_q,), name="row_sum", space="threadgroup")
    q_storage = CVal("query", (query_length * heads * dim,), "float", space="device", readonly=True)
    k_storage = CVal("key", (key_length * heads * dim,), "float", space="device", readonly=True)
    v_storage = CVal("value", (key_length * heads * dim,), "float", space="device", readonly=True)
    out_tile = CVal(
        f"(output + q_base + q_start * {heads * dim})",
        (tile_q * heads * dim,),
        "float",
        space="device",
    )

    with cursor.strided_loop("tid", "BQ * D", "THREADS", name="i"):
        cursor.emit(f"{accumulator.at('i')} = 0.0f;")
    with cursor.strided_loop("tid", "BQ", "THREADS", name="row"):
        cursor.emit(f"{row_max.at('row')} = -INFINITY;")
        cursor.emit(f"{row_sum.at('row')} = 0.0f;")
    cursor.barrier()
    score_op.emit_declaration(cursor)
    value_op.emit_declaration(cursor)

    key_loop_stop = "(((q_start + BQ + BK - 1) / BK) * BK)" if causal else str(key_length)
    with cursor.strided_loop("0", key_loop_stop, "BK", name="k_start"):
        q_tile = _TensorView(
            dataclasses.replace(q_storage, expr=f"query + q_base + q_start * {heads * dim}"),
            (tile_q, dim),
            ("D", "BQ"),
            (1, heads * dim),
        ).emit(cursor, "q_tile")
        k_tile = _TensorView(
            dataclasses.replace(k_storage, expr=f"key + kv_base + k_start * {heads * dim}"),
            (tile_k, dim),
            ("D", "BK"),
            (1, heads * dim),
        ).emit(cursor, "k_tile")
        score_tile = _TensorView(scores, (tile_q, tile_k), ("BK", "BQ"), (1, "BK")).emit(
            cursor, "score_tile"
        )
        score_op.emit_run(cursor, q_tile, k_tile, score_tile)
        cursor.barrier()

        score_tensor = CooperativeValue(scores, "tensorops")
        scale_operands = tuple(
            score_tensor if atom is dots[0].outvars[0] else Environment().val(scan.invars[3])
            for atom in online_softmax.score_scale.invars
        )
        emit_online_softmax_rows(
            cursor,
            online_softmax,
            score_tensor,
            row_max,
            row_sum,
            accumulator,
            scale_operands,
            rows=tile_q,
            columns=tile_k,
            width=dim,
            thread_count="THREADS",
            causal_offsets=("k_start", "q_start") if causal else None,
        )
        cursor.barrier()

        probability_tile = _TensorView(scores, (tile_q, tile_k), ("BK", "BQ"), (1, "BK")).emit(
            cursor, "probability_tile"
        )
        value_tile = _TensorView(
            dataclasses.replace(v_storage, expr=f"value + kv_base + k_start * {heads * dim}"),
            (tile_k, dim),
            ("D", "BK"),
            (1, heads * dim),
        ).emit(cursor, "value_tile")
        output_tile = _TensorView(accumulator, (tile_q, dim), ("D", "BQ"), (1, "D")).emit(
            cursor, "output_tile"
        )
        value_op.emit_run(cursor, probability_tile, value_tile, output_tile)
        cursor.barrier()

    with cursor.strided_loop("tid", "BQ * D", "THREADS", name="i"):
        cursor.emit("const uint row = i / D;")
        normalized = elementwise_expression(
            final_div_eqn,
            "float",
            (
                (final_div_eqn.invars[0], accumulator.at("i")),
                (final_div_eqn.invars[1], row_sum.at("row")),
            ),
        )
        cursor.emit(f"{out_tile.at(f'row * {heads * dim} + i % D')} = {normalized};")

    params = (
        "device float* query [[buffer(0)]]",
        "device float* key [[buffer(1)]]",
        "device float* value [[buffer(2)]]",
        "device float* output [[buffer(3)]]",
        "uint3 group [[threadgroup_position_in_grid]]",
        "uint tid [[thread_index_in_threadgroup]]",
        "uint3 threads_per_group [[threads_per_threadgroup]]",
    )
    source = _kernel_source(name, params, cursor.lines)
    return source, cursor.threadgroup_bytes
