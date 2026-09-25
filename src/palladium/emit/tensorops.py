"""Narrow TensorOps lowerings for tiled matrix products and attention."""

from __future__ import annotations

from jax.extend.core import Jaxpr, Literal, Var, subjaxprs

from palladium.errors import EmitError
from palladium.trace import BlockInfo, KernelSpec

SIMDGROUPS = 4


def _index_map_is(info: BlockInfo, axes: tuple[int | None, ...]) -> bool:
    """Whether the map returns the named grid axes or integer literals."""
    jaxpr = info.index_map_jaxpr.jaxpr
    if jaxpr.eqns or len(jaxpr.invars) < max((a for a in axes if a is not None), default=-1) + 1:
        return False
    if len(jaxpr.outvars) != len(axes):
        return False
    for outvar, axis in zip(jaxpr.outvars, axes, strict=True):
        if axis is None:
            if not isinstance(outvar, Literal) or int(outvar.val) != 0:
                return False
        elif outvar is not jaxpr.invars[axis]:
            return False
    return True


def has_dot_general(jaxpr: Jaxpr) -> bool:
    """Whether a jaxpr contains a dot, including in nested control flow."""
    return any(eqn.primitive.name == "dot_general" for eqn in jaxpr.eqns) or any(
        has_dot_general(child) for child in subjaxprs(jaxpr)
    )


def uses_tensorops(spec: KernelSpec, dot_general: str) -> bool:
    """Whether the requested lowering applies to this kernel."""
    return dot_general == "tensorops" and has_dot_general(spec.jaxpr)


def _is_empty_get(eqn, ref: Var) -> bool:
    return (
        eqn.primitive.name == "get"
        and len(eqn.invars) == 1
        and eqn.invars[0] is ref
        and not eqn.params["tree"].flatten_up_to(())
    )


def emit_tensorops_matmul(spec: KernelSpec, kernel_name: str | None = None) -> str:
    """Emit a group-cooperative MSL kernel for a tiled matrix product."""
    eqns = spec.jaxpr.eqns
    if (
        len(spec.inputs) != 2
        or len(spec.outputs) != 1
        or spec.scratch
        or spec.aliases
        or len(spec.grid) not in (2, 3)
        or len(spec.jaxpr.invars) != 3
    ):
        raise EmitError(
            "dot_general='tensorops' requires one full-block matmul whose result is "
            "stored directly to one output"
        )

    lhs_ref, rhs_ref, out_ref = spec.jaxpr.invars
    if len(spec.grid) == 2:
        if tuple(e.primitive.name for e in eqns) != ("get", "get", "dot_general", "swap"):
            raise EmitError("dot_general='tensorops' requires a direct matrix product")
        lhs_get, rhs_get, dot, store = eqns
        if not _is_empty_get(lhs_get, lhs_ref) or not _is_empty_get(rhs_get, rhs_ref):
            raise EmitError("dot_general='tensorops' requires full-block input reads")
        if dot.invars != [lhs_get.outvars[0], rhs_get.outvars[0]]:
            raise EmitError("dot_general='tensorops' does not support transformed dot operands")
        if store.invars[0] is not out_ref or store.invars[1] is not dot.outvars[0]:
            raise EmitError("dot_general='tensorops' requires a direct full-block output store")
    else:
        expected = ("get", "get", "squeeze", "squeeze", "dot_general", "broadcast_in_dim", "swap")
        if tuple(e.primitive.name for e in eqns) != expected:
            raise EmitError("dot_general='tensorops' batched form requires a squeezed batch axis")
        lhs_get, rhs_get, lhs_squeeze, rhs_squeeze, dot, expand, store = eqns
        if not _is_empty_get(lhs_get, lhs_ref) or not _is_empty_get(rhs_get, rhs_ref):
            raise EmitError("dot_general='tensorops' requires full-block input reads")
        if (
            tuple(lhs_squeeze.params["dimensions"]) != (0,)
            or tuple(rhs_squeeze.params["dimensions"]) != (0,)
            or dot.invars != [lhs_squeeze.outvars[0], rhs_squeeze.outvars[0]]
        ):
            raise EmitError("dot_general='tensorops' requires one leading singleton batch tile")
        if (
            expand.invars[0] is not dot.outvars[0]
            or tuple(expand.params["broadcast_dimensions"]) != (1, 2)
            or store.invars[0] is not out_ref
            or store.invars[1] is not expand.outvars[0]
        ):
            raise EmitError("dot_general='tensorops' requires a direct batched output store")
    if store.params["tree"].flatten_up_to(()) != []:
        raise EmitError("dot_general='tensorops' requires a full-block output store")
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dot.params["dimension_numbers"]

    a, b = spec.inputs
    c = spec.outputs[0]
    if any(info.dtype.name != "float32" for info in (a, b, c)):
        raise EmitError("dot_general='tensorops' currently supports float32 only")
    rank = len(a.array_shape)
    if rank == 2:
        if len(b.array_shape) != 2 or len(c.array_shape) != 2:
            raise EmitError("dot_general='tensorops' requires rank-2 matrices")
        if any(len(info.block_shape) != 2 for info in (a, b, c)):
            raise EmitError("dot_general='tensorops' requires rank-2 matrix blocks")
        if lhs_batch or rhs_batch or tuple(lhs_contract) != (1,) or tuple(rhs_contract) != (0,):
            raise EmitError("dot_general='tensorops' rank-2 form requires unbatched A @ B")
        m, k = a.array_shape
        kb, n = b.array_shape
        if k != kb or c.array_shape != (m, n):
            raise EmitError("dot_general='tensorops' matrix dimensions do not match")
        tm, ka = a.block_shape
        kb_tile, tn = b.block_shape
        if k != ka or kb != kb_tile or c.block_shape != (tm, tn) or spec.grid != (m // tm, n // tn):
            raise EmitError("dot_general='tensorops' requires full-K, row-major matrix tiles")
        maps_match = (
            _index_map_is(a, (0, None)) and _index_map_is(b, (None, 1)) and _index_map_is(c, (0, 1))
        )
        a_offset = f"_pid.x * {tm * k}"
        b_offset = f"_pid.y * {tn}"
        c_offset = f"_pid.x * {tm * n} + _pid.y * {tn}"
    elif rank == 3:
        if len(b.array_shape) != 3 or len(c.array_shape) != 3:
            raise EmitError("dot_general='tensorops' batched form requires rank-3 arrays")
        if any(len(info.block_shape) != 3 for info in (a, b, c)):
            raise EmitError("dot_general='tensorops' requires rank-3 batch blocks")
        if lhs_batch or rhs_batch or tuple(lhs_contract) != (1,) or tuple(rhs_contract) != (0,):
            raise EmitError("dot_general='tensorops' batched tiles require unbatched local matmuls")
        batch, m, k = a.array_shape
        batch_b, kb, n = b.array_shape
        if batch != batch_b or k != kb or c.array_shape != (batch, m, n):
            raise EmitError("dot_general='tensorops' batched matrix dimensions do not match")
        ba, tm, ka = a.block_shape
        bb, kb_tile, tn = b.block_shape
        bc, cm, cn = c.block_shape
        expected_blocks = ((1, tm, ka), (1, kb_tile, tn), (1, cm, cn))
        if (
            (ba, bb, bc) != (1, 1, 1)
            or any(info.full_block_shape is None for info in (a, b, c))
            or tuple(a.full_block_shape) != expected_blocks[0]
            or tuple(b.full_block_shape) != expected_blocks[1]
            or tuple(c.full_block_shape) != expected_blocks[2]
            or ka != k
            or kb_tile != k
            or (cm, cn) != (tm, tn)
            or spec.grid != (batch, m // tm, n // tn)
        ):
            raise EmitError("dot_general='tensorops' requires full-K batch-local row-major tiles")
        maps_match = (
            _index_map_is(a, (0, 1, None))
            and _index_map_is(b, (0, None, 2))
            and _index_map_is(c, (0, 1, 2))
        )
        a_offset = f"_pid.x * {m * k} + _pid.y * {tm * k}"
        b_offset = f"_pid.x * {k * n} + _pid.z * {tn}"
        c_offset = f"_pid.x * {m * n} + _pid.y * {tm * n} + _pid.z * {tn}"
    else:
        raise EmitError("dot_general='tensorops' supports rank-2 and batched rank-3 arrays")

    if k != kb:
        raise EmitError("dot_general='tensorops' requires matching full-K blocks")
    if rank == 2 and any(info.full_block_shape != info.block_shape for info in (a, b, c)):
        raise EmitError("dot_general='tensorops' requires complete matrix blocks")
    if m % tm or n % tn:
        raise EmitError("dot_general='tensorops' requires evenly tiled M and N dimensions")
    expected_grid = (m // tm, n // tn) if rank == 2 else (batch, m // tm, n // tn)
    if spec.grid != expected_grid:
        raise EmitError("dot_general='tensorops' grid does not match its matrix tiles")
    if not maps_match:
        raise EmitError("dot_general='tensorops' requires standard row-major matrix grid maps")

    name = kernel_name or spec.name
    return f"""#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp;

using matrix_view = tensor<device float, dextents<int, 2>, tensor_inline>;

kernel void {name}(
    device float* arg0 [[buffer(0)]],
    device float* arg1 [[buffer(1)]],
    device float* arg2 [[buffer(2)]],
    uint3 _pid [[threadgroup_position_in_grid]])
{{
    constexpr tensor_ops::matmul2d_descriptor desc({tm}, {tn}, {k}, false, false, false);
    matrix_view a(arg0 + {a_offset}, dextents<int, 2>({k}, {tm}),
                  array<int, 2>({{ 1, {k} }}));
    matrix_view b(arg1 + {b_offset}, dextents<int, 2>({tn}, {k}),
                  array<int, 2>({{ 1, {n} }}));
    matrix_view c(arg2 + {c_offset},
                  dextents<int, 2>({tn}, {tm}), array<int, 2>({{ 1, {n} }}));

    tensor_ops::matmul2d<desc, execution_simdgroups<{SIMDGROUPS}>> op;
    op.run(a, b, c);
}}
"""


def _attention_body(spec: KernelSpec):
    scans = [eqn for eqn in spec.jaxpr.eqns if eqn.primitive.name == "scan"]
    if len(scans) != 1:
        raise EmitError("TensorOps attention requires one scan over key tiles")
    scan = scans[0]
    body = scan.params["jaxpr"]
    if hasattr(body, "jaxpr"):
        body = body.jaxpr
    names = [eqn.primitive.name for eqn in body.eqns]
    dots = [eqn for eqn in body.eqns if eqn.primitive.name == "dot_general"]
    if len(dots) != 2:
        raise EmitError("TensorOps attention requires score and value dot_general operations")
    causal = "le" in names
    noncausal_body = [
        "add",
        "mul",
        "get",
        "get",
        "transpose",
        "dot_general",
        "mul",
        "broadcast_in_dim",
        "reduce_max",
        "max",
        "gt",
        "sub",
        "exp",
        "jit",
        "broadcast_in_dim",
        "sub",
        "exp",
        "jit",
        "mul",
        "reduce_sum",
        "add",
        "broadcast_in_dim",
        "mul",
        "dot_general",
        "add",
    ]
    causal_body = [
        "add",
        "mul",
        "get",
        "get",
        "transpose",
        "dot_general",
        "mul",
        "iota",
        "convert_element_type",
        "add",
        "broadcast_in_dim",
        "broadcast_in_dim",
        "le",
        "jit",
        "reduce_max",
        "max",
        "gt",
        "sub",
        "exp",
        "jit",
        "broadcast_in_dim",
        "sub",
        "exp",
        "jit",
        "mul",
        "reduce_sum",
        "add",
        "broadcast_in_dim",
        "mul",
        "dot_general",
        "add",
    ]
    if names != (causal_body if causal else noncausal_body):
        raise EmitError("TensorOps attention scan does not match the supported online-softmax body")
    expected_outer = (
        [
            "program_id",
            "mul",
            "get",
            "iota",
            "add",
            "broadcast_in_dim",
            "broadcast_in_dim",
            "broadcast_in_dim",
            "scan",
            "broadcast_in_dim",
            "div",
            "swap",
        ]
        if causal
        else [
            "get",
            "broadcast_in_dim",
            "broadcast_in_dim",
            "broadcast_in_dim",
            "scan",
            "broadcast_in_dim",
            "div",
            "swap",
        ]
    )
    if [eqn.primitive.name for eqn in spec.jaxpr.eqns] != expected_outer:
        raise EmitError("TensorOps attention has unsupported operations outside its scan")
    if scan.params.get("reverse") or scan.params.get("unroll") != 1:
        raise EmitError("TensorOps attention requires a forward, single-step static scan")
    if len(spec.inputs) != 3 or len(spec.outputs) != 1 or len(spec.jaxpr.invars) != 4:
        raise EmitError("TensorOps attention requires Q, K, V, and one output buffer")
    if spec.scratch or spec.aliases or len(spec.grid) != 3:
        raise EmitError("TensorOps attention does not support scratch, aliases, or other grids")
    q, k, v = spec.inputs
    out = spec.outputs[0]
    infos = (q, k, v, out)
    if any(i.dtype.name != "float32" or len(i.array_shape) != 4 for i in infos):
        raise EmitError("TensorOps attention requires rank-4 float32 buffers")
    if not (q.array_shape == k.array_shape == v.array_shape == out.array_shape):
        raise EmitError("TensorOps attention requires matching Q, K, V, and output shapes")
    batch, sequence, heads, dim = q.array_shape
    if dim != 64 or sequence % int(scan.params["length"]):
        raise EmitError("TensorOps attention currently requires D=64 and complete key tiles")
    scan_start_index = 5 if causal else 4
    if (
        len(scan.invars) <= scan_start_index
        or not isinstance(scan.invars[3], Literal)
        or float(scan.invars[3].val) != 0.125
        or not isinstance(scan.invars[scan_start_index], Literal)
        or int(scan.invars[scan_start_index].val) != 0
    ):
        raise EmitError("TensorOps attention requires scale=1/sqrt(64) and scan start zero")
    if any(i.full_block_shape is None for i in infos):
        raise EmitError("TensorOps attention requires explicit rank-4 block mappings")
    q_block = tuple(1 if d is None else d for d in q.full_block_shape)
    kv_block = tuple(1 if d is None else d for d in k.full_block_shape)
    if q_block[0] != 1 or q_block[1] < 1 or q_block[2:] != (1, dim):
        raise EmitError("TensorOps attention requires query blocks shaped (1, BQ, 1, 64)")
    tile_q = q_block[1]
    tile_k = sequence // int(scan.params["length"])
    if kv_block != (1, sequence, 1, dim) or v.full_block_shape != k.full_block_shape:
        raise EmitError("TensorOps attention requires full-sequence K/V block mappings")
    if out.full_block_shape != q.full_block_shape or spec.grid != (
        batch,
        heads,
        sequence // tile_q,
    ):
        raise EmitError("TensorOps attention requires grid (batch, heads, query_tiles)")
    if not (
        _index_map_is(q, (0, 2, 1, None))
        and _index_map_is(k, (0, None, 1, None))
        and _index_map_is(v, (0, None, 1, None))
        and _index_map_is(out, (0, 2, 1, None))
    ):
        raise EmitError("TensorOps attention requires the standard [batch, sequence, head, D] maps")
    expected_dots = (
        ((tile_q, dim), (dim, tile_k), (tile_q, tile_k)),
        ((tile_q, tile_k), (tile_k, dim), (tile_q, dim)),
    )
    for dot, shapes in zip(dots, expected_dots, strict=True):
        actual = tuple(tuple(v.aval.shape) for v in dot.invars + dot.outvars)
        if actual != shapes:
            raise EmitError("TensorOps attention dot dimensions do not match its tile layout")
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dot.params["dimension_numbers"]
        if lhs_batch or rhs_batch or tuple(lhs_contract) != (1,) or tuple(rhs_contract) != (0,):
            raise EmitError("TensorOps attention requires row-major unbatched dot products")
    if tile_q % 16 or tile_k % 16 or dim % 16:
        raise EmitError("TensorOps attention tile dimensions must be multiples of 16")
    return scan, (batch, sequence, heads, dim), tile_q, tile_k, causal


def emit_tensorops_attention(spec: KernelSpec, kernel_name: str | None = None) -> tuple[str, int]:
    """Emit the recognized Pallas online-softmax attention pattern."""
    _, shape, tile_q, tile_k, causal = _attention_body(spec)
    _, sequence, heads, dim = shape
    name = kernel_name or spec.name
    causal_code = (
        """
                if (k_start + col > q_start + row)
                    score = -INFINITY;
"""
        if causal
        else ""
    )
    source = f"""#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp;

kernel void {name}(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    device float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]])
{{
    constexpr int BQ = {tile_q};
    constexpr int BK = {tile_k};
    constexpr int D = {dim};
    const uint THREADS = threads_per_group.x * threads_per_group.y * threads_per_group.z;
    const uint batch = group.x;
    const uint head = group.y;
    const uint q_start = group.z * BQ;
    const uint base = (batch * {sequence} * {heads} + head) * D;

    threadgroup float scores[BQ * BK];
    threadgroup float accumulator[BQ * D];
    threadgroup float row_max[BQ];
    threadgroup float row_sum[BQ];
    for (uint i = tid; i < BQ * D; i += THREADS) accumulator[i] = 0.0f;
    for (uint row = tid; row < BQ; row += THREADS) {{
        row_max[row] = -INFINITY;
        row_sum[row] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr tensor_ops::matmul2d_descriptor score_desc(
        BQ, BK, D, false, true, false);
    constexpr tensor_ops::matmul2d_descriptor value_desc(
        BQ, D, BK, false, false, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<score_desc, execution_simdgroups<{SIMDGROUPS}>> score_op;
    tensor_ops::matmul2d<value_desc, execution_simdgroups<{SIMDGROUPS}>> value_op;

    for (uint k_start = 0; k_start < {sequence}; k_start += BK) {{
        auto q_tile = tensor<device float, dextents<int, 2>, tensor_inline>(
            query + base + q_start * {heads * dim}, dextents<int, 2>(D, BQ),
            array<int, 2>{{1, {heads * dim}}});
        auto k_tile = tensor<device float, dextents<int, 2>, tensor_inline>(
            key + base + k_start * {heads * dim}, dextents<int, 2>(D, BK),
            array<int, 2>{{1, {heads * dim}}});
        auto score_tile = tensor<threadgroup float, dextents<int, 2>, tensor_inline>(
            scores, dextents<int, 2>(BK, BQ), array<int, 2>{{1, BK}});
        score_op.run(q_tile, k_tile, score_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint row = tid; row < BQ; row += THREADS) {{
            float block_max = -INFINITY;
            for (uint col = 0; col < BK; ++col) {{
                float score = scores[row * BK + col] * 0.125f;
{causal_code}                scores[row * BK + col] = score;
                block_max = max(block_max, score);
            }}
            const float old_max = row_max[row];
            const float new_max = max(old_max, block_max);
            const float old_scale = isinf(old_max) ? 0.0f : exp(old_max - new_max);
            float block_sum = 0.0f;
            for (uint col = 0; col < BK; ++col) {{
                const float p = exp(scores[row * BK + col] - new_max);
                scores[row * BK + col] = p;
                block_sum += p;
            }}
            row_sum[row] = old_scale * row_sum[row] + block_sum;
            row_max[row] = new_max;
            for (uint d = 0; d < D; ++d) accumulator[row * D + d] *= old_scale;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto probability_tile = tensor<threadgroup float, dextents<int, 2>, tensor_inline>(
            scores, dextents<int, 2>(BK, BQ), array<int, 2>{{1, BK}});
        auto value_tile = tensor<device float, dextents<int, 2>, tensor_inline>(
            value + base + k_start * {heads * dim}, dextents<int, 2>(D, BK),
            array<int, 2>{{1, {heads * dim}}});
        auto output_tile = tensor<threadgroup float, dextents<int, 2>, tensor_inline>(
            accumulator, dextents<int, 2>(D, BQ), array<int, 2>{{1, D}});
        value_op.run(probability_tile, value_tile, output_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (uint i = tid; i < BQ * D; i += THREADS) {{
        const uint row = i / D;
        output[base + q_start * {heads * dim} + row * {heads * dim} + i % D] =
            accumulator[i] / row_sum[row];
    }}
}}
"""
    shared_bytes = (tile_q * tile_k + tile_q * dim + 2 * tile_q) * 4
    return source, shared_bytes


def emit_tensorops(spec: KernelSpec, kernel_name: str | None = None) -> tuple[str, int]:
    """Dispatch TensorOps to the supported standalone or fused pattern."""
    if any(e.primitive.name == "dot_general" for e in spec.jaxpr.eqns):
        return emit_tensorops_matmul(spec, kernel_name), 0
    return emit_tensorops_attention(spec, kernel_name)
