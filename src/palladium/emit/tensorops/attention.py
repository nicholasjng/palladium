"""IR-driven lowering for cooperative Pallas online-softmax attention."""

from __future__ import annotations

import dataclasses
import math

from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var

from palladium.emit.cooperative import (
    CooperativeValue,
    OnlineSoftmaxPlan,
    elementwise_expression,
    emit_online_softmax_rows,
)
from palladium.emit.core import Cursor, CVal, Environment
from palladium.errors import EmitError
from palladium.trace import KernelSpec

from ._shared import (
    _index_map_is,
    _kernel_source,
    _TensorOpsMatmul,
    _TensorView,
)
from .ir import IROperation, KernelIR
from .plan import Distribution, ProgramScope


@dataclasses.dataclass(frozen=True)
class _IRAttentionPlan:
    """IR operations and static geometry needed by the cooperative emitter."""

    scan: JaxprEqn
    query_length: int
    key_length: int
    heads: int
    dim: int
    tile_q: int
    tile_k: int
    causal: bool
    dots: tuple[JaxprEqn, JaxprEqn]
    score_op: _TensorOpsMatmul
    value_op: _TensorOpsMatmul
    online_softmax: OnlineSoftmaxPlan
    final_div: JaxprEqn


def lower_attention_ir(kernel: KernelIR, kernel_name: str | None = None) -> tuple[str, int]:
    """Lower a supported scan by analyzing its imported IR, without the legacy matcher."""
    if kernel.plan.scope is not ProgramScope.THREADGROUP:
        raise EmitError("tensorops attention lowering requires threadgroup program scope")
    spec = kernel.plan.spec
    scans = [op for op in kernel.body.operations if op.name == "scan"]
    if len(scans) != 1:
        raise EmitError("tensorops attention requires one top-level scan region")
    scan_ir = scans[0]
    body = scan_ir.equation.params["jaxpr"]
    if isinstance(body, ClosedJaxpr):
        body = body.jaxpr
    if not isinstance(body, Jaxpr):
        raise EmitError("tensorops attention scan has no imported jaxpr body")
    body_ir = scan_ir.regions[0] if scan_ir.regions else None
    if body_ir is None:
        raise EmitError("tensorops attention scan body is missing from the imported IR")
    body_ops = {id(operation.equation): operation for operation in body_ir.operations}
    plan = _analyze_attention(spec, scan_ir, body, body_ops)
    return _emit(plan, spec, kernel_name)


def _shape(atom) -> tuple[int, ...]:
    shape = getattr(atom.aval, "shape", ())
    return tuple(int(dimension) for dimension in shape)


def _analyze_attention(
    spec: KernelSpec,
    scan_ir: IROperation,
    body: Jaxpr,
    body_ops: dict[int, IROperation],
) -> _IRAttentionPlan:
    scan = scan_ir.equation
    if scan.params.get("reverse") or scan.params.get("unroll") != 1:
        raise EmitError("tensorops attention requires a forward, single-step scan")
    if len(spec.inputs) != 3 or len(spec.outputs) != 1 or spec.scratch or spec.aliases:
        raise EmitError("tensorops attention requires Q, K, V, and one non-aliased output")
    if len(spec.grid) != 3 or len(body_ops) != len(body.eqns):
        raise EmitError("tensorops attention requires a complete three-dimensional Pallas grid")

    query, key, value = spec.inputs
    output = spec.outputs[0]
    if any(
        info.dtype.name != "float32" or len(info.array_shape) != 4
        for info in (*spec.inputs, output)
    ):
        raise EmitError("tensorops attention currently requires rank-4 float32 buffers")
    batch, query_length, heads, dim = query.array_shape
    key_batch, key_length, key_heads, key_dim = key.array_shape
    if (
        output.array_shape != query.array_shape
        or value.array_shape != key.array_shape
        or (key_batch, key_heads, key_dim) != (batch, heads, dim)
    ):
        raise EmitError("tensorops attention Q, K, V, and output shapes are incompatible")
    q_block = query.full_block_shape
    k_block = key.full_block_shape
    if q_block is None or k_block is None or value.full_block_shape != k_block:
        raise EmitError("tensorops attention requires explicit query and key/value block shapes")
    q_block = tuple(1 if size is None else int(size) for size in q_block)
    k_block = tuple(1 if size is None else int(size) for size in k_block)
    if q_block[0] != 1 or q_block[2:] != (1, dim) or k_block != (1, key_length, 1, dim):
        raise EmitError("tensorops attention requires (1, BQ, 1, D) and full-sequence K/V blocks")
    tile_q = q_block[1]
    scan_length = int(scan.params["length"])
    if (
        key_length % scan_length
        or query_length % tile_q
        or output.full_block_shape != query.full_block_shape
        or spec.grid != (batch, heads, query_length // tile_q)
        or not (
            _index_map_is(query, (0, 2, 1, None))
            and _index_map_is(key, (0, None, 1, None))
            and _index_map_is(value, (0, None, 1, None))
            and _index_map_is(output, (0, 2, 1, None))
        )
    ):
        raise EmitError("tensorops attention requires standard full-sequence block mappings")
    tile_k = key_length // scan_length
    causal_ops = [eqn for eqn in body.eqns if eqn.primitive.name == "le"]
    if len(causal_ops) > 1 or (causal_ops and query_length != key_length):
        raise EmitError("tensorops attention supports at most one square causal comparison")
    causal = bool(causal_ops)
    scale_index, start_index = (3, 5) if causal else (3, 4)
    scale = scan.invars[scale_index] if len(scan.invars) > scale_index else None
    start = scan.invars[start_index] if len(scan.invars) > start_index else None
    if (
        not isinstance(scale, Literal)
        or not math.isclose(float(scale.val), dim**-0.5, rel_tol=1e-6)
        or not isinstance(start, Literal)
        or int(start.val) != 0
    ):
        raise EmitError("tensorops attention requires scale=1/sqrt(D) and scan start zero")

    dots = [eqn for eqn in body.eqns if eqn.primitive.name == "dot_general"]
    if len(dots) != 2:
        raise EmitError("tensorops attention scan must contain score and value dot operations")
    expected = (
        ((tile_q, dim), (dim, tile_k), (tile_q, tile_k)),
        ((tile_q, tile_k), (tile_k, dim), (tile_q, dim)),
    )
    for dot, shapes in zip(dots, expected, strict=True):
        if tuple(_shape(atom) for atom in (*dot.invars, *dot.outvars)) != shapes:
            raise EmitError(
                "tensorops attention dot shapes do not match the imported tile geometry"
            )
        operation = body_ops[id(dot)]
        if any(
            result.layout is None or result.layout.distribution is not Distribution.TENSOROPS
            for result in operation.results
        ):
            raise EmitError("tensorops attention dot result does not have TensorOps ownership")

    producers = {
        variable: eqn for eqn in body.eqns for variable in eqn.outvars if isinstance(variable, Var)
    }
    if tile_q % 16 or tile_k % 16 or dim % 16:
        raise EmitError("TensorOps attention tile and head dimensions must be multiples of 16")
    score_dot, value_dot = dots
    score_scale = _one(
        body.eqns,
        "mul",
        (tile_q, tile_k),
        lambda eqn: score_dot.outvars[0] in eqn.invars,
        "score scale",
    )
    score_max = _one(
        body.eqns,
        "reduce_max",
        (tile_q,),
        lambda eqn: (
            _shape(eqn.invars[0]) == (tile_q, tile_k) and tuple(eqn.params.get("axes", ())) == (1,)
        ),
        "row maximum",
    )
    causal_score = score_max.invars[0]
    if causal:
        score_mask = producers.get(causal_score) if isinstance(causal_score, Var) else None
        if (
            score_mask is None
            or score_mask.primitive.name != "jit"
            or causal_ops[0].outvars[0] not in score_mask.invars
            or score_scale.outvars[0] not in score_mask.invars
            or not any(
                isinstance(atom, Literal) and float(atom.val) == float("-inf")
                for atom in score_mask.invars
            )
        ):
            raise EmitError(
                "tensorops causal score maximum must mask future scores to negative infinity"
            )
        _validate_causal_positions(spec, scan, body, producers, causal_ops[0], tile_q, tile_k)
    elif causal_score is not score_scale.outvars[0]:
        raise EmitError("tensorops unmasked row maximum must consume the scaled score tile")
    running_max = _one(
        body.eqns,
        "max",
        (tile_q,),
        lambda eqn: score_max.outvars[0] in eqn.invars,
        "running maximum update",
    )

    def centers_on_running_max(eqn):
        for atom in eqn.invars:
            producer = producers.get(atom) if isinstance(atom, Var) else None
            if (
                producer is not None
                and producer.primitive.name == "broadcast_in_dim"
                and running_max.outvars[0] in producer.invars
            ):
                return True
        return False

    nonempty = _one(body.eqns, "gt", (tile_q,), lambda _: True, "nonempty-row predicate")
    probability_center = _one(
        body.eqns,
        "sub",
        (tile_q, tile_k),
        centers_on_running_max,
        "probability centering",
    )
    score_input = next(
        atom for atom in probability_center.invars if _shape(atom) == (tile_q, tile_k)
    )
    if score_input is not causal_score:
        raise EmitError(
            "tensorops softmax centering must consume the same scores used by row maximum"
        )
    running_max_broadcast = producers.get(probability_center.invars[1])
    if (
        probability_center.invars[0] is not score_input
        or running_max_broadcast is None
        or running_max_broadcast.primitive.name != "broadcast_in_dim"
        or running_max.outvars[0] not in running_max_broadcast.invars
    ):
        raise EmitError(
            "tensorops softmax probabilities must compute scores minus the running maximum"
        )
    probability_exp = _one(
        body.eqns,
        "exp",
        (tile_q, tile_k),
        lambda _: True,
        "probability exponential",
    )
    if tuple(probability_exp.invars) != (probability_center.outvars[0],):
        raise EmitError("tensorops probabilities must exponentiate the centered score tile")
    probability_sum = _one(
        body.eqns,
        "reduce_sum",
        (tile_q,),
        lambda eqn: (
            _shape(eqn.invars[0]) == (tile_q, tile_k) and tuple(eqn.params.get("axes", ())) == (1,)
        ),
        "row probability sum",
    )
    sum_scale = _one(
        body.eqns,
        "mul",
        (tile_q,),
        lambda eqn: any(atom in body.invars and _shape(atom) == (tile_q,) for atom in eqn.invars),
        "running sum rescale",
    )
    sum_update = _one(
        body.eqns,
        "add",
        (tile_q,),
        lambda eqn: probability_sum.outvars[0] in eqn.invars,
        "running sum update",
    )
    accumulator_scale = _one(
        body.eqns,
        "mul",
        (tile_q, dim),
        lambda eqn: any(
            atom in body.invars and _shape(atom) == (tile_q, dim) for atom in eqn.invars
        ),
        "output accumulator rescale",
    )
    final_div = _one(
        spec.jaxpr.eqns,
        "div",
        (tile_q, dim),
        lambda eqn: any(atom in scan.outvars for atom in eqn.invars),
        "final attention normalization",
    )
    if (
        len([eqn for eqn in body.eqns if eqn.primitive.name == "reduce_max"]) != 1
        or len([eqn for eqn in body.eqns if eqn.primitive.name == "reduce_sum"]) != 1
    ):
        raise EmitError("tensorops attention requires one row max and one row sum")

    online = OnlineSoftmaxPlan(
        body_invars=tuple(body.invars),
        score_scale=score_scale,
        score_mask=causal_ops[0] if causal_ops else None,
        score_max=score_max,
        running_max=running_max,
        nonempty=nonempty,
        probability_center=probability_center,
        probability_exp=probability_exp,
        probability_sum=probability_sum,
        sum_scale=sum_scale,
        sum_update=sum_update,
        accumulator_scale=accumulator_scale,
    )
    score_op = _TensorOpsMatmul.from_eqn(score_dot, producers, name="score_op", accumulate=False)
    value_op = _TensorOpsMatmul.from_eqn(value_dot, producers, name="value_op", accumulate=True)
    return _IRAttentionPlan(
        scan,
        query_length,
        key_length,
        heads,
        dim,
        tile_q,
        tile_k,
        causal,
        (score_dot, value_dot),
        score_op,
        value_op,
        online,
        final_div,
    )


def _one(eqns, name, shape, predicate, role):
    matches = [
        eqn
        for eqn in eqns
        if eqn.primitive.name == name and _shape(eqn.outvars[0]) == shape and predicate(eqn)
    ]
    if len(matches) != 1:
        raise EmitError(f"tensorops attention expected one {role}, found {len(matches)}")
    return matches[0]


def _validate_causal_positions(
    spec: KernelSpec,
    scan: JaxprEqn,
    body: Jaxpr,
    body_producers: dict[Var, JaxprEqn],
    mask: JaxprEqn,
    tile_q: int,
    tile_k: int,
) -> None:
    """Check the mask compares absolute key and query positions for this grid tile."""
    if _shape(mask.outvars[0]) != (tile_q, tile_k) or len(body.invars) <= 5:
        raise EmitError("tensorops causal mask must cover the score tile")
    key_position = body_producers.get(mask.invars[0])
    query_position = body_producers.get(mask.invars[1])
    if (
        key_position is None
        or query_position is None
        or key_position.primitive.name != "broadcast_in_dim"
        or query_position.primitive.name != "broadcast_in_dim"
        or tuple(key_position.params.get("shape", ())) != (1, tile_k)
        or tuple(key_position.params.get("broadcast_dimensions", ())) != (1,)
        or tuple(query_position.params.get("shape", ())) != (tile_q, 1)
        or tuple(query_position.params.get("broadcast_dimensions", ())) != (0,)
        or query_position.invars[0] is not body.invars[4]
    ):
        raise EmitError(
            "tensorops causal mask must broadcast absolute positions over rows and columns"
        )

    key_coordinates = body_producers.get(key_position.invars[0])
    if key_coordinates is None or key_coordinates.primitive.name != "add":
        raise EmitError("tensorops causal key positions must add the scan offset to a tile iota")
    key_terms = [body_producers.get(atom) for atom in key_coordinates.invars]
    key_iota = next(
        (eqn for eqn in key_terms if eqn is not None and eqn.primitive.name == "iota"), None
    )
    converted_offset = next(
        (
            eqn
            for eqn in key_terms
            if eqn is not None and eqn.primitive.name == "convert_element_type"
        ),
        None,
    )
    if key_iota is None or tuple(key_iota.params.get("shape", ())) != (tile_k,):
        raise EmitError("tensorops causal key coordinates need an iota matching the key tile")
    key_offset = (
        body_producers.get(converted_offset.invars[0]) if converted_offset is not None else None
    )
    if (
        converted_offset is None
        or key_offset is None
        or key_offset.primitive.name != "mul"
        or body.invars[5] not in key_offset.invars
        or not any(
            isinstance(atom, Literal) and int(atom.val) == tile_k for atom in key_offset.invars
        )
    ):
        raise EmitError("tensorops causal key offset must follow the scan's key-tile index")

    outer_producers = {
        variable: eqn
        for eqn in spec.jaxpr.eqns
        for variable in eqn.outvars
        if isinstance(variable, Var)
    }
    query_coordinates = outer_producers.get(scan.invars[4])
    if query_coordinates is None or query_coordinates.primitive.name != "add":
        raise EmitError("tensorops causal query positions must be derived from the Pallas grid")
    query_terms = [outer_producers.get(atom) for atom in query_coordinates.invars]
    query_iota = next(
        (eqn for eqn in query_terms if eqn is not None and eqn.primitive.name == "iota"), None
    )
    query_offset = next(
        (eqn for eqn in query_terms if eqn is not None and eqn.primitive.name == "mul"), None
    )
    program_id = (
        next(
            (
                outer_producers.get(atom)
                for atom in query_offset.invars
                if isinstance(atom, Var)
                and outer_producers.get(atom) is not None
                and outer_producers[atom].primitive.name == "program_id"
            ),
            None,
        )
        if query_offset is not None
        else None
    )
    if (
        query_iota is None
        or tuple(query_iota.params.get("shape", ())) != (tile_q,)
        or query_offset is None
        or not any(
            isinstance(atom, Literal) and int(atom.val) == tile_q for atom in query_offset.invars
        )
        or program_id is None
        or program_id.params.get("axis") != 2
    ):
        raise EmitError("tensorops causal query positions must use the query-tile grid axis")


def _emit(plan: _IRAttentionPlan, spec: KernelSpec, kernel_name: str | None):
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
