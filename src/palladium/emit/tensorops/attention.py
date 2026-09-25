"""Recognition and emission for fused online-softmax attention."""

from __future__ import annotations

import dataclasses
import math

from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var

from palladium.emit.cooperative import (
    CooperativeValue,
    OnlineSoftmaxPlan,
    elementwise_expression,
    emit_online_softmax_rows,
    row_reduction_expression,
)
from palladium.emit.core import Cursor, CVal, Environment
from palladium.errors import EmitError
from palladium.trace import KernelSpec

from ._shared import _index_map_is, _shape, _TensorOpsMatmul, _TensorView


@dataclasses.dataclass(frozen=True)
class _AttentionPlan:
    """Validated scan equations and tensor shapes for fused attention."""

    spec: KernelSpec
    scan: JaxprEqn
    batch: int
    query_length: int
    key_length: int
    heads: int
    dim: int
    tile_q: int
    tile_k: int
    causal: bool
    body: Jaxpr
    dots: tuple[JaxprEqn, JaxprEqn]
    score_op: _TensorOpsMatmul
    value_op: _TensorOpsMatmul
    online_softmax: OnlineSoftmaxPlan
    final_div: JaxprEqn


def _attention_body(spec: KernelSpec):
    scans = [eqn for eqn in spec.jaxpr.eqns if eqn.primitive.name == "scan"]
    if len(scans) != 1:
        raise EmitError("TensorOps attention requires one scan over key tiles")
    scan = scans[0]
    body = scan.params["jaxpr"]
    if hasattr(body, "jaxpr"):
        body = body.jaxpr
    dots = [eqn for eqn in body.eqns if eqn.primitive.name == "dot_general"]
    if len(dots) != 2:
        raise EmitError("TensorOps attention requires score and value dot_general operations")
    causal = any(eqn.primitive.name == "le" for eqn in body.eqns)
    if sum(eqn.primitive.name == "le" for eqn in body.eqns) > 1:
        raise EmitError("TensorOps attention supports one causal comparison")
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
    batch, query_length, heads, dim = q.array_shape
    key_batch, key_length, key_heads, key_dim = k.array_shape
    if (
        out.array_shape != q.array_shape
        or v.array_shape != k.array_shape
        or (key_batch, key_heads, key_dim) != (batch, heads, dim)
    ):
        raise EmitError("TensorOps attention requires compatible Q, K, V, and output shapes")
    if key_length % int(scan.params["length"]):
        raise EmitError("TensorOps attention requires complete key tiles")
    scan_start_index = 5 if causal else 4
    scale_atom = scan.invars[3] if len(scan.invars) > 3 else None
    start_atom = scan.invars[scan_start_index] if len(scan.invars) > scan_start_index else None
    if (
        not isinstance(scale_atom, Literal)
        or not math.isclose(float(scale_atom.val), dim**-0.5, rel_tol=1e-6)
        or not isinstance(start_atom, Literal)
        or int(start_atom.val) != 0
    ):
        raise EmitError("TensorOps attention requires scale=1/sqrt(D) and scan start zero")
    q_full = q.full_block_shape
    k_full = k.full_block_shape
    v_full = v.full_block_shape
    out_full = out.full_block_shape
    if q_full is None or k_full is None or v_full is None or out_full is None:
        raise EmitError("TensorOps attention requires explicit rank-4 block mappings")
    q_block = tuple(1 if d is None else d for d in q_full)
    kv_block = tuple(1 if d is None else d for d in k_full)
    if q_block[0] != 1 or q_block[1] < 1 or q_block[2:] != (1, dim):
        raise EmitError("TensorOps attention requires query blocks shaped (1, BQ, 1, D)")
    tile_q = q_block[1]
    tile_k = key_length // int(scan.params["length"])
    if query_length % tile_q:
        raise EmitError("TensorOps attention requires complete query tiles")
    if causal and query_length != key_length:
        raise EmitError("TensorOps causal attention requires matching query and key lengths")
    if kv_block != (1, key_length, 1, dim) or v_full != k_full:
        raise EmitError("TensorOps attention requires full-sequence K/V block mappings")
    if out_full != q_full or spec.grid != (
        batch,
        heads,
        query_length // tile_q,
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
        actual = tuple(_shape(value) for value in dot.invars + dot.outvars)
        if actual != shapes:
            raise EmitError("TensorOps attention dot dimensions do not match its tile layout")
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dot.params["dimension_numbers"]
        if lhs_batch or rhs_batch or tuple(lhs_contract) != (1,) or tuple(rhs_contract) != (0,):
            raise EmitError("TensorOps attention requires row-major unbatched dot products")
    if tile_q % 16 or tile_k % 16 or dim % 16:
        raise EmitError("TensorOps attention tile and head dimensions must be multiples of 16")
    return scan, (batch, query_length, key_length, heads, dim), tile_q, tile_k, causal, body, dots


def _score_scale(body: Jaxpr, dot: JaxprEqn, scale: Literal, dim: int) -> JaxprEqn:
    """Find the scalar multiply directly consuming the score dot result."""
    matches = [
        eqn for eqn in body.eqns if eqn.primitive.name == "mul" and dot.outvars[0] in eqn.invars
    ]
    if len(matches) != 1:
        raise EmitError("TensorOps attention requires one score scaling equation")
    eqn = matches[0]
    if sum(atom is dot.outvars[0] for atom in eqn.invars) != 1:
        raise EmitError("TensorOps attention score scale must use the dot result once")
    other = next(atom for atom in eqn.invars if atom is not dot.outvars[0])
    if (
        len(body.invars) <= 3
        or other is not body.invars[3]
        or not math.isclose(float(scale.val), dim**-0.5, rel_tol=1e-6)
    ):
        raise EmitError("TensorOps attention requires scale=1/sqrt(D)")
    return eqn


def _causal_mask(body: Jaxpr, tile_q: int, tile_k: int) -> JaxprEqn:
    """Recognize the broadcast `key_position <= query_position` predicate."""
    matches = [eqn for eqn in body.eqns if eqn.primitive.name == "le"]
    if len(matches) != 1:
        raise EmitError("TensorOps causal attention requires one causal comparison")
    mask = matches[0]
    producers = {outvar: eqn for eqn in body.eqns for outvar in eqn.outvars}
    lhs = producers.get(mask.invars[0])
    rhs = producers.get(mask.invars[1])
    if (
        lhs is None
        or rhs is None
        or lhs.primitive.name != "broadcast_in_dim"
        or rhs.primitive.name != "broadcast_in_dim"
        or _shape(mask.outvars[0]) != (tile_q, tile_k)
        or tuple(lhs.params["shape"]) != (1, tile_k)
        or tuple(lhs.params["broadcast_dimensions"]) != (1,)
        or tuple(rhs.params["shape"]) != (tile_q, 1)
        or tuple(rhs.params["broadcast_dimensions"]) != (0,)
        or len(body.invars) <= 5
        or rhs.invars[0] is not body.invars[4]
        or _shape(body.invars[4]) != (tile_q,)
        or _shape(body.invars[5]) != ()
    ):
        raise EmitError("TensorOps causal comparison does not match the supported broadcast layout")

    key_position = producers.get(lhs.invars[0])
    if key_position is None or key_position.primitive.name != "add":
        raise EmitError("TensorOps causal key positions require an offset plus iota")
    offset = next(
        (
            producer
            for atom in key_position.invars
            if (producer := producers.get(atom)) is not None
            and producer.primitive.name == "convert_element_type"
        ),
        None,
    )
    iota = next(
        (
            producers.get(atom)
            for atom in key_position.invars
            if (producer := producers.get(atom)) is not None and producer.primitive.name == "iota"
        ),
        None,
    )
    if (
        offset is None
        or offset.primitive.name != "convert_element_type"
        or iota is None
        or tuple(iota.params["shape"]) != (tile_k,)
    ):
        raise EmitError("TensorOps causal key positions require a scan offset and tile iota")
    scan_offset = producers.get(offset.invars[0])
    if (
        scan_offset is None
        or scan_offset.primitive.name != "mul"
        or body.invars[5] not in scan_offset.invars
        or not any(
            isinstance(atom, Literal) and int(atom.val) == tile_k for atom in scan_offset.invars
        )
    ):
        raise EmitError("TensorOps causal key offset does not match the key tile width")
    return mask


def _attention_reductions(body: Jaxpr, tile_q: int, tile_k: int):
    """Find the row-wise score maximum and probability sum equations."""
    reductions = {
        name: [eqn for eqn in body.eqns if eqn.primitive.name == name]
        for name in ("reduce_max", "reduce_sum")
    }
    result = {}
    for name, matches in reductions.items():
        if len(matches) != 1:
            raise EmitError(f"TensorOps attention requires one {name} equation")
        eqn = matches[0]
        row_reduction_expression(eqn, (tile_q, tile_k), "float", "x", "y")
        result[name] = eqn
    return result["reduce_max"], result["reduce_sum"]


def _attention_state_ops(body: Jaxpr, reduce_max: JaxprEqn, tile_q: int):
    """Find the running-maximum update and the nonempty-row predicate."""
    max_eqns = [
        eqn
        for eqn in body.eqns
        if eqn.primitive.name == "max" and reduce_max.outvars[0] in eqn.invars
    ]
    gt_eqns = [
        eqn
        for eqn in body.eqns
        if eqn.primitive.name == "gt"
        and any(isinstance(atom, Literal) and float(atom.val) == 0.0 for atom in eqn.invars)
        and any(
            isinstance(atom, Var) and atom in body.invars and _shape(atom) == (tile_q,)
            for atom in eqn.invars
        )
    ]
    if len(max_eqns) != 1 or len(gt_eqns) != 1:
        raise EmitError("TensorOps attention requires one max update and one nonempty-row test")
    max_eqn, gt_eqn = max_eqns[0], gt_eqns[0]
    if _shape(max_eqn.outvars[0]) != (tile_q,):
        raise EmitError("TensorOps attention running maximum must be row-shaped")
    return max_eqn, gt_eqn


def _attention_updates(
    body: Jaxpr,
    reduce_sum: JaxprEqn,
    value_dot: JaxprEqn,
    tile_q: int,
    dim: int,
):
    """Match the running-sum and accumulator update equations."""
    producers = {outvar: eqn for eqn in body.eqns for outvar in eqn.outvars}
    sum_adds = [
        eqn
        for eqn in body.eqns
        if eqn.primitive.name == "add" and reduce_sum.outvars[0] in eqn.invars
    ]
    if len(sum_adds) != 1:
        raise EmitError("TensorOps attention requires one running-sum update")
    sum_add = sum_adds[0]
    sum_term = next(atom for atom in sum_add.invars if atom is not reduce_sum.outvars[0])
    sum_mul = producers.get(sum_term)
    if sum_mul is None or sum_mul.primitive.name != "mul":
        raise EmitError("TensorOps attention running sum requires a scaled carry")

    accumulator_muls = [
        eqn
        for eqn in body.eqns
        if eqn.primitive.name == "mul"
        and _shape(eqn.outvars[0]) == (tile_q, dim)
        and any(atom in body.invars and _shape(atom) == (tile_q, dim) for atom in eqn.invars)
    ]
    if len(accumulator_muls) != 1:
        raise EmitError("TensorOps attention requires one accumulator rescale equation")
    accumulator_mul = accumulator_muls[0]
    accumulator_adds = [
        eqn
        for eqn in body.eqns
        if eqn.primitive.name == "add"
        and accumulator_mul.outvars[0] in eqn.invars
        and value_dot.outvars[0] in eqn.invars
    ]
    if len(accumulator_adds) != 1:
        raise EmitError("TensorOps attention accumulator must add the value dot")
    return sum_add, sum_mul, accumulator_mul, accumulator_adds[0]


def _attention_probability_ops(body: Jaxpr, tile_q: int, tile_k: int):
    """Find the matrix subtraction and exponentiation for block probabilities."""
    exps = [
        eqn
        for eqn in body.eqns
        if eqn.primitive.name == "exp" and _shape(eqn.outvars[0]) == (tile_q, tile_k)
    ]
    if len(exps) != 1:
        raise EmitError("TensorOps attention requires one matrix probability exponential")
    exp_eqn = exps[0]
    producers = {outvar: eqn for eqn in body.eqns for outvar in eqn.outvars}
    sub_eqn = producers.get(exp_eqn.invars[0])
    if (
        sub_eqn is None
        or sub_eqn.primitive.name != "sub"
        or _shape(sub_eqn.outvars[0]) != (tile_q, tile_k)
    ):
        raise EmitError("TensorOps probability exponential must subtract the row maximum")
    return sub_eqn, exp_eqn


def _validate_attention_body(
    body: Jaxpr,
    dots: tuple[JaxprEqn, JaxprEqn],
    scale_eqn: JaxprEqn,
    mask_eqn: JaxprEqn | None,
    reduce_max: JaxprEqn,
    reduce_sum: JaxprEqn,
    running_max: JaxprEqn,
    nonempty: JaxprEqn,
    sum_add: JaxprEqn,
    sum_mul: JaxprEqn,
    accumulator_mul: JaxprEqn,
    accumulator_add: JaxprEqn,
    probability_sub: JaxprEqn,
    probability_exp: JaxprEqn,
    tile_q: int,
    tile_k: int,
    dim: int,
) -> None:
    """Validate the online-softmax dataflow and account for every body equation."""
    producers = {outvar: eqn for eqn in body.eqns for outvar in eqn.outvars}
    used: set[int] = set()

    def claim(eqn: JaxprEqn | None, role: str) -> JaxprEqn:
        if eqn is None:
            raise EmitError(f"TensorOps attention is missing its {role} equation")
        if id(eqn) in used:
            raise EmitError(f"TensorOps attention reuses an equation for {role}")
        used.add(id(eqn))
        return eqn

    score_dot, value_dot = (claim(eqn, "dot") for eqn in dots)
    scale_eqn = claim(scale_eqn, "score scale")
    reduce_max = claim(reduce_max, "score reduction")
    reduce_sum = claim(reduce_sum, "probability reduction")
    running_max = claim(running_max, "running maximum")
    nonempty = claim(nonempty, "nonempty-row predicate")
    sum_add = claim(sum_add, "running sum update")
    sum_mul = claim(sum_mul, "running sum scale")
    accumulator_mul = claim(accumulator_mul, "output rescale")
    accumulator_add = claim(accumulator_add, "output update")
    probability_sub = claim(probability_sub, "probability centering")
    probability_exp = claim(probability_exp, "probability exponential")

    def producer(atom) -> JaxprEqn | None:
        return producers.get(atom) if isinstance(atom, Var) else None

    if score_dot.invars[0] is not body.invars[2]:
        raise EmitError("TensorOps score dot must consume the current query tile")
    transpose = claim(producer(score_dot.invars[1]), "key transpose")
    if transpose.primitive.name != "transpose" or tuple(transpose.params["permutation"]) != (1, 0):
        raise EmitError("TensorOps score dot must consume transposed keys")
    key_read = claim(producer(transpose.invars[0]), "key tile read")
    value_read = claim(producer(value_dot.invars[1]), "value tile read")
    if (
        key_read.primitive.name != "get"
        or value_read.primitive.name != "get"
        or key_read.invars[0] is not body.invars[0]
        or value_read.invars[0] is not body.invars[1]
        or len(key_read.invars) != 4
        or len(value_read.invars) != 4
        or any(
            not isinstance(index, Literal) or int(index.val) != 0
            for index in (
                key_read.invars[1],
                key_read.invars[3],
                value_read.invars[1],
                value_read.invars[3],
            )
        )
        or key_read.invars[2] is not value_read.invars[2]
    ):
        raise EmitError("TensorOps attention dots must read matching K and V tiles")

    loop_index = body.invars[5] if mask_eqn is not None else body.invars[4]
    key_offset = claim(producer(key_read.invars[2]), "key-tile offset")
    if (
        key_offset.primitive.name != "mul"
        or loop_index not in key_offset.invars
        or not any(
            isinstance(atom, Literal) and int(atom.val) == tile_k for atom in key_offset.invars
        )
    ):
        raise EmitError("TensorOps attention key reads must use the scan's tile offset")
    loop_updates = [
        eqn
        for eqn in body.eqns
        if eqn.primitive.name == "add"
        and loop_index in eqn.invars
        and any(isinstance(atom, Literal) and int(atom.val) == 1 for atom in eqn.invars)
    ]
    if len(loop_updates) != 1:
        raise EmitError("TensorOps attention scan must increment its key-tile index")
    loop_update = claim(loop_updates[0], "scan index update")
    if body.outvars[0] is not loop_update.outvars[0]:
        raise EmitError("TensorOps attention must carry the updated key-tile index")

    score_values = scale_eqn.outvars[0]
    score_eqn = producer(reduce_max.invars[0]) if reduce_max.invars else None
    if score_eqn is None:
        raise EmitError("TensorOps row maximum must reduce score values")
    if mask_eqn is None:
        if score_eqn is not scale_eqn:
            raise EmitError("TensorOps score reduction must consume the scaled scores")
    else:
        score_mask = claim(score_eqn, "causal score mask")
        if (
            score_mask.primitive.name != "jit"
            or mask_eqn.outvars[0] not in score_mask.invars
            or score_values not in score_mask.invars
            or not any(
                isinstance(atom, Literal) and float(atom.val) == float("-inf")
                for atom in score_mask.invars
            )
        ):
            raise EmitError("TensorOps causal mask must exclude future scores")

    active_eqn = mask_eqn
    if mask_eqn is None:
        probability_where = producer(value_dot.invars[0])
        active_eqn = producer(probability_where.invars[0]) if probability_where else None
        active_eqn = claim(active_eqn, "unmasked probability predicate")
        if (
            active_eqn.primitive.name != "broadcast_in_dim"
            or not isinstance(active_eqn.invars[0], Literal)
            or bool(active_eqn.invars[0].val) is not True
        ):
            raise EmitError("Unmasked TensorOps attention requires an all-true predicate")

    if mask_eqn is not None:
        claim(mask_eqn, "causal comparison")
        key_position = producer(mask_eqn.invars[0])
        query_position = producer(mask_eqn.invars[1])
        if (
            key_position is None
            or query_position is None
            or key_position.primitive.name != "broadcast_in_dim"
            or query_position.primitive.name != "broadcast_in_dim"
            or tuple(key_position.params["broadcast_dimensions"]) != (1,)
            or tuple(query_position.params["broadcast_dimensions"]) != (0,)
        ):
            raise EmitError("TensorOps causal positions must broadcast over rows and columns")
        key_coordinates = claim(producer(key_position.invars[0]), "key coordinates")
        if (
            key_coordinates.primitive.name != "add"
            or query_position.invars[0] is not body.invars[4]
            or len(key_coordinates.invars) != 2
        ):
            raise EmitError("TensorOps causal mask must compare absolute query and key positions")
        key_producers = [producer(atom) for atom in key_coordinates.invars]
        converted_offset = next(
            (
                eqn
                for eqn in key_producers
                if eqn is not None and eqn.primitive.name == "convert_element_type"
            ),
            None,
        )
        key_iota = next(
            (eqn for eqn in key_producers if eqn is not None and eqn.primitive.name == "iota"),
            None,
        )
        if converted_offset is None or key_iota is None:
            raise EmitError("TensorOps causal key positions require the scan offset and tile iota")
        claim(key_iota, "key-position iota")
        claim(converted_offset, "key-offset conversion")
        if converted_offset.invars[0] is not key_offset.outvars[0]:
            raise EmitError("TensorOps causal key offset must use the scan key offset")
        claim(key_position, "key-position broadcast")
        claim(query_position, "query-position broadcast")

    if _shape(probability_sub.outvars[0]) != (tile_q, tile_k):
        raise EmitError("TensorOps probability centering has the wrong tile shape")
    max_broadcast = claim(producer(probability_sub.invars[1]), "running-maximum broadcast")
    if (
        max_broadcast.primitive.name != "broadcast_in_dim"
        or max_broadcast.invars[0] is not running_max.outvars[0]
        or tuple(max_broadcast.params["broadcast_dimensions"]) != (0,)
        or probability_sub.invars[0] is not reduce_max.invars[0]
    ):
        raise EmitError("TensorOps probabilities must subtract the current row maximum")
    if probability_exp.invars[0] is not probability_sub.outvars[0]:
        raise EmitError("TensorOps probability exponential must consume centered scores")

    probability_where = claim(producer(value_dot.invars[0]), "probability mask")
    if active_eqn is None:
        raise EmitError("TensorOps attention is missing its active-score predicate")
    if (
        probability_where.primitive.name != "jit"
        or probability_exp.outvars[0] not in probability_where.invars
        or active_eqn.outvars[0] not in probability_where.invars
        or not any(
            isinstance(atom, Literal) and float(atom.val) == 0.0
            for atom in probability_where.invars
        )
        or reduce_sum.invars[0] is not probability_where.outvars[0]
    ):
        raise EmitError("TensorOps probability mask must zero inactive scores")

    old_scale_exp_candidates = [
        eqn
        for eqn in body.eqns
        if eqn.primitive.name == "exp" and _shape(eqn.outvars[0]) == (tile_q,)
    ]
    if len(old_scale_exp_candidates) != 1:
        raise EmitError("TensorOps attention requires one row-wise carry rescale exponential")
    old_scale_exp = old_scale_exp_candidates[0]
    old_scale_sub = producer(old_scale_exp.invars[0])
    old_max_carry = body.invars[6] if mask_eqn is not None else body.invars[5]
    sum_carry = body.invars[7] if mask_eqn is not None else body.invars[6]
    output_carry = body.invars[8] if mask_eqn is not None else body.invars[7]
    if (
        old_scale_sub is None
        or old_scale_sub.primitive.name != "sub"
        or old_scale_sub.invars[0] is not old_max_carry
        or old_scale_sub.invars[1] is not running_max.outvars[0]
        or old_max_carry not in running_max.invars
        or reduce_max.outvars[0] not in running_max.invars
        or len(running_max.invars) != 2
        or not any(isinstance(atom, Var) and atom is sum_carry for atom in nonempty.invars)
        or not any(isinstance(atom, Literal) and float(atom.val) == 0.0 for atom in nonempty.invars)
    ):
        raise EmitError("TensorOps attention carry rescale must use the old and new row maxima")
    old_scale_where = producer(sum_mul.invars[0])
    if old_scale_where is sum_carry:
        old_scale_where = producer(sum_mul.invars[1])
    if old_scale_where is None:
        raise EmitError("TensorOps attention is missing its carry rescale selection")
    if (
        old_scale_where.primitive.name != "jit"
        or nonempty.outvars[0] not in old_scale_where.invars
        or old_scale_exp.outvars[0] not in old_scale_where.invars
        or not any(
            isinstance(atom, Literal) and float(atom.val) == 0.0 for atom in old_scale_where.invars
        )
        or sum_carry not in sum_mul.invars
        or old_scale_where.outvars[0] not in sum_mul.invars
        or len(sum_mul.invars) != 2
        or reduce_sum.outvars[0] not in sum_add.invars
        or sum_mul.outvars[0] not in sum_add.invars
        or len(sum_add.invars) != 2
    ):
        raise EmitError(
            "TensorOps attention running sum must use the rescaled carry and probabilities"
        )
    claim(old_scale_sub, "carry rescale subtraction")
    claim(old_scale_exp, "carry rescale exponential")
    claim(old_scale_where, "carry rescale selection")

    output_scale_broadcast = claim(
        producer(next(atom for atom in accumulator_mul.invars if atom is not output_carry)),
        "output carry scale broadcast",
    )
    if (
        output_scale_broadcast.primitive.name != "broadcast_in_dim"
        or output_scale_broadcast.invars[0] is not old_scale_where.outvars[0]
        or tuple(output_scale_broadcast.params["broadcast_dimensions"]) != (0,)
        or output_carry not in accumulator_mul.invars
        or output_scale_broadcast.outvars[0] not in accumulator_mul.invars
        or len(accumulator_mul.invars) != 2
        or accumulator_mul.outvars[0] not in accumulator_add.invars
        or value_dot.outvars[0] not in accumulator_add.invars
        or len(accumulator_add.invars) != 2
    ):
        raise EmitError("TensorOps attention output update must rescale and add the value dot")
    if (
        len(body.outvars) != 4
        or body.outvars[1] is not running_max.outvars[0]
        or body.outvars[2] is not sum_add.outvars[0]
        or body.outvars[3] is not accumulator_add.outvars[0]
    ):
        raise EmitError("TensorOps attention must carry the updated max, sum, and output")
    if len(used) != len(body.eqns):
        unsupported = sorted(eqn.primitive.name for eqn in body.eqns if id(eqn) not in used)
        raise EmitError(f"TensorOps attention has unsupported scan equations: {unsupported}")


def _attention_final_div(spec: KernelSpec, tile_q: int, dim: int) -> JaxprEqn:
    """Find the final row-normalization equation outside the scan."""
    matches = [eqn for eqn in spec.jaxpr.eqns if eqn.primitive.name == "div"]
    if len(matches) != 1:
        raise EmitError("TensorOps attention requires one final normalization equation")
    eqn = matches[0]
    if (
        _shape(eqn.invars[0]) != (tile_q, dim)
        or _shape(eqn.invars[1]) != (tile_q, 1)
        or _shape(eqn.outvars[0]) != (tile_q, dim)
    ):
        raise EmitError("TensorOps attention normalization must broadcast one sum per row")
    return eqn


def _validate_attention_outer(
    spec: KernelSpec, scan: JaxprEqn, causal: bool, tile_q: int, dim: int, final_div: JaxprEqn
) -> None:
    """Bind scan carry values to normalization, query positions, and output store."""
    eqns = spec.jaxpr.eqns
    producers = {outvar: eqn for eqn in eqns for outvar in eqn.outvars}
    used = {id(scan), id(final_div)}

    stores = [eqn for eqn in eqns if eqn.primitive.name == "swap"]
    if len(stores) != 1:
        raise EmitError("TensorOps attention requires one output store")
    store = stores[0]
    out_ref = spec.jaxpr.invars[-1]
    if (
        len(spec.jaxpr.invars) != 4
        or len(store.invars) != 4
        or store.invars[0] is not out_ref
        or store.invars[1] is not final_div.outvars[0]
        or any(not isinstance(index, Literal) or int(index.val) != 0 for index in store.invars[2:])
    ):
        raise EmitError("TensorOps attention requires a direct full-block output store")
    used.add(id(store))

    if len(scan.outvars) != 4 or len(scan.invars) < 8:
        raise EmitError("TensorOps attention scan must carry max, sum, and output state")
    if final_div.invars[0] is not scan.outvars[3]:
        raise EmitError("TensorOps attention must normalize the scan's output accumulator")
    sum_broadcast = producers.get(final_div.invars[1])
    if (
        sum_broadcast is None
        or sum_broadcast.primitive.name != "broadcast_in_dim"
        or len(sum_broadcast.invars) != 1
        or sum_broadcast.invars[0] is not scan.outvars[2]
        or tuple(sum_broadcast.params["broadcast_dimensions"]) != (0,)
    ):
        raise EmitError("TensorOps attention must normalize by the scan's running sum")
    used.add(id(sum_broadcast))

    max_init, sum_init, output_init = scan.invars[-3:]
    init_roles = (
        (max_init, float("-inf"), (tile_q,)),
        (sum_init, 0.0, (tile_q,)),
        (output_init, 0.0, (tile_q, dim)),
    )
    for atom, expected, shape in init_roles:
        init = producers.get(atom)
        if (
            init is None
            or init.primitive.name != "broadcast_in_dim"
            or len(init.invars) != 1
            or not isinstance(init.invars[0], Literal)
            or float(init.invars[0].val) != expected
            or _shape(atom) != shape
        ):
            raise EmitError("TensorOps attention scan carry has unsupported initial state")
        used.add(id(init))
    start = scan.invars[-4]
    if not isinstance(start, Literal) or int(start.val) != 0:
        raise EmitError("TensorOps attention scan must start at key offset zero")

    query_ref, key_ref, value_ref = spec.jaxpr.invars[:3]
    if scan.invars[0] is not key_ref or scan.invars[1] is not value_ref:
        raise EmitError("TensorOps attention scan must read the key and value inputs")
    query_tile = producers.get(scan.invars[2])
    if (
        query_tile is None
        or query_tile.primitive.name != "get"
        or query_tile.invars[0] is not query_ref
        or any(
            not isinstance(index, Literal) or int(index.val) != 0 for index in query_tile.invars[1:]
        )
        or _shape(query_tile.outvars[0]) != (tile_q, dim)
    ):
        raise EmitError("TensorOps attention scan must use the current query tile")
    used.add(id(query_tile))

    if causal:
        query_positions = producers.get(scan.invars[4])
        if query_positions is None or query_positions.primitive.name != "add":
            raise EmitError("Causal TensorOps attention requires query positions from the grid")
        position_operands = [producers.get(atom) for atom in query_positions.invars]
        offset_mul = next(
            (eqn for eqn in position_operands if eqn is not None and eqn.primitive.name == "mul"),
            None,
        )
        tile_iota = next(
            (eqn for eqn in position_operands if eqn is not None and eqn.primitive.name == "iota"),
            None,
        )
        if offset_mul is None or tile_iota is None:
            raise EmitError("Causal TensorOps attention query positions do not match its grid")
        program_ids = [producers.get(atom) for atom in offset_mul.invars if isinstance(atom, Var)]
        program_id = next(
            (eqn for eqn in program_ids if eqn is not None and eqn.primitive.name == "program_id"),
            None,
        )
        if (
            len(offset_mul.invars) != 2
            or len(query_positions.invars) != 2
            or tuple(tile_iota.params["shape"]) != (tile_q,)
            or not any(
                isinstance(atom, Literal) and int(atom.val) == tile_q for atom in offset_mul.invars
            )
            or program_id is None
        ):
            raise EmitError("Causal TensorOps attention query positions do not match its grid")
        if program_id.params.get("axis") != 2:
            raise EmitError("Causal TensorOps attention must use the query-tile grid axis")
        used.update(
            (
                id(query_positions),
                id(offset_mul),
                id(tile_iota),
                id(program_id),
            )
        )

    if len(used) != len(eqns):
        unsupported = sorted(eqn.primitive.name for eqn in eqns if id(eqn) not in used)
        raise EmitError(f"TensorOps attention has unsupported outer equations: {unsupported}")


def _recognize_tensorops_attention(spec: KernelSpec) -> _AttentionPlan:
    """Match a supported streaming-softmax scan and return its checked plan."""
    scan, shape, tile_q, tile_k, causal, body, dots = _attention_body(spec)
    batch, query_length, key_length, heads, dim = shape
    producers = {outvar: eqn for eqn in body.eqns for outvar in eqn.outvars}
    score_op = _TensorOpsMatmul.from_eqn(dots[0], producers, name="score_op", accumulate=False)
    value_op = _TensorOpsMatmul.from_eqn(dots[1], producers, name="value_op", accumulate=True)
    scale_eqn = _score_scale(body, dots[0], scan.invars[3], dim)
    mask_eqn = _causal_mask(body, tile_q, tile_k) if causal else None
    reduce_max_eqn, reduce_sum_eqn = _attention_reductions(body, tile_q, tile_k)
    max_eqn, nonempty_eqn = _attention_state_ops(body, reduce_max_eqn, tile_q)
    sum_add_eqn, sum_mul_eqn, accumulator_mul_eqn, accumulator_add_eqn = _attention_updates(
        body, reduce_sum_eqn, dots[1], tile_q, dim
    )
    probability_sub_eqn, probability_exp_eqn = _attention_probability_ops(body, tile_q, tile_k)
    _validate_attention_body(
        body,
        (dots[0], dots[1]),
        scale_eqn,
        mask_eqn,
        reduce_max_eqn,
        reduce_sum_eqn,
        max_eqn,
        nonempty_eqn,
        sum_add_eqn,
        sum_mul_eqn,
        accumulator_mul_eqn,
        accumulator_add_eqn,
        probability_sub_eqn,
        probability_exp_eqn,
        tile_q,
        tile_k,
        dim,
    )
    online_softmax = OnlineSoftmaxPlan(
        body_invars=tuple(body.invars),
        score_scale=scale_eqn,
        score_mask=mask_eqn,
        score_max=reduce_max_eqn,
        running_max=max_eqn,
        nonempty=nonempty_eqn,
        probability_center=probability_sub_eqn,
        probability_exp=probability_exp_eqn,
        probability_sum=reduce_sum_eqn,
        sum_scale=sum_mul_eqn,
        sum_update=sum_add_eqn,
        accumulator_scale=accumulator_mul_eqn,
    )
    final_div_eqn = _attention_final_div(spec, tile_q, dim)
    _validate_attention_outer(spec, scan, causal, tile_q, dim, final_div_eqn)
    return _AttentionPlan(
        spec=spec,
        scan=scan,
        batch=batch,
        query_length=query_length,
        key_length=key_length,
        heads=heads,
        dim=dim,
        tile_q=tile_q,
        tile_k=tile_k,
        causal=causal,
        body=body,
        dots=(dots[0], dots[1]),
        score_op=score_op,
        value_op=value_op,
        online_softmax=online_softmax,
        final_div=final_div_eqn,
    )


def emit_tensorops_attention(spec: KernelSpec, kernel_name: str | None = None) -> tuple[str, int]:
    """Emit the recognized Pallas online-softmax pattern through Cursor."""
    plan = _recognize_tensorops_attention(spec)
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
        "device float* query [[buffer(0)]],",
        "device float* key [[buffer(1)]],",
        "device float* value [[buffer(2)]],",
        "device float* output [[buffer(3)]],",
        "uint3 group [[threadgroup_position_in_grid]],",
        "uint tid [[thread_index_in_threadgroup]],",
        "uint3 threads_per_group [[threads_per_threadgroup]]",
    )
    source = "\n".join(
        (
            "#include <metal_stdlib>",
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>",
            "",
            "using namespace metal;",
            "using namespace mpp;",
            "",
            f"kernel void {name}(\n    " + "\n    ".join(params) + ")\n{",
            *cursor.lines,
            "}",
            "",
        )
    )
    return source, cursor.threadgroup_bytes
