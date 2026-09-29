"""Cooperative elementwise emission for values shared by a threadgroup."""

from __future__ import annotations

import dataclasses
import math
import string
from typing import Literal as TypingLiteral

from jax.extend.core import JaxprEqn, Literal, Var

from palladium.emit.core import ELEMENTWISE, Cursor, CVal, EmitError, Environment, shaped
from palladium.emit.numeric import typed_expression


@dataclasses.dataclass(frozen=True)
class CooperativeValue:
    """Threadgroup storage and the rule assigning its elements to lanes."""

    storage: CVal
    ownership: TypingLiteral["flat_strided", "row_strided", "tensorops"]

    def __post_init__(self) -> None:
        if self.storage.space != "threadgroup":
            raise ValueError("cooperative values must use threadgroup storage")


def emit_elementwise(
    cursor: Cursor,
    eqn: JaxprEqn,
    operands: tuple[CVal | CooperativeValue, ...],
    output: CooperativeValue,
    *,
    thread_count: str | None = None,
    row_index: str | None = None,
) -> CooperativeValue:
    """Emit a supported elementwise equation with one lane per output element.

    Shaped operands must match the output shape; scalar operands broadcast.
    The output may alias one input because each lane reads and writes only its
    own element. Flat-strided consumers synchronize before another lane reads
    the result; row-strided work can continue on the same row owner.
    """
    dst = output.storage
    if output.ownership not in ("flat_strided", "row_strided"):
        raise EmitError("cooperative elementwise output needs a lane-owned layout")
    input_values = tuple(op.storage if isinstance(op, CooperativeValue) else op for op in operands)
    shape = tuple(int(d) for d in shaped(eqn.outvars[0].aval).shape)
    if shape != dst.shape or len(input_values) != len(eqn.invars):
        raise EmitError("cooperative elementwise shapes do not match the jaxpr equation")
    if any(op.shape not in ((), shape) for op in input_values):
        raise EmitError("cooperative elementwise operands must be scalar or match the output")

    opname = eqn.primitive.name
    template = typed_expression(opname, dst.ctype) or ELEMENTWISE.get(opname)
    if template is None:
        raise EmitError(f"unsupported cooperative elementwise primitive {opname!r}")
    fields = {field for _, field, _, _ in string.Formatter().parse(template) if field}
    names = string.ascii_lowercase[: len(input_values)]
    if fields != set(names):
        raise EmitError(f"{opname} requires {len(fields)} operands, got {len(operands)}")

    def emit_at(index: str) -> None:
        expressions = {
            name: op.at(index) if op.shape else op.expr
            for name, op in zip(names, input_values, strict=True)
        }
        cursor.emit(f"{dst.at(index)} = {template.format(**expressions)};")

    if output.ownership == "flat_strided":
        if thread_count is None:
            raise EmitError("flat-strided elementwise output requires a thread count")
        with cursor.strided_loop("tid", str(dst.size), thread_count, name="element") as index:
            emit_at(index)
    else:
        if row_index is None or len(shape) != 2:
            raise EmitError("row-strided elementwise output requires a matrix row index")
        with cursor.loop(shape[1], "_column") as column:
            emit_at(f"{row_index} * {shape[1]} + {column}")
    return output


def emit_elementwise_store(
    cursor: Cursor,
    eqn: JaxprEqn,
    operands: tuple[CVal | CooperativeValue, ...],
    output: CVal,
    *,
    thread_count: str,
) -> None:
    """Apply one elementwise equation across a cooperative tile into a buffer."""
    values = tuple(op.storage if isinstance(op, CooperativeValue) else op for op in operands)
    shape = tuple(int(d) for d in shaped(eqn.outvars[0].aval).shape)
    if math.prod(shape) != output.size or len(values) != len(eqn.invars):
        raise EmitError("cooperative store shapes do not match the jaxpr equation")
    if any(
        value.shape
        and value.size != output.size
        and not (len(shape) >= 2 and value.shape == (shape[-1],))
        for value in values
    ):
        raise EmitError("cooperative store operands must be scalar or match the output")

    opname = eqn.primitive.name
    template = typed_expression(opname, output.ctype) or ELEMENTWISE.get(opname)
    if template is None:
        raise EmitError(f"unsupported cooperative store primitive {opname!r}")
    fields = {field for _, field, _, _ in string.Formatter().parse(template) if field}
    names = string.ascii_lowercase[: len(values)]
    if fields != set(names):
        raise EmitError(f"{opname} requires {len(fields)} operands, got {len(values)}")

    with cursor.strided_loop("tid", str(output.size), thread_count, name="element") as index:
        expressions = {
            name: (
                value.at(f"{index} % {shape[-1]}")
                if len(value.shape) == 1 and value.shape == (shape[-1],)
                else value.at(index)
                if value.shape
                else value.expr
            )
            for name, value in zip(names, values, strict=True)
        }
        cursor.emit(f"{output.at(index)} = {template.format(**expressions)};")


def row_reduction_expression(
    eqn: JaxprEqn,
    input_shape: tuple[int, int],
    ctype: str,
    left: str,
    right: str,
) -> str:
    """Return one combine expression for a recognized row reduction."""
    expected = (input_shape[0],)
    if (
        eqn.primitive.name not in ("reduce_max", "reduce_sum")
        or tuple(eqn.params["axes"]) != (1,)
        or tuple(shaped(eqn.invars[0].aval).shape) != input_shape
        or tuple(shaped(eqn.outvars[0].aval).shape) != expected
    ):
        raise EmitError("cooperative reduction must reduce matrix rows along axis 1")
    if eqn.primitive.name == "reduce_max":
        return f"max({left}, {right})"
    return f"({left} + {right})"


def elementwise_expression(
    eqn: JaxprEqn, ctype: str, bindings: tuple[tuple[object, str], ...]
) -> str:
    """Lower one scalar equation instance from its jaxpr-atom bindings."""
    opname = eqn.primitive.name
    operands = []
    for input_atom in eqn.invars:
        matches = [value for atom, value in bindings if atom is input_atom]
        if len(matches) != 1:
            raise EmitError(f"{opname} equation has unbound or ambiguous cooperative operands")
        operands.append(matches[0])
    operands = tuple(operands)
    if opname == "max" and len(operands) == 2:
        return f"max({operands[0]}, {operands[1]})"
    template = typed_expression(opname, ctype) or ELEMENTWISE.get(opname)
    if template is None:
        raise EmitError(f"unsupported cooperative elementwise primitive {opname!r}")
    fields = {field for _, field, _, _ in string.Formatter().parse(template) if field}
    names = string.ascii_lowercase[: len(operands)]
    if len(operands) != len(eqn.invars) or fields != set(names):
        raise EmitError(f"{opname} equation has unsupported arity")
    return template.format(**dict(zip(names, operands, strict=True)))


@dataclasses.dataclass(frozen=True)
class OnlineSoftmaxPlan:
    """Jaxpr equations that update one row of a streaming softmax tile."""

    body_invars: tuple[Var | Literal, ...]
    score_scale: JaxprEqn
    score_mask: JaxprEqn | None
    score_max: JaxprEqn
    running_max: JaxprEqn
    nonempty: JaxprEqn
    probability_center: JaxprEqn
    probability_exp: JaxprEqn
    probability_sum: JaxprEqn
    sum_scale: JaxprEqn
    sum_update: JaxprEqn
    accumulator_scale: JaxprEqn


def _simd_reduction(eqn: JaxprEqn, input_shape: tuple[int, int], value: str) -> str:
    """The SIMD-group reduction matching a recognized row reduction."""
    row_reduction_expression(eqn, input_shape, "float", value, value)
    return f"simd_max({value})" if eqn.primitive.name == "reduce_max" else f"simd_sum({value})"


def emit_online_softmax_simd(
    cursor: Cursor,
    plan: OnlineSoftmaxPlan,
    scores: CooperativeValue,
    row_max: CVal,
    row_sum: CVal,
    row_scale: CVal,
    scale: str,
    *,
    rows: int,
    columns: int,
    lanes: int,
    simdgroups: int,
    causal_offsets: tuple[str, str] | None = None,
) -> None:
    """Emit one online-softmax step with rows owned by SIMD groups.

    SIMD group `sg` owns rows sg, sg + simdgroups, ...; within a row, lane
    `lane` owns columns lane, lane + lanes, .... Each lane reads its scores
    once into registers, the row max and probability sum are SIMD
    reductions, and probabilities are written back in place for the value
    matmul. Lane 0 commits the row carries and the accumulator rescale
    factor to `row_scale`; the caller applies it to the accumulator after
    a barrier. Rows are disjoint across groups, so the step itself needs no
    barrier.
    """
    score_storage = scores.storage
    if scores.ownership != "tensorops":
        raise EmitError("online softmax input scores must be TensorOps-owned")
    if score_storage.shape != (rows, columns):
        raise EmitError("online softmax score tile shape does not match its plan")
    if any(carry.shape != (rows,) for carry in (row_max, row_sum, row_scale)):
        raise EmitError("online softmax row carries must match the score rows")
    if (plan.score_mask is None) != (causal_offsets is None):
        raise EmitError("causal offsets must be present exactly when a score mask is used")
    if plan.score_mask is not None and plan.score_mask.primitive.name != "le":
        raise EmitError("cooperative causal score mask only supports `le`")
    ctype = score_storage.ctype
    per_lane = -(-columns // lanes)
    partial = columns % lanes != 0
    score_atom = next(
        atom
        for atom in plan.score_scale.invars
        if isinstance(atom, Var | Literal) and tuple(shaped(atom.aval).shape) == (rows, columns)
    )
    scale_atom = next(atom for atom in plan.score_scale.invars if atom is not score_atom)
    center_score = next(
        atom
        for atom in plan.probability_center.invars
        if isinstance(atom, Var | Literal) and tuple(shaped(atom.aval).shape) == (rows, columns)
    )
    center_max = next(atom for atom in plan.probability_center.invars if atom is not center_score)
    max_carry = next(
        atom for atom in plan.running_max.invars if atom is not plan.score_max.outvars[0]
    )
    sum_carry = next(
        atom
        for atom in plan.body_invars
        if atom in plan.sum_scale.invars and tuple(shaped(atom.aval).shape) == (rows,)
    )
    sum_factor = next(atom for atom in plan.sum_scale.invars if atom is not sum_carry)
    sum_block = plan.probability_sum.outvars[0]
    sum_scaled = next(atom for atom in plan.sum_update.invars if atom is not sum_block)

    def column(j: int) -> str:
        return "lane" if j == 0 else f"(lane + {j * lanes})"

    with cursor.block(f"for (uint row = sg; row < {rows}; row += {simdgroups})"):
        cursor.emit(f"{ctype} s[{per_lane}];")
        cursor.emit(f"{ctype} lane_max = -INFINITY;")
        for j in range(per_lane):
            index = f"row * {columns} + {column(j)}"
            scaled = elementwise_expression(
                plan.score_scale,
                ctype,
                ((score_atom, score_storage.at(index)), (scale_atom, scale)),
            )
            guards = []
            if partial:
                guards.append(f"{column(j)} < {columns}")
            if causal_offsets is not None:
                guards.append(f"({causal_offsets[0]} + {column(j)}) <= ({causal_offsets[1]} + row)")
            if guards:
                cursor.emit(f"s[{j}] = ({' && '.join(guards)}) ? {scaled} : -INFINITY;")
            else:
                cursor.emit(f"s[{j}] = {scaled};")
            combine = row_reduction_expression(
                plan.score_max, (rows, columns), ctype, "lane_max", f"s[{j}]"
            )
            cursor.emit(f"lane_max = {combine};")
        cursor.emit(
            f"const {ctype} block_max = {_simd_reduction(plan.score_max, (rows, columns), 'lane_max')};"
        )
        cursor.emit(f"const {ctype} old_max = {row_max.at('row')};")
        cursor.emit(f"const {ctype} old_sum = {row_sum.at('row')};")
        new_max = elementwise_expression(
            plan.running_max,
            ctype,
            ((plan.score_max.outvars[0], "block_max"), (max_carry, "old_max")),
        )
        cursor.emit(f"const {ctype} new_max = {new_max};")
        nonempty_bindings = tuple(
            (
                atom,
                "old_sum"
                if any(atom is carry for carry in plan.body_invars)
                else Environment().val(atom).expr,
            )
            for atom in plan.nonempty.invars
        )
        nonempty = elementwise_expression(plan.nonempty, "bool", nonempty_bindings)
        cursor.emit(f"const {ctype} old_scale = {nonempty} ? exp(old_max - new_max) : 0.0f;")
        cursor.emit(f"{ctype} lane_sum = 0.0f;")
        for j in range(per_lane):
            index = f"row * {columns} + {column(j)}"
            centered = elementwise_expression(
                plan.probability_center, ctype, ((center_score, f"s[{j}]"), (center_max, "new_max"))
            )
            probability = elementwise_expression(
                plan.probability_exp, ctype, ((plan.probability_exp.invars[0], centered),)
            )
            store = f"{score_storage.at(index)} = s[{j}];"
            cursor.emit(f"s[{j}] = {probability};")
            if partial:
                with cursor.block(f"if ({column(j)} < {columns})"):
                    cursor.emit(store)
            else:
                cursor.emit(store)
            combine = row_reduction_expression(
                plan.probability_sum, (rows, columns), ctype, "lane_sum", f"s[{j}]"
            )
            cursor.emit(f"lane_sum = {combine};")
        cursor.emit(
            f"const {ctype} block_sum = "
            f"{_simd_reduction(plan.probability_sum, (rows, columns), 'lane_sum')};"
        )
        scaled_sum = elementwise_expression(
            plan.sum_scale, ctype, ((sum_carry, "old_sum"), (sum_factor, "old_scale"))
        )
        sum_update = elementwise_expression(
            plan.sum_update, ctype, ((sum_block, "block_sum"), (sum_scaled, scaled_sum))
        )
        with cursor.block("if (lane == 0)"):
            cursor.emit(f"{row_sum.at('row')} = {sum_update};")
            cursor.emit(f"{row_max.at('row')} = new_max;")
            cursor.emit(f"{row_scale.at('row')} = old_scale;")


def accumulator_rescale_expression(
    plan: OnlineSoftmaxPlan, ctype: str, element: str, factor: str, *, rows: int, width: int
) -> str:
    """The per-element accumulator rescale from the plan's scale equation."""
    carry = next(
        atom
        for atom in plan.body_invars
        if atom in plan.accumulator_scale.invars and tuple(shaped(atom.aval).shape) == (rows, width)
    )
    factor_atom = next(atom for atom in plan.accumulator_scale.invars if atom is not carry)
    return elementwise_expression(
        plan.accumulator_scale, ctype, ((carry, element), (factor_atom, factor))
    )
