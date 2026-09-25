"""Cooperative elementwise emission for values shared by a threadgroup."""

from __future__ import annotations

import dataclasses
import math
import string
from collections.abc import Callable
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


def emit_row_reduction(
    cursor: Cursor,
    eqn: JaxprEqn,
    row_index: str,
    initial: str,
    value_at: Callable[[str, str], str],
    *,
    ctype: str,
) -> str:
    """Emit a serial row fold owned by one lane and return its local value.

    `value_at` may emit an in-place transform of the element before returning
    the value to fold. That lets masked reductions stay fused with their row
    owner without allocating another group-wide tile or adding a barrier.
    """
    input_shape = tuple(int(d) for d in shaped(eqn.invars[0].aval).shape)
    if len(input_shape) != 2:
        raise EmitError("cooperative row reduction requires a rank-2 input")
    result = cursor.fresh("_reduce")
    cursor.emit(f"{ctype} {result} = {initial};")
    with cursor.loop(input_shape[1], "_column") as column:
        index = f"{row_index} * {input_shape[1]} + {column}"
        value = value_at(index, column)
        combine = row_reduction_expression(eqn, input_shape, ctype, result, value)
        cursor.emit(f"{result} = {combine};")
    return result


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


def emit_online_softmax_rows(
    cursor: Cursor,
    plan: OnlineSoftmaxPlan,
    scores: CooperativeValue,
    row_max: CVal,
    row_sum: CVal,
    accumulator: CVal,
    scale_operands: tuple[CVal | CooperativeValue, ...],
    *,
    rows: int,
    columns: int,
    width: int,
    thread_count: str,
    causal_offsets: tuple[str, str] | None = None,
) -> None:
    """Emit row ownership, masking, online-softmax carries, and rescaling.

    Each row belongs to one lane. The score max and probability sum fold
    callbacks can update shared scores in place, keeping the recurrence fused
    without extra scratch or barriers.
    """
    score_storage = scores.storage
    if scores.ownership != "tensorops":
        raise EmitError("online softmax input scores must be TensorOps-owned")
    if score_storage.shape != (rows, columns):
        raise EmitError("online softmax score tile shape does not match its plan")
    if row_max.shape != (rows,) or row_sum.shape != (rows,):
        raise EmitError("online softmax row carries must match the score rows")
    if accumulator.shape != (rows, width):
        raise EmitError("online softmax accumulator shape does not match its rows")
    if (plan.score_mask is None) != (causal_offsets is None):
        raise EmitError("causal offsets must be present exactly when a score mask is used")

    with cursor.strided_loop("tid", str(rows), thread_count, name="row"):
        emit_elementwise(
            cursor,
            plan.score_scale,
            scale_operands,
            CooperativeValue(score_storage, "row_strided"),
            row_index="row",
        )
    with cursor.strided_loop("tid", str(rows), thread_count, name="row"):
        block_max = emit_row_reduction(
            cursor,
            plan.score_max,
            "row",
            "-INFINITY",
            lambda index, column: _masked_score(
                cursor, score_storage, index, column, plan.score_mask, causal_offsets
            ),
            ctype=score_storage.ctype,
        )
        cursor.emit(f"const {score_storage.ctype} block_max = {block_max};")
        cursor.emit(f"const {score_storage.ctype} old_max = {row_max.at('row')};")
        max_carry = next(
            atom for atom in plan.running_max.invars if atom is not plan.score_max.outvars[0]
        )
        new_max = elementwise_expression(
            plan.running_max,
            score_storage.ctype,
            ((plan.score_max.outvars[0], "block_max"), (max_carry, "old_max")),
        )
        cursor.emit(f"const {score_storage.ctype} new_max = {new_max};")
        nonempty_bindings = tuple(
            (
                atom,
                row_sum.at("row")
                if any(atom is carry for carry in plan.body_invars)
                else Environment().val(atom).expr,
            )
            for atom in plan.nonempty.invars
        )
        nonempty = elementwise_expression(plan.nonempty, "bool", nonempty_bindings)
        cursor.emit(
            f"const {score_storage.ctype} old_scale = {nonempty} ? exp(old_max - new_max) : 0.0f;"
        )

        def probability(index: str, _column: str) -> str:
            score_atom = next(
                atom
                for atom in plan.probability_center.invars
                if isinstance(atom, Var | Literal)
                and tuple(shaped(atom.aval).shape) == (rows, columns)
            )
            centered = elementwise_expression(
                plan.probability_center,
                score_storage.ctype,
                (
                    (score_atom, score_storage.at(index)),
                    (
                        next(
                            atom
                            for atom in plan.probability_center.invars
                            if atom is not score_atom
                        ),
                        "new_max",
                    ),
                ),
            )
            value = elementwise_expression(
                plan.probability_exp,
                score_storage.ctype,
                ((plan.probability_exp.invars[0], centered),),
            )
            local = cursor.fresh("_probability")
            cursor.emit(f"const {score_storage.ctype} {local} = {value};")
            cursor.emit(f"{score_storage.at(index)} = {local};")
            return local

        block_sum = emit_row_reduction(
            cursor,
            plan.probability_sum,
            "row",
            "0.0f",
            probability,
            ctype=score_storage.ctype,
        )
        sum_carry = next(
            atom
            for atom in plan.body_invars
            if atom in plan.sum_scale.invars and tuple(shaped(atom.aval).shape) == (rows,)
        )
        sum_factor = next(atom for atom in plan.sum_scale.invars if atom is not sum_carry)
        scaled_sum = elementwise_expression(
            plan.sum_scale,
            score_storage.ctype,
            ((sum_carry, row_sum.at("row")), (sum_factor, "old_scale")),
        )
        sum_update = elementwise_expression(
            plan.sum_update,
            score_storage.ctype,
            (
                (plan.probability_sum.outvars[0], block_sum),
                (
                    next(
                        atom
                        for atom in plan.sum_update.invars
                        if atom is not plan.probability_sum.outvars[0]
                    ),
                    scaled_sum,
                ),
            ),
        )
        cursor.emit(f"{row_sum.at('row')} = {sum_update};")
        cursor.emit(f"{row_max.at('row')} = new_max;")
        accumulator_carry = next(
            atom
            for atom in plan.body_invars
            if atom in plan.accumulator_scale.invars
            and tuple(shaped(atom.aval).shape) == (rows, width)
        )
        accumulator_factor = next(
            atom for atom in plan.accumulator_scale.invars if atom is not accumulator_carry
        )
        with cursor.loop(width, "_d") as d:
            index = f"row * {width} + {d}"
            rescaled = elementwise_expression(
                plan.accumulator_scale,
                score_storage.ctype,
                ((accumulator_carry, accumulator.at(index)), (accumulator_factor, "old_scale")),
            )
            cursor.emit(f"{accumulator.at(index)} = {rescaled};")


def _masked_score(
    cursor: Cursor,
    scores: CVal,
    index: str,
    column: str,
    mask: JaxprEqn | None,
    offsets: tuple[str, str] | None,
) -> str:
    score = cursor.fresh("_score")
    cursor.emit(f"{scores.ctype} {score} = {scores.at(index)};")
    if mask is not None and offsets is not None:
        if mask.primitive.name != "le":
            raise EmitError("cooperative causal score mask only supports `le`")
        with cursor.block(f"if (!(({offsets[0]} + {column}) <= ({offsets[1]} + row)))"):
            cursor.emit(f"{score} = -INFINITY;")
    cursor.emit(f"{scores.at(index)} = {score};")
    return score
