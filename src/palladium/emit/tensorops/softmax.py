"""Online-softmax emission for the cooperative attention lowering."""

from __future__ import annotations

import dataclasses

from jax.extend.core import JaxprEqn, Literal, Var

from palladium.emit.core import Cursor, CVal, literal, shaped
from palladium.emit.numeric import format_scalar
from palladium.errors import EmitError


def row_reduction_expression(
    eqn: JaxprEqn, input_shape: tuple[int, int], left: str, right: str
) -> str:
    """Render the scalar combine operation of a validated matrix-row reduction.

    Parameters
    ----------
    eqn : JaxprEqn
        Staged equation to inspect or lower.
    input_shape : tuple of int
        Expected score matrix shape (rows, columns).
    left : str
        Left scalar operand.
    right : str
        Right scalar operand.

    Returns
    -------
    str
        Scalar maximum or sum expression.

    Raises
    ------
    EmitError
        The equation is not reduce_max or reduce_sum over axis one with a row-vector
        result.
    """
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
    """Render one cooperative scalar equation using explicit atom bindings.

    Parameters
    ----------
    eqn : JaxprEqn
        Staged equation to inspect or lower.
    ctype : str
        MSL scalar type name, such as "float", "half", "bfloat", "int", "uint", or
        "bool".
    bindings : tuple of tuple of (object, str)
        Pairs of jaxpr atoms and MSL expressions, matched by identity.

    Returns
    -------
    str
        Scalar expression for the equation.

    Raises
    ------
    EmitError
        An input binding is absent or ambiguous, or no supported scalar template exists.

    Notes
    -----
    Each input needs exactly one identity match. Helper-only operations are unavailable
    because this helper does not register functions on a cursor.
    """
    opname = eqn.primitive.name
    operands = []
    for input_atom in eqn.invars:
        matches = [value for atom, value in bindings if atom is input_atom]
        if len(matches) != 1:
            raise EmitError(f"{opname} equation has unbound or ambiguous cooperative operands")
        operands.append(matches[0])
    if opname == "max" and len(operands) == 2:
        return f"max({operands[0]}, {operands[1]})"
    return format_scalar(opname, ctype, tuple(operands))


@dataclasses.dataclass(frozen=True)
class OnlineSoftmaxPlan:
    """Keep the matched equations needed to update streaming softmax row state.

    Attributes
    ----------
    body_invars : tuple of Var or Literal
        Scan-body inputs used to identify row carries.
    score_scale : JaxprEqn
        Score scaling equation.
    score_mask : JaxprEqn or None
        Causal <= mask, or None for noncausal attention.
    score_max : JaxprEqn
        Per-row maximum reduction.
    running_max : JaxprEqn
        Update combining old and current maxima.
    nonempty : JaxprEqn
        Test that the previous running sum is nonempty.
    probability_center : JaxprEqn
        Subtract running maximum from scores.
    probability_exp : JaxprEqn
        Exponentiate centered scores.
    probability_sum : JaxprEqn
        Sum probabilities within each row.
    sum_scale : JaxprEqn
        Rescale the previous running sum.
    sum_update : JaxprEqn
        Add the current probability sum.
    accumulator_scale : JaxprEqn
        Rescale the previous weighted-value accumulator.

    Notes
    -----
    Constructed by the attention dataflow recognizer. Equations must refer to the same scan
    body; arbitrary equation collections are not validated by this dataclass.
    """

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
    row_reduction_expression(eqn, input_shape, value, value)
    return f"simd_max({value})" if eqn.primitive.name == "reduce_max" else f"simd_sum({value})"


def emit_online_softmax_simd(
    cursor: Cursor,
    plan: OnlineSoftmaxPlan,
    scores: CVal,
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
    """Emit one streaming softmax update with rows distributed across SIMD groups.

    Parameters
    ----------
    cursor : Cursor
        Destination for generated MSL statements and storage accounting.
    plan : OnlineSoftmaxPlan
        Matched equations defining the row-state update.
    scores : CVal
        Writable (rows, columns) score tile, overwritten with probabilities.
    row_max : CVal
        Writable (rows,) running maxima.
    row_sum : CVal
        Writable (rows,) running probability sums.
    row_scale : CVal
        Writable (rows,) rescale factors for the value accumulator.
    scale : str
        MSL score-scale expression.
    rows : int
        Positive number of score rows.
    columns : int
        Positive number of score columns.
    lanes : int
        SIMD-group width used by the caller.
    simdgroups : int
        Number of SIMD groups sharing the row work.
    causal_offsets : tuple of str or None, optional
        Absolute key and query tile starts, in that order. Required exactly when the
        plan has a causal mask.

    Raises
    ------
    EmitError
        Tile/carry shapes disagree, causal offsets and mask presence differ, or the mask
        is not <=.

    Notes
    -----
    Each group owns disjoint rows and lane zero commits row state. The caller must insert a
    barrier before other threads consume row_scale or probabilities. Partial lane columns
    are guarded; all-masked prior state is handled by the nonempty test.
    """
    score_storage = scores
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
                plan.score_max, (rows, columns), "lane_max", f"s[{j}]"
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

        def nonempty_operand(atom) -> str:
            if any(atom is carry for carry in plan.body_invars):
                return "old_sum"
            if not isinstance(atom, Literal):
                raise EmitError("online softmax must compare the running sum with a constant")
            return literal(atom).expr

        nonempty_bindings = tuple((atom, nonempty_operand(atom)) for atom in plan.nonempty.invars)
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
                plan.probability_sum, (rows, columns), "lane_sum", f"s[{j}]"
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
    """Render the planned per-element rescaling of the weighted-value accumulator.

    Parameters
    ----------
    plan : OnlineSoftmaxPlan
        Recognized softmax plan whose accumulator equation identifies the carry.
    ctype : str
        MSL scalar type name, such as "float", "half", "bfloat", "int", "uint", or
        "bool".
    element : str
        MSL accumulator element.
    factor : str
        MSL row rescale factor.
    rows : int
        Accumulator row count.
    width : int
        Accumulator column count.

    Returns
    -------
    str
        Scalar expression applying the plan rescale equation.

    Notes
    -----
    The plan must already contain the recognized (rows, width) carry; mismatched plans are
    not repaired by this helper.
    """
    carry = next(
        atom
        for atom in plan.body_invars
        if atom in plan.accumulator_scale.invars and tuple(shaped(atom.aval).shape) == (rows, width)
    )
    factor_atom = next(atom for atom in plan.accumulator_scale.invars if atom is not carry)
    return elementwise_expression(
        plan.accumulator_scale, ctype, ((carry, element), (factor_atom, factor))
    )
