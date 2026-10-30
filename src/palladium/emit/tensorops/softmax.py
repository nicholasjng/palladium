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


class _RowUpdate:
    """Scalar expressions of one online-softmax row update, bound to a plan's
    atoms by identity, so the emitted arithmetic is the recognized jaxpr's."""

    def __init__(self, plan: OnlineSoftmaxPlan, ctype: str, rows: int, columns: int):
        if plan.score_mask is not None and plan.score_mask.primitive.name != "le":
            raise EmitError("cooperative causal score mask only supports `le`")
        self.plan, self.ctype, self.shape = plan, ctype, (rows, columns)

        def tile_operand(eqn: JaxprEqn):
            return next(
                atom
                for atom in eqn.invars
                if isinstance(atom, Var | Literal)
                and tuple(shaped(atom.aval).shape) == (rows, columns)
            )

        self.score = tile_operand(plan.score_scale)
        self.scale = next(atom for atom in plan.score_scale.invars if atom is not self.score)
        self.center_score = tile_operand(plan.probability_center)
        self.center_max = next(
            atom for atom in plan.probability_center.invars if atom is not self.center_score
        )
        self.max_carry = next(
            atom for atom in plan.running_max.invars if atom is not plan.score_max.outvars[0]
        )
        self.sum_carry = next(
            atom
            for atom in plan.body_invars
            if atom in plan.sum_scale.invars and tuple(shaped(atom.aval).shape) == (rows,)
        )
        self.sum_factor = next(atom for atom in plan.sum_scale.invars if atom is not self.sum_carry)
        self.sum_block = plan.probability_sum.outvars[0]
        self.sum_scaled = next(
            atom for atom in plan.sum_update.invars if atom is not self.sum_block
        )

    def scaled(self, score: str, scale: str) -> str:
        return elementwise_expression(
            self.plan.score_scale, self.ctype, ((self.score, score), (self.scale, scale))
        )

    def max_combine(self, left: str, right: str) -> str:
        return row_reduction_expression(self.plan.score_max, self.shape, left, right)

    def sum_combine(self, left: str, right: str) -> str:
        return row_reduction_expression(self.plan.probability_sum, self.shape, left, right)

    def new_max(self, block_max: str, old_max: str) -> str:
        return elementwise_expression(
            self.plan.running_max,
            self.ctype,
            ((self.plan.score_max.outvars[0], block_max), (self.max_carry, old_max)),
        )

    def old_scale(self, old_max: str, new_max: str, old_sum: str) -> str:
        def operand(atom) -> str:
            if any(atom is carry for carry in self.plan.body_invars):
                return old_sum
            if not isinstance(atom, Literal):
                raise EmitError("online softmax must compare the running sum with a constant")
            return literal(atom).expr

        bindings = tuple((atom, operand(atom)) for atom in self.plan.nonempty.invars)
        nonempty = elementwise_expression(self.plan.nonempty, "bool", bindings)
        return f"{nonempty} ? exp({old_max} - {new_max}) : 0.0f"

    def probability(self, score: str, new_max: str) -> str:
        centered = elementwise_expression(
            self.plan.probability_center,
            self.ctype,
            ((self.center_score, score), (self.center_max, new_max)),
        )
        return elementwise_expression(
            self.plan.probability_exp,
            self.ctype,
            ((self.plan.probability_exp.invars[0], centered),),
        )

    def new_sum(self, block_sum: str, old_sum: str, old_scale: str) -> str:
        scaled_sum = elementwise_expression(
            self.plan.sum_scale,
            self.ctype,
            ((self.sum_carry, old_sum), (self.sum_factor, old_scale)),
        )
        return elementwise_expression(
            self.plan.sum_update,
            self.ctype,
            ((self.sum_block, block_sum), (self.sum_scaled, scaled_sum)),
        )


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
    ctype = score_storage.ctype
    update = _RowUpdate(plan, ctype, rows, columns)
    per_lane = -(-columns // lanes)
    partial = columns % lanes != 0

    def column(j: int) -> str:
        return "lane" if j == 0 else f"(lane + {j * lanes})"

    with cursor.block(f"for (uint row = sg; row < {rows}; row += {simdgroups})"):
        cursor.emit(f"{ctype} s[{per_lane}];")
        cursor.emit(f"{ctype} lane_max = -INFINITY;")
        for j in range(per_lane):
            scaled = update.scaled(score_storage.at(f"row * {columns} + {column(j)}"), scale)
            guards = []
            if partial:
                guards.append(f"{column(j)} < {columns}")
            if causal_offsets is not None:
                guards.append(f"({causal_offsets[0]} + {column(j)}) <= ({causal_offsets[1]} + row)")
            if guards:
                cursor.emit(f"s[{j}] = ({' && '.join(guards)}) ? {scaled} : -INFINITY;")
            else:
                cursor.emit(f"s[{j}] = {scaled};")
            cursor.emit(f"lane_max = {update.max_combine('lane_max', f's[{j}]')};")
        cursor.emit(
            f"const {ctype} block_max = {_simd_reduction(plan.score_max, (rows, columns), 'lane_max')};"
        )
        cursor.emit(f"const {ctype} old_max = {row_max.at('row')};")
        cursor.emit(f"const {ctype} old_sum = {row_sum.at('row')};")
        cursor.emit(f"const {ctype} new_max = {update.new_max('block_max', 'old_max')};")
        cursor.emit(
            f"const {ctype} old_scale = {update.old_scale('old_max', 'new_max', 'old_sum')};"
        )
        cursor.emit(f"{ctype} lane_sum = 0.0f;")
        for j in range(per_lane):
            store = f"{score_storage.at(f'row * {columns} + {column(j)}')} = s[{j}];"
            cursor.emit(f"s[{j}] = {update.probability(f's[{j}]', 'new_max')};")
            if partial:
                with cursor.block(f"if ({column(j)} < {columns})"):
                    cursor.emit(store)
            else:
                cursor.emit(store)
            cursor.emit(f"lane_sum = {update.sum_combine('lane_sum', f's[{j}]')};")
        cursor.emit(
            f"const {ctype} block_sum = "
            f"{_simd_reduction(plan.probability_sum, (rows, columns), 'lane_sum')};"
        )
        with cursor.block("if (lane == 0)"):
            cursor.emit(
                f"{row_sum.at('row')} = {update.new_sum('block_sum', 'old_sum', 'old_scale')};"
            )
            cursor.emit(f"{row_max.at('row')} = new_max;")
            cursor.emit(f"{row_scale.at('row')} = old_scale;")


def emit_online_softmax_fragments(
    cursor: Cursor,
    plan: OnlineSoftmaxPlan,
    scale: str,
    *,
    rows: int,
    columns: int,
    width: int,
    fragment_rows: int,
    causal: tuple[str, str, str] | None = None,
) -> None:
    """Emit one streaming softmax update on scores held in 8x8 register fragments.

    Parameters
    ----------
    cursor : Cursor
        Destination for generated MSL statements.
    plan : OnlineSoftmaxPlan
        Matched equations defining the row-state update.
    scale : str
        MSL score-scale expression.
    rows, columns : int
        The plan's score tile shape.
    width : int
        Columns of the weighted-value accumulator.
    fragment_rows : int
        Fragment rows this SIMD group owns.
    causal : tuple of str or None, optional
        For a causal mask: a SIMD-group-uniform condition under which the block
        needs masking, then the key position of element `c` of fragment `j` and
        the query position of fragment row `i`, as MSL over `i`, `j` and `c`.
        Required exactly when the plan has a causal mask.

    Notes
    -----
    Expects `_s[i][j]` (scores, overwritten with probabilities) and `_o[i][d]`
    (accumulator) as `vec<float, 2>` fragments, and per-lane row state `_m[i]`,
    `_l[i]`. A lane holds two adjacent elements of one row of each fragment, and
    the four lanes sharing that row differ in lane bits 0 and 3, so row
    reductions finish with two shuffles. No threadgroup memory is involved.
    """
    if (plan.score_mask is None) != (causal is None):
        raise EmitError("causal positions must be present exactly when a score mask is used")
    update = _RowUpdate(plan, "float", rows, columns)
    unroll = '_Pragma("clang loop unroll(full)")'
    elements = (
        f"{unroll} for (short j = 0; j < {columns // 8}; ++j) "
        f"{unroll} for (short c = 0; c < 2; ++c)"
    )
    with cursor.block(f"{unroll} for (short i = 0; i < {fragment_rows}; ++i)"):
        with cursor.block(elements):
            cursor.emit(f"_s[i][j][c] = {update.scaled('_s[i][j][c]', scale)};")
        if causal is not None:
            needed, key, query = causal
            with cursor.block(f"if ({needed})"), cursor.block(elements):
                cursor.emit(f"_s[i][j][c] = ({key}) <= ({query}) ? _s[i][j][c] : -INFINITY;")
        cursor.emit("float block_max = -INFINITY;")
        with cursor.block(elements):
            cursor.emit(f"block_max = {update.max_combine('block_max', '_s[i][j][c]')};")
        for mask in (1, 8):
            shuffled = f"simd_shuffle_xor(block_max, {mask})"
            cursor.emit(f"block_max = {update.max_combine('block_max', shuffled)};")
        cursor.emit(f"const float new_max = {update.new_max('block_max', '_m[i]')};")
        cursor.emit(f"const float old_scale = {update.old_scale('_m[i]', 'new_max', '_l[i]')};")
        cursor.emit("float block_sum = 0.0f;")
        with cursor.block(elements):
            cursor.emit(f"_s[i][j][c] = {update.probability('_s[i][j][c]', 'new_max')};")
            cursor.emit(f"block_sum = {update.sum_combine('block_sum', '_s[i][j][c]')};")
        for mask in (1, 8):
            shuffled = f"simd_shuffle_xor(block_sum, {mask})"
            cursor.emit(f"block_sum = {update.sum_combine('block_sum', shuffled)};")
        cursor.emit(f"_l[i] = {update.new_sum('block_sum', '_l[i]', 'old_scale')};")
        cursor.emit("_m[i] = new_max;")
        rescaled = accumulator_rescale_expression(
            plan, "float", "_o[i][d][c]", "old_scale", rows=rows, width=width
        )
        with cursor.block(
            f"{unroll} for (short d = 0; d < {width // 8}; ++d) "
            f"{unroll} for (short c = 0; c < 2; ++c)"
        ):
            cursor.emit(f"_o[i][d][c] = {rescaled};")


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
