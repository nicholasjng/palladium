"""Scalar and elementwise primitive lowerings."""

from __future__ import annotations

from jax.extend.core import JaxprEqn

from palladium.emit.core import (
    ELEMENTWISE,
    PRIMITIVE_INVARS,
    RULES,
    Cursor,
    CVal,
    EmitError,
    Environment,
    _element_strides,
    _flat_index,
    _template_fields,
    _unwrapped,
    declare,
)
from palladium.emit.numeric import typed_expression


def _rule_elementwise(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """One rule for every pure elementwise primitive.

    Three templates are derived rather than looked up: integer_pow expands
    to repeated multiplication (reciprocal for negative y); convert_element_type
    casts to the output's ctype; bitcast_convert_type uses as_type<T>.

    Operands broadcast against `dst`'s shape numpy-style: right-aligned,
    size-1 dims replicate. The arity check requires every template field
    to be filled and every operand consumed.
    """

    dst = declare(env, cursor, eqn.outvars[0])
    ops = [env.val(v) for v in eqn.invars]
    opname = eqn.primitive.name

    typed = typed_expression(opname, dst.ctype)
    if typed is not None:
        template = typed
    elif opname == "not":
        template = "(!{a})" if ops[0].ctype == "bool" else "(~{a})"
    elif opname == "sign":
        # Return the original zero/NaN, preserving signed zero and NaNs.
        template = f"(({{a}} > 0) ? {dst.ctype}(1) : (({{a}} < 0) ? {dst.ctype}(-1) : {{a}}))"
    elif opname == "rem":
        if ops[0].ctype in ("float", "half", "bfloat"):
            template = f"{dst.ctype}(fmod(float({{a}}), float({{b}})))"
        elif ops[0].ctype == "uint":
            template = "({b} == 0 ? {a} : ({a} % {b}))"
        else:
            # Avoid undefined integer remainder at zero and INT_MIN/-1.
            template = "({b} == 0 ? {a} : ({b} == -1 ? 0 : ({a} % {b})))"
    elif opname == "integer_pow":
        exp: int = eqn.params["y"]
        if exp == 0:
            cursor.copy(dst, CVal("1", (), dst.ctype), dst.size)
            return
        body = " * ".join(["{a}"] * abs(exp))
        template = f"({body})" if exp > 0 else f"(1.0f / ({body}))"
    elif opname == "convert_element_type":
        template = f"(({dst.ctype}){{a}})"
    elif opname == "bitcast_convert_type":
        # Reinterprets bits without a numeric conversion (unlike
        # convert_element_type's cast); Metal's equivalent is as_type<T>.
        template = f"as_type<{dst.ctype}>({{a}})"
    elif opname == "select_n":
        if (pred_type := ops[0].ctype) != "bool":
            raise EmitError(f"select_n requires predicate of type bool, got {pred_type}")
        template = ELEMENTWISE[opname]
    else:
        template = ELEMENTWISE[opname]

    fields = _template_fields(template)
    expected = set(PRIMITIVE_INVARS[: len(ops)])
    if fields != expected:
        raise EmitError(f"{opname} requires {len(fields)} operands, got {len(ops)}")

    rank = len(dst.shape)
    dst_strides = _element_strides(dst.shape)

    def op_index(op: CVal, idx_vars: list[str]) -> str:
        # numpy right-alignment: an operand's dim d lines up with dst's
        # dim d + (rank - len(op.shape)); size-1 dims always read index 0.
        rank_diff = rank - len(op.shape)
        op_strides = _element_strides(op.shape)
        return _flat_index(
            [
                (idx_vars[d + rank_diff], op_strides[d])
                for d, size in enumerate(op.shape)
                if size != 1
            ]
        )

    def assign(idx_vars: list[str]) -> str:
        dst_idx = _flat_index(list(zip(idx_vars, dst_strides)))
        inputs = {name: op.at(op_index(op, idx_vars)) for name, op in zip(PRIMITIVE_INVARS, ops)}
        return f"{dst.at(dst_idx)} = {_unwrapped(template.format(**inputs))};"

    idx_vars = [cursor.fresh("_i") for _ in range(rank)]

    def emit_loops(d: int) -> None:
        if d == rank:
            cursor.emit(assign(idx_vars))
            return
        with cursor.block(
            f"for (uint {idx_vars[d]} = 0; {idx_vars[d]} < {dst.shape[d]}; ++{idx_vars[d]})"
        ):
            emit_loops(d + 1)

    emit_loops(0)


for _name in [
    *ELEMENTWISE,
    "integer_pow",
    "convert_element_type",
    "bitcast_convert_type",
    "not",
    "sign",
    "rem",
    "min",
    "max",
    "log2",
    "one_minus_square",
    "shift_left",
    "shift_right_logical",
    "shift_right_arithmetic",
]:
    RULES[_name] = _rule_elementwise
