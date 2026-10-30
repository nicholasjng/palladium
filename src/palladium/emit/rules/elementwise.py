"""Scalar and elementwise primitive lowerings."""

from __future__ import annotations

import numpy as np
from jax.extend.core import JaxprEqn

from palladium.emit.addressing import element_strides, flat_index
from palladium.emit.core import RULES, Cursor, CVal, Environment, declare, msl_type, shaped
from palladium.emit.numeric import (
    ELEMENTWISE,
    HELPERS,
    PRIMITIVE_INVARS,
    enclosed,
    helper_template,
    template_fields,
    typed_expression,
    unwrapped,
)
from palladium.emit.rules.control import store_or_declare
from palladium.errors import EmitError

_FLAT_READERS = frozenset(
    {"reduce_sum", "reduce_max", "reduce_min", "cumsum", "cumprod", "cummax", "cummin"}
)


def lowers_elementwise(eqn: JaxprEqn) -> bool:
    """Determine whether an equation uses ordinary per-element scalar emission.

    Parameters
    ----------
    eqn : JaxprEqn
        Staged equation to inspect or lower.

    Returns
    -------
    bool
        True for registered elementwise rules or a two-case boolean select; False for
        other select forms.
    """
    if eqn.primitive.name == "select_n":
        return len(eqn.invars) == 3 and shaped(eqn.invars[0].aval).dtype == np.bool_
    return RULES.get(eqn.primitive.name) is _rule_elementwise


def reads_by_flat_index(env: Environment, var) -> bool:
    """Check whether all uses of a value preserve lazy flat-index evaluation.

    Parameters
    ----------
    env : Environment
        Bindings and def-use information for the current jaxpr.
    var : Var
        Variable considered for an unmaterialized expression.

    Returns
    -------
    bool
        True only for a nonescaping, used variable whose consumers are elementwise or
        supported reduction/prefix readers.

    Notes
    -----
    Unknown, unused, and returned values fail this check. It classifies consumers; separate
    fusion checks decide whether evaluation would be duplicated.
    """
    uses = env.consumer_eqns(var)
    return (
        bool(uses)
        and not env.escapes(var)
        and all(lowers_elementwise(use) or use.primitive.name in _FLAT_READERS for use in uses)
    )


def _fuses_into_consumer(env: Environment, var, shape: tuple[int, ...], ops: list[CVal]) -> bool:
    """Whether `var` can stay an expression: it is consumed exactly once, by
    an elementwise equation of the same shape at this jaxpr level, and its
    own operands are addressable by the same flat index."""
    consumer = env.sole_consumer(var)
    # An inlined jit (jnp.where wraps its select in one) shares the C scope,
    # so follow `var` to its single use inside the body.
    while consumer is not None and consumer.primitive.name in ("jit", "pjit"):
        if consumer.invars.count(var) != 1:
            return False
        body = consumer.params["jaxpr"].jaxpr
        var = body.invars[consumer.invars.index(var)]
        uses = [eqn for eqn in body.eqns if var in eqn.invars]
        if var in body.outvars or len(uses) != 1:
            return False
        consumer = uses[0]
    if consumer is None or consumer.invars.count(var) != 1:
        return False
    name = consumer.primitive.name
    if name in _FLAT_READERS:
        # These read each input element once, by flat index.
        return bool(shape) and all(op.shape in ((), shape) and not op.transposed for op in ops)
    if not lowers_elementwise(consumer):
        return False
    consumer_shape = tuple(int(d) for d in shaped(consumer.outvars[0].aval).shape)
    if consumer_shape != shape or not shape:
        return False
    # The consumer must evaluate this operand exactly once per element, so
    # templates that repeat it (NaN-aware min/max, integer div) and the
    # special cases in _template (sign, rem, integer_pow) do not qualify.
    consumer_ctype = msl_type(shaped(consumer.outvars[0].aval).dtype)
    if name in HELPERS or name == "convert_element_type":
        template = "{a}"
    else:
        template = typed_expression(name, consumer_ctype) or ELEMENTWISE.get(name)
    if template is None:
        return False
    field = "{" + PRIMITIVE_INVARS[consumer.invars.index(var)] + "}"
    if template.count(field) != 1:
        return False
    return all(op.shape in ((), shape) and not op.transposed for op in ops)


def _template(cursor: Cursor, eqn: JaxprEqn, opname: str, ops: list[CVal], ctype: str) -> str:
    """The MSL expression template for one elementwise equation."""
    typed = typed_expression(opname, ctype)
    if typed is not None:
        template = typed
    elif opname == "not":
        template = "(!{a})" if ops[0].ctype == "bool" else "(~{a})"
    elif opname == "sign":
        # Zero and NaN pass through unchanged, keeping signed zero.
        template = f"(({{a}} > 0) ? {ctype}(1) : (({{a}} < 0) ? {ctype}(-1) : {{a}}))"
    elif opname == "rem":
        if ops[0].ctype in ("float", "half", "bfloat"):
            template = f"{ctype}(fmod(float({{a}}), float({{b}})))"
        elif ops[0].ctype == "uint":
            template = "({b} == 0 ? {a} : ({a} % {b}))"
        else:
            # Avoid undefined integer remainder at zero and INT_MIN/-1.
            template = "({b} == 0 ? {a} : ({b} == -1 ? 0 : ({a} % {b})))"
    elif opname == "integer_pow":
        exp: int = eqn.params["y"]
        body = " * ".join(["{a}"] * abs(exp))
        template = f"({body})" if exp > 0 else f"(1.0f / ({body}))"
    elif opname in HELPERS:
        template = helper_template(cursor, opname, ctype)
    elif opname == "round":
        # 0 rounds half away from zero (MSL round), 1 half to even (rint).
        template = "round({a})" if eqn.params["rounding_method"] == 0 else "rint({a})"
    elif opname == "convert_element_type":
        template = f"(({ctype}){{a}})"
    elif opname == "bitcast_convert_type":
        template = f"as_type<{ctype}>({{a}})"
    else:
        template = ELEMENTWISE[opname]

    return template


def scalar_expression(cursor: Cursor, eqn: JaxprEqn, operands: list[CVal]) -> str:
    """Format one element of an elementwise equation from scalar operands.

    Parameters
    ----------
    cursor : Cursor
        Receives helper functions the expression calls.
    eqn : JaxprEqn
        An equation for which `lowers_elementwise` holds.
    operands : list of CVal
        One scalar per invar; `expr` is the operand's expression and `ctype` its
        type.

    Returns
    -------
    str
        MSL expression of the output element.

    Raises
    ------
    EmitError
        The template does not consume exactly the given operands.
    """
    out_ctype = msl_type(shaped(eqn.outvars[0].aval).dtype)
    opname = eqn.primitive.name
    if opname == "integer_pow" and eqn.params["y"] == 0:
        return f"{out_ctype}(1)"
    template = _template(cursor, eqn, opname, operands, out_ctype)
    if template_fields(template) != set(PRIMITIVE_INVARS[: len(operands)]):
        raise EmitError(f"{opname} does not take {len(operands)} operands")
    return enclosed(
        template.format(**{name: op.expr for name, op in zip(PRIMITIVE_INVARS, operands)})
    )


def _rule_elementwise(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """One rule for every pure elementwise primitive.

    Operands broadcast against `dst`'s shape numpy-style: right-aligned,
    size-1 dims replicate. Every template field must be filled and every
    operand consumed.
    """
    out_aval = shaped(eqn.outvars[0].aval)
    out_shape = tuple(int(d) for d in out_aval.shape)
    out_ctype = msl_type(out_aval.dtype)
    ops = [env.val(v) for v in eqn.invars]
    opname = eqn.primitive.name

    if opname == "integer_pow" and eqn.params["y"] == 0:
        dst = declare(env, cursor, eqn.outvars[0])
        cursor.copy(dst, CVal("1", (), dst.ctype), dst.size)
        return
    template = _template(cursor, eqn, opname, ops, out_ctype)
    fields = template_fields(template)
    expected = set(PRIMITIVE_INVARS[: len(ops)])
    if fields != expected:
        raise EmitError(f"{opname} requires {len(fields)} operands, got {len(ops)}")

    if _fuses_into_consumer(env, eqn.outvars[0], out_shape, ops):
        # Every operand shares the output shape (or is a scalar), so one
        # flat index addresses them all; the consumer substitutes it.
        inputs = {
            name: op.at("$i") if op.shape else op.expr for name, op in zip(PRIMITIVE_INVARS, ops)
        }
        env.bind(
            eqn.outvars[0],
            CVal("", out_shape, out_ctype, lazy=enclosed(template.format(**inputs))),
        )
        return

    dst = store_or_declare(env, cursor, eqn.outvars[0])
    rank = len(dst.shape)
    dst_strides = element_strides(dst.shape)

    def op_index(op: CVal, idx_vars: list[str]) -> str:
        # numpy right-alignment: an operand's dim d lines up with dst's
        # dim d + (rank - len(op.shape)); size-1 dims always read index 0.
        rank_diff = rank - len(op.shape)
        op_strides = element_strides(op.shape)
        return flat_index(
            [
                (idx_vars[d + rank_diff], op_strides[d])
                for d, size in enumerate(op.shape)
                if size != 1
            ]
        )

    with cursor.loop_nest(dst.shape) as idx_vars:
        dst_idx = flat_index(list(zip(idx_vars, dst_strides)))
        inputs = {name: op.at(op_index(op, idx_vars)) for name, op in zip(PRIMITIVE_INVARS, ops)}
        cursor.emit(f"{dst.at(dst_idx)} = {unwrapped(template.format(**inputs))};")


# select_n has its own rule in array.py, which falls back to this one.
for _name in [
    *(name for name in ELEMENTWISE if name != "select_n"),
    *HELPERS,
    "round",
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
