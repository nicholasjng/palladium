"""Scalar and elementwise primitive lowerings."""

from __future__ import annotations

from jax.extend.core import JaxprEqn

from palladium.emit.addressing import element_strides, flat_index
from palladium.emit.core import CTYPES, RULES, Cursor, CVal, Environment, declare, shaped
from palladium.emit.numeric import (
    ELEMENTWISE,
    PRIMITIVE_INVARS,
    enclosed,
    template_fields,
    typed_expression,
    unwrapped,
)
from palladium.errors import EmitError

# MSL has no erf, erfinv, expm1, or log1p; these are float32 helpers emitted
# once per kernel on first use. erf: Abramowitz and Stegun 7.1.26 (abs error
# 1.5e-7). erfinv: Giles, "Approximating the erfinv function" (2010), the
# single-precision branch. expm1 and log1p switch to short series below
# |x| = 0.25, where the builtin exp and log would cancel.
HELPERS: dict[str, tuple[str, str]] = {
    "erf": (
        "pd_erf",
        """inline float pd_erf(float x) {
    float a = fabs(x);
    float t = 1.0f / fma(0.3275911f, a, 1.0f);
    float poly = fma(fma(fma(fma(1.061405429f, t, -1.453152027f), t, 1.421413741f), t,
                         -0.284496736f), t, 0.254829592f) * t;
    float y = 1.0f - poly * exp(-a * a);
    return x < 0.0f ? -y : y;
}""",
    ),
    "erf_inv": (
        "pd_erfinv",
        """inline float pd_erfinv(float x) {
    float w = -log((1.0f - x) * (1.0f + x));
    float p;
    if (w < 5.0f) {
        w = w - 2.5f;
        p = 2.81022636e-08f;
        p = fma(p, w, 3.43273939e-07f);
        p = fma(p, w, -3.5233877e-06f);
        p = fma(p, w, -4.39150654e-06f);
        p = fma(p, w, 0.00021858087f);
        p = fma(p, w, -0.00125372503f);
        p = fma(p, w, -0.00417768164f);
        p = fma(p, w, 0.246640727f);
        p = fma(p, w, 1.50140941f);
    } else {
        w = sqrt(w) - 3.0f;
        p = -0.000200214257f;
        p = fma(p, w, 0.000100950558f);
        p = fma(p, w, 0.00134934322f);
        p = fma(p, w, -0.00367342844f);
        p = fma(p, w, 0.00573950773f);
        p = fma(p, w, -0.0076224613f);
        p = fma(p, w, 0.00943887047f);
        p = fma(p, w, 1.00167406f);
        p = fma(p, w, 2.83297682f);
    }
    return p * x;
}""",
    ),
    "expm1": (
        "pd_expm1",
        """inline float pd_expm1(float x) {
    if (fabs(x) >= 0.25f) {
        return exp(x) - 1.0f;
    }
    float p = fma(x, 1.0f / 7.0f, 1.0f);
    p = fma(x / 6.0f, p, 1.0f);
    p = fma(x / 5.0f, p, 1.0f);
    p = fma(x / 4.0f, p, 1.0f);
    p = fma(x / 3.0f, p, 1.0f);
    p = fma(x / 2.0f, p, 1.0f);
    return x * p;
}""",
    ),
    "log1p": (
        "pd_log1p",
        """inline float pd_log1p(float x) {
    if (fabs(x) >= 0.25f) {
        return log(1.0f + x);
    }
    float s = x / (2.0f + x);
    float s2 = s * s;
    float p = fma(s2, 1.0f / 9.0f, 1.0f / 7.0f);
    p = fma(s2, p, 1.0f / 5.0f);
    p = fma(s2, p, 1.0f / 3.0f);
    p = fma(s2, p, 1.0f);
    return 2.0f * s * p;
}""",
    ),
}


def _fuses_into_consumer(env: Environment, var, shape: tuple[int, ...], ops: list[CVal]) -> bool:
    """Whether `var` can stay an expression: it is consumed exactly once, by
    an elementwise equation of the same shape at this jaxpr level, and its
    own operands are addressable by the same flat index."""
    consumer = env.sole_consumer(var)
    if consumer is None:
        return False
    name = consumer.primitive.name
    if RULES.get(name) is not _rule_elementwise:
        return False
    consumer_shape = tuple(int(d) for d in shaped(consumer.outvars[0].aval).shape)
    if consumer_shape != shape or not shape:
        return False
    # The consumer must evaluate this operand exactly once per element:
    # only plain table templates and helper calls qualify, since derived
    # templates (sign, rem, integer_pow, min/max with NaN handling) repeat
    # their operands.
    consumer_ctype = CTYPES[str(shaped(consumer.outvars[0].aval).dtype)]
    if typed_expression(name, consumer_ctype) is not None:
        return False
    if name in HELPERS or name == "convert_element_type":
        template = "{a}"
    else:
        template = ELEMENTWISE.get(name)
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
        function, source = HELPERS[opname]
        cursor.require(function, source)
        template = f"{ctype}({function}(float({{a}})))"
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


def _rule_elementwise(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """One rule for every pure elementwise primitive.

    Operands broadcast against `dst`'s shape numpy-style: right-aligned,
    size-1 dims replicate. Every template field must be filled and every
    operand consumed.
    """
    out_aval = shaped(eqn.outvars[0].aval)
    out_shape = tuple(int(d) for d in out_aval.shape)
    out_ctype = CTYPES[str(out_aval.dtype)]
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

    dst = declare(env, cursor, eqn.outvars[0])
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

    def assign(idx_vars: list[str]) -> str:
        dst_idx = flat_index(list(zip(idx_vars, dst_strides)))
        inputs = {name: op.at(op_index(op, idx_vars)) for name, op in zip(PRIMITIVE_INVARS, ops)}
        return f"{dst.at(dst_idx)} = {unwrapped(template.format(**inputs))};"

    with cursor.loop_nest(dst.shape) as idx_vars:
        cursor.emit(assign(idx_vars))


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
