"""Scalar MSL expression templates: the elementwise table and the
dtype-aware variants shared by the rules and the TensorOps lowerings."""

import functools
import re
import string

from palladium.errors import EmitError

MAX_PRIMITIVE_ARITY = 6
PRIMITIVE_INVARS = string.ascii_lowercase[:MAX_PRIMITIVE_ARITY]


@functools.cache
def template_fields(template: str) -> frozenset[str]:
    return frozenset(f for _, f, _, _ in string.Formatter().parse(template) if f)


def unwrapped(expr: str) -> str:
    if not (expr.startswith("(") and expr.endswith(")")):
        return expr
    depth = 0
    for i, ch in enumerate(expr):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and i < len(expr) - 1:
            # The leading paren closes early: shapes like (a) * (b).
            return expr
    return expr[1:-1]


_CALLEE = re.compile(r"[A-Za-z_][\w:<>]*\(")
_OPERAND = re.compile(r"[\w.$]+(\[[^\[\]]*\])?")


def _closes_last(expr: str, start: int) -> bool:
    """Whether the paren opened at `start` is closed by the final character."""
    depth = 0
    for i in range(start, len(expr)):
        depth += expr[i] == "("
        depth -= expr[i] == ")"
        if depth == 0:
            return i == len(expr) - 1
    return False


def enclosed(expr: str) -> str:
    """`expr`, parenthesized unless it already binds as one operand: an
    identifier, literal, or indexed name, a call, or one bracketed group."""
    if _OPERAND.fullmatch(expr) or (expr.startswith("(") and _closes_last(expr, 0)):
        return expr
    call = _CALLEE.match(expr)
    if call and _closes_last(expr, call.end() - 1):
        return expr
    return f"({expr})"


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

# Metal's tanh, sinh, asinh, and atanh lose relative accuracy below
# |x| ~ 1e-3 in every math mode (tanh(1e-8) returns 0), and their precise::
# forms are 2.4x slower. Below 0.01, two series terms are exact to ~1e-9.
_SMALL_ARGUMENT = {
    "tanh": ("pd_tanh", "1.0f / -3.0f"),
    "sinh": ("pd_sinh", "1.0f / 6.0f"),
    "asinh": ("pd_asinh", "1.0f / -6.0f"),
    "atanh": ("pd_atanh", "1.0f / 3.0f"),
}
for _op, (_name, _coefficient) in _SMALL_ARGUMENT.items():
    HELPERS[_op] = (
        _name,
        f"""inline float {_name}(float x) {{
    return fabs(x) < 0.01f ? x * fma(x * x, {_coefficient}, 1.0f) : {_op}(x);
}}""",
    )


ELEMENTWISE: dict[str, str] = {
    # binary
    "add": "({a} + {b})",
    # AD cotangent accumulation; supported numeric arrays use ordinary addition.
    "add_any": "({a} + {b})",
    "sub": "({a} - {b})",
    "mul": "({a} * {b})",
    "div": "({a} / {b})",
    "pow": "pow({a}, {b})",
    # unary
    "neg": "-{a}",
    "abs": "fabs({a})",
    "exp": "exp({a})",
    "log": "log({a})",
    "sin": "sin({a})",
    "cos": "cos({a})",
    "sqrt": "sqrt({a})",
    "rsqrt": "rsqrt({a})",
    "exp2": "exp2({a})",
    "tan": "tan({a})",
    "asin": "asin({a})",
    "acos": "acos({a})",
    "atan": "atan({a})",
    "cosh": "cosh({a})",
    "acosh": "acosh({a})",
    "atan2": "atan2({a}, {b})",
    "floor": "floor({a})",
    "ceil": "ceil({a})",
    "square": "({a} * {a})",
    "logistic": "(1.0f / (1.0f + exp(-{a})))",
    "is_finite": "isfinite({a})",
    # ternary
    "select_n": "({a} ? {c} : {b})",  # a: predicate (bool), c when true, b when false
    "clamp": "clamp({b}, {a}, {c})",  # jaxpr order (min, x, max) -> metal (x, min, max)
    # logical
    "lt": "({a} < {b})",
    "le": "({a} <= {b})",
    "gt": "({b} < {a})",
    "ge": "({b} <= {a})",
    "eq": "({a} == {b})",
    "ne": "({a} != {b})",
    # bitwise, integer/bool operands.
    "and": "({a} & {b})",
    "or": "({a} | {b})",
    "xor": "({a} ^ {b})",
}


def extremum(op: str, ctype: str, a: str, b: str) -> str:
    if ctype == "bool":
        return f"({a} {'&&' if op == 'min' else '||'} {b})"
    if ctype in ("int", "uint"):
        return f"{op}({ctype}({a}), {ctype}({b}))"
    # Metal extrema need explicit NaN propagation and signed-zero ties.
    bits = "|" if op == "min" else "&"
    finite = (
        f"({a} == 0 && {b} == 0 ? "
        f"as_type<float>(as_type<uint>(float({a})) {bits} as_type<uint>(float({b})))"
        f" : f{op}(float({a}), float({b})))"
    )
    return f"{ctype}(isnan(float({a})) ? float({a}) : (isnan(float({b})) ? float({b}) : {finite}))"


def extremum_identity(op: str, ctype: str) -> str:
    if ctype == "int":
        return "2147483647" if op == "min" else "(-2147483647 - 1)"
    if ctype == "uint":
        return "4294967295u" if op == "min" else "0u"
    if ctype == "bool":
        return "true" if op == "min" else "false"
    return f"{ctype}({'INFINITY' if op == 'min' else '-INFINITY'})"


def typed_expression(op: str, ctype: str) -> str | None:
    """Return a format template, or None to use the generic rule."""
    if op in ("min", "max"):
        return extremum(op, ctype, "{a}", "{b}")
    if op == "log2":
        return f"{ctype}(precise::log2(float({{a}})))"
    if op == "one_minus_square":
        if ctype in ("int", "uint"):
            expr = "((1u + uint({a})) * (1u - uint({a})))"
            return f"as_type<int>({expr})" if ctype == "int" else expr
        if ctype not in ("float", "half", "bfloat"):
            raise EmitError(f"one_minus_square does not support {ctype}")
        # Match upstream's factored lowering, including narrow intermediates.
        return f"{ctype}({ctype}({ctype}(1) + {{a}}) * {ctype}({ctype}(1) - {{a}}))"
    if op == "abs" and ctype in ("int", "uint"):
        return "({a})" if ctype == "uint" else "({a} < 0 ? as_type<int>(0u - uint({a})) : {a})"
    if op == "div" and ctype in ("int", "uint"):
        if ctype == "uint":
            return "({b} == 0u ? 4294967295u : ({a} / {b}))"
        return "({b} == 0 ? -1 : ({b} == -1 ? as_type<int>(0u - uint({a})) : ({a} / {b})))"
    if op.startswith("shift_"):
        if ctype not in ("int", "uint"):
            raise EmitError(f"{op} requires int32 or uint32, got {ctype}")
        if op == "shift_right_arithmetic":
            # Sign extension refers to the high bit, even for uint32 operands.
            expr = "(as_type<int>(uint({a})) >> min(uint({b}), 31u))"
            return f"as_type<uint>({expr})" if ctype == "uint" else expr
        symbol = "<<" if op == "shift_left" else ">>"
        expr = f"(uint({{b}}) < 32u ? (uint({{a}}) {symbol} (uint({{b}}) & 31u)) : 0u)"
        return f"as_type<int>({expr})" if ctype == "int" else expr
    return None


def helper_template(cursor, op: str, ctype: str) -> str:
    """The call template for a HELPERS op, registering its function."""
    function, source = HELPERS[op]
    cursor.require(function, source)
    return f"{ctype}({function}(float({{a}})))"


def format_scalar(op: str, ctype: str, operands: tuple[str, ...], cursor=None) -> str:
    """`op` applied to scalar C `operands` in `ctype`, from the typed,
    helper, or table template; raises EmitError for an unknown op or wrong
    arity. Helper ops need the `cursor` that emits their function."""
    if op in HELPERS and cursor is not None:
        template = helper_template(cursor, op, ctype)
    else:
        template = typed_expression(op, ctype) or ELEMENTWISE.get(op)
    if template is None:
        raise EmitError(f"no scalar template for primitive {op!r}")
    names = PRIMITIVE_INVARS[: len(operands)]
    if template_fields(template) != set(names):
        raise EmitError(
            f"{op} takes {len(template_fields(template))} operands, got {len(operands)}"
        )
    return template.format(**dict(zip(names, operands, strict=True)))
