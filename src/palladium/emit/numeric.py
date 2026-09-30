"""Scalar MSL expression templates: the elementwise table and the
dtype-aware variants shared by the rules and the TensorOps lowerings."""

import functools
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
    "tanh": "tanh({a})",
    "exp2": "exp2({a})",
    "tan": "tan({a})",
    "asin": "asin({a})",
    "acos": "acos({a})",
    "atan": "atan({a})",
    "sinh": "sinh({a})",
    "cosh": "cosh({a})",
    "asinh": "asinh({a})",
    "acosh": "acosh({a})",
    "atanh": "atanh({a})",
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
