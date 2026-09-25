"""Dtype-aware MSL expressions shared by elementwise and reduction rules."""

from palladium.errors import EmitError


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
