"""MSL scalar types, literals and expression templates of the PJRT compiler."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from palladium.emit.core import CTYPES
from palladium.emit.numeric import format_scalar
from palladium.errors import EmitError
from palladium.stablehlo.program import UnsupportedProgram


class Helpers(Protocol):
    """Where a scalar template registers the MSL function it calls."""

    def require(self, name: str, source: str) -> None: ...


# StableHLO op names to the JAX primitive names of palladium's scalar templates.
ELEMENTWISE = {
    "add": "add",
    "subtract": "sub",
    "multiply": "mul",
    "divide": "div",
    "maximum": "max",
    "minimum": "min",
    "power": "pow",
    "atan2": "atan2",
    "negate": "neg",
    "abs": "abs",
    "exponential": "exp",
    "exponential_minus_one": "expm1",
    "exp2": "exp2",
    "log": "log",
    "log2": "log2",
    "log_plus_one": "log1p",
    "sine": "sin",
    "cosine": "cos",
    "tan": "tan",
    "tanh": "tanh",
    "sqrt": "sqrt",
    "rsqrt": "rsqrt",
    "logistic": "logistic",
    "floor": "floor",
    "ceil": "ceil",
    "is_finite": "is_finite",
    "clamp": "clamp",
    "and": "and",
    "or": "or",
    "xor": "xor",
    "shift_left": "shift_left",
    "shift_right_logical": "shift_right_logical",
    "shift_right_arithmetic": "shift_right_arithmetic",
}


# The Pallas emitter's types, and those only this compiler handles: adding
# them there would broaden the type contract of Pallas kernels.
_CTYPES = CTYPES | {
    np.dtype(np.int64): "long",
    np.dtype(np.uint64): "ulong",
    np.dtype(np.int8): "char",
    np.dtype(np.uint8): "uchar",
    np.dtype(np.int16): "short",
    np.dtype(np.uint16): "ushort",
}


# 8- and 16-bit integers: the 32-bit type they compute in, their unsigned and
# signed counterparts, and their width. Results wrap when cast back.
NARROW = {
    "char": ("int", "uchar", "char", 8),
    "uchar": ("uint", "uchar", "char", 8),
    "short": ("int", "ushort", "short", 16),
    "ushort": ("uint", "ushort", "short", 16),
}


def ctype_of(dtype: np.dtype) -> str:
    if dtype not in _CTYPES:
        raise UnsupportedProgram(f"metal: kernels on {dtype} are not supported")
    return _CTYPES[dtype]


def accumulator(ctype: str) -> str:
    """The type a fold over `ctype` accumulates in: float32 for the half
    types, as XLA does, and 32 bits for narrow integers, which wrap alike."""
    if ctype in ("half", "bfloat"):
        return "float"
    return NARROW[ctype][0] if ctype in NARROW else ctype


def literal(value: float, ctype: str) -> str:
    """`value` as an MSL literal of `ctype`, one operand anywhere: a negative
    one is parenthesized, since `-{a}` of a bare one would spell a decrement."""
    if ctype == "ulong":
        return f"{int(value) % (1 << 64)}ul"
    if ctype == "long":
        if int(value) == -(1 << 63):
            return "(-9223372036854775807l - 1l)"
        return f"({int(value)}l)" if value < 0 else f"{int(value)}l"
    if ctype in ("float", "half", "bfloat"):
        if math.isinf(value):
            expr = "-INFINITY" if value < 0 else "INFINITY"
        elif math.isnan(value):
            expr = "NAN"
        else:
            expr = f"{np.float32(value)!s}f"
        if ctype != "float":
            return f"{ctype}({expr})"
        return f"({expr})" if expr.startswith("-") else expr
    return f"{ctype}({int(value)})"


# Templates palladium's scalar table lacks, by operand ctype class.
EXTRA = {
    "sign": {
        "float": "(isnan({a}) ? {a} : sign({a}))",
        "int": "int(({a} > 0) - ({a} < 0))",
        "uint": "uint({a} > 0u)",
    },
    "remainder": {
        "float": "pd_fmod(float({a}), float({b}))",  # needs numeric.FMOD
        "int": "({b} == 0 ? {a} : ({b} == -1 ? 0 : {a} % {b}))",
        "uint": "({b} == 0u ? {a} : {a} % {b})",
    },
    "round_nearest_even": {"float": "rint({a})"},
    "round_nearest_afz": {"float": "round({a})"},
    # MSL has no cbrt, and pow rejects negative bases.
    "cbrt": {"float": "copysign(pow(fabs(float({a})), 1.0f / 3.0f), float({a}))"},
    "popcnt": {"int": "popcount({a})", "uint": "popcount({a})"},
    "count_leading_zeros": {"int": "clz({a})", "uint": "clz({a})"},
    "not": {"bool": "!{a}", "int": "~{a}", "uint": "~{a}"},
    "fma": {"float": "fma(float({a}), float({b}), float({c}))"},
}


# The MSL function ranking floats in IEEE total order: as integers, negative
# NaNs come first, then -inf up to -0, +0 up to +inf, and positive NaNs.
TOTAL_ORDER = (
    "total_order",
    """inline int total_order(float x) {
    int bits = as_type<int>(x);
    return bits < 0 ? (bits ^ 0x7fffffff) : bits;
}""",
)

COMPARISONS = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


# Ops computing each element from the same element of their operands.
SCALAR = {
    *ELEMENTWISE,
    *EXTRA,
    "compare",
    "select",
    "convert",
    "bitcast_convert",
    "reduce_precision",
}


# Threefry-2x32 with 20 rounds, the generator behind `rng_bits`.
THREEFRY = (
    "threefry2x32",
    """inline uint2 threefry2x32(uint2 key, uint2 counter) {
    uint ks[3] = {key.x, key.y, 0x1BD11BDAu ^ key.x ^ key.y};
    uint x0 = counter.x + ks[0], x1 = counter.y + ks[1];
    const uint rotations[8] = {13u, 15u, 26u, 6u, 17u, 29u, 16u, 24u};
    for (uint group = 0u; group < 5u; ++group) {
        for (uint k = 0u; k < 4u; ++k) {
            uint r = rotations[(group % 2u) * 4u + k];
            x0 += x1;
            x1 = ((x1 << r) | (x1 >> (32u - r))) ^ x0;
        }
        x0 += ks[(group + 1u) % 3u];
        x1 += ks[(group + 2u) % 3u] + group + 1u;
    }
    return uint2(x0, x1);
}""",
)


def wide_scalar(name: str, ctype: str, values: list[str]) -> str:
    """64-bit integer expressions, with wrapping arithmetic and bounded shifts."""
    a, *rest = values
    b = rest[0] if rest else ""

    def bits(expr: str) -> str:
        return f"as_type<long>({expr})" if ctype == "long" else expr

    if name in ("add", "subtract", "multiply"):
        op = {"add": "+", "subtract": "-", "multiply": "*"}[name]
        return bits(f"(ulong({a}) {op} ulong({b}))")
    if name in ("negate", "abs"):
        negative = bits(f"(0ul - ulong({a}))")
        return negative if name == "negate" else f"({a} < 0 ? {negative} : {a})"
    if name == "shift_right_arithmetic":
        expr = f"(as_type<long>(ulong({a})) >> uint(min(ulong({b}), 63ul)))"
        return expr if ctype == "long" else f"as_type<ulong>({expr})"
    if name in ("shift_left", "shift_right_logical"):
        op = "<<" if name == "shift_left" else ">>"
        return bits(f"(ulong({b}) < 64ul ? (ulong({a}) {op} uint(ulong({b}) & 63ul)) : 0ul)")
    if name in ("maximum", "minimum"):
        return f"{('max' if name == 'maximum' else 'min')}({a}, {b})"
    if name == "clamp":
        low, value, high = values
        return f"min(max({value}, {low}), {high})"
    if name in ("and", "or", "xor"):
        op = {"and": "&", "or": "|", "xor": "^"}[name]
        return f"({a} {op} {b})"
    if name == "not":
        return f"(~{a})"
    if name == "sign":
        return f"{ctype}(({a} > 0) - ({a} < 0))"
    if name in ("popcnt", "count_leading_zeros"):
        return f"{ctype}({'popcount' if name == 'popcnt' else 'clz'}(ulong({a})))"
    if name in ("divide", "remainder"):
        zero = f"{ctype}(~0ul)" if name == "divide" else a
        expr = f"({a} {'/' if name == 'divide' else '%'} {b})"
        if ctype == "long":
            overflow = bits(f"(0ul - ulong({a}))") if name == "divide" else "0l"
            expr = f"({b} == -1l ? {overflow} : {expr})"
        return f"({b} == 0 ? {zero} : {expr})"
    raise UnsupportedProgram(f"metal: stablehlo.{name} on {ctype} is not supported")


def narrow_scalar(name: str, ctype: str, values: list[str], kernel: Helpers) -> str:
    """8- and 16-bit integer expressions, computed in 32 bits and wrapped.

    Sign- or zero-extended operands give the low bits of every arithmetic,
    bitwise and comparison result, and the 32-bit templates' division and
    shift rules hold at the narrow width too. Logical right shifts and bit
    counts see the unsigned bit pattern, arithmetic right shifts the signed.
    """
    wide, unsigned, signed, bits = NARROW[ctype]
    if name in ("shift_right_logical", "popcnt", "count_leading_zeros"):
        wide, pattern = "uint", unsigned
    elif name == "shift_right_arithmetic":
        wide, pattern = "int", signed
    else:
        pattern = ctype
    first = values[0] if pattern == ctype else f"as_type<{pattern}>({values[0]})"
    values = [f"{wide}({first})", *(f"{wide}({v})" for v in values[1:])]
    if name == "count_leading_zeros":
        expr = f"(clz({values[0]}) - {32 - bits}u)"
    elif name in EXTRA:
        if wide not in EXTRA[name]:
            raise UnsupportedProgram(f"metal: stablehlo.{name} on {ctype}")
        expr = EXTRA[name][wide].format(**dict(zip("ab", values, strict=False)))
    else:
        try:
            expr = format_scalar(ELEMENTWISE[name], wide, tuple(values), kernel)
        except EmitError as e:
            raise UnsupportedProgram(f"metal: stablehlo.{name}: {e}") from e
    return f"{ctype}({expr})"
