"""StableHLO programs to Metal kernels, for the `metal` PJRT plugin.

The plugin's PJRT_Client_Compile passes the portable artifact jax serializes to
`compile_program`, which parses it with jaxlib's MLIR bindings and returns a
`Program`: buffer slots and kernel launches, which `palladium._pjrt` hands to
the plugin. A program may hold elementwise and index-mapping ops
(broadcast, reshape, transpose, slices, concatenate, pad, reverse, iota), fused
into one kernel per output shape. Single-result reductions end a fusion and
materialize a buffer for subsequent kernels; unsupported ops and reducer bodies
raise `UnsupportedProgram`.

Each kernel thread computes one output element at the coordinates of its flat
index. A value is generated at the coordinates it is read at, once per
coordinate tuple, so index-mapping ops cost index arithmetic, not memory.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Sequence
from typing import Literal

import ml_dtypes
import numpy as np
from jax._src.interpreters import mlir
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

from palladium.emit.core import CTYPES
from palladium.emit.numeric import extremum_identity, format_scalar
from palladium.errors import EmitError

__all__ = ["Launch", "Program", "Slot", "UnsupportedProgram", "compile_program"]


class UnsupportedProgram(Exception):
    """The program uses an op or dtype the Metal compiler does not handle yet."""


@dataclasses.dataclass(frozen=True)
class Slot:
    """A buffer of the program: a parameter, an output it computes, or a constant."""

    kind: Literal["parameter", "fresh", "constant"]
    dtype: np.dtype
    shape: tuple[int, ...]
    data: bytes | None = None


@dataclasses.dataclass(frozen=True)
class Launch:
    """One kernel over `grid` threads, its buffers bound to `slots` in order."""

    source: str
    function: str
    slots: tuple[int, ...]
    grid: int
    threadgroup: int = 0
    math_mode: int = 2  # runtime MathMode: Safe=0, Relaxed=1, Fast=2


@dataclasses.dataclass(frozen=True)
class Program:
    slots: list[Slot]
    launches: list[Launch]
    outputs: list[int]


_DTYPES = {
    "f32": np.dtype(np.float32),
    "f16": np.dtype(np.float16),
    "bf16": np.dtype(ml_dtypes.bfloat16),
    "f64": np.dtype(np.float64),
    "i1": np.dtype(np.bool_),
    "i8": np.dtype(np.int8),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "ui8": np.dtype(np.uint8),
    "ui16": np.dtype(np.uint16),
    "ui32": np.dtype(np.uint32),
    "ui64": np.dtype(np.uint64),
}

# StableHLO op names to the JAX primitive names of palladium's scalar templates.
_ELEMENTWISE = {
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
    "log": "log",
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


def _tensor(value: ir.Value) -> tuple[np.dtype, tuple[int, ...]]:
    t = ir.RankedTensorType(value.type)
    name = str(t.element_type)
    if name not in _DTYPES:
        raise UnsupportedProgram(f"metal: element type {name} is not supported")
    return _DTYPES[name], tuple(t.shape)


def _ctype(dtype: np.dtype) -> str:
    if dtype not in CTYPES:
        raise UnsupportedProgram(f"metal: kernels on {dtype} are not supported")
    return CTYPES[dtype]


def _literal(value: float, ctype: str) -> str:
    if ctype in ("float", "half", "bfloat"):
        if math.isinf(value):
            expr = "-INFINITY" if value < 0 else "INFINITY"
        elif math.isnan(value):
            expr = "NAN"
        else:
            expr = f"{np.float32(value)!s}f"
        return expr if ctype == "float" else f"{ctype}({expr})"
    return f"{ctype}({int(value)})"


# A name or literal, which a value can stand for without a variable of its own.
_ATOM = re.compile(r"-?[\w.]+(\(-?[\w.]+\))?")


def _strides(shape: Sequence[int]) -> list[int]:
    strides, step = [], 1
    for n in reversed(shape):
        strides.append(step)
        step *= n
    return strides[::-1]


def _linear(coords: Sequence[str], shape: Sequence[int]) -> str:
    terms = [
        c if s == 1 else f"{c} * {s}u"
        for c, s, n in zip(coords, _strides(shape), shape, strict=True)
        if n > 1
    ]
    return " + ".join(terms) or "0u"


def _delinear(flat: str, shape: Sequence[int]) -> tuple[str, ...]:
    """Coordinates of a flat row-major index into `shape`."""
    base = flat if _ATOM.fullmatch(flat) else f"({flat})"
    coords = []
    for k, (s, n) in enumerate(zip(_strides(shape), shape, strict=True)):
        quotient = base if s == 1 else f"{base} / {s}u"
        coords.append("0u" if n == 1 else quotient if k == 0 else f"{quotient} % {n}u")
    return tuple(coords)


class _Node:
    dtype: np.dtype
    shape: tuple[int, ...]


@dataclasses.dataclass(eq=False)
class _Parameter(_Node):
    index: int
    dtype: np.dtype
    shape: tuple[int, ...]


@dataclasses.dataclass(eq=False)
class _Constant(_Node):
    dtype: np.dtype
    shape: tuple[int, ...]
    splat: float | None  # None: `array` holds the elements
    array: np.ndarray | None = None


@dataclasses.dataclass(eq=False)
class _Op(_Node):
    name: str  # StableHLO op name without the dialect
    operands: list[_Node]
    dtype: np.dtype
    shape: tuple[int, ...]
    attributes: dict


def _constant(op) -> _Constant:
    dtype, shape = _tensor(op.result)
    attr = ir.DenseElementsAttr(op.attributes["value"])
    if attr.is_splat:
        value = attr.get_splat_value()
        for kind in (ir.BoolAttr, ir.FloatAttr, ir.IntegerAttr):
            if isinstance(value, kind):
                return _Constant(dtype, shape, value.value)
    if dtype == ml_dtypes.bfloat16:
        fp = ir.DenseFPElementsAttr(attr)
        array = np.array([fp[i] for i in range(len(fp))], dtype).reshape(shape)
    else:
        array = np.array(attr, dtype).reshape(shape)
    return _Constant(dtype, shape, None, array)


def _attributes(op) -> dict:
    a = op.attributes
    name = op.operation.name.removeprefix("stablehlo.")
    if name == "reduce":
        block = op.regions[0].blocks[0]
        body = list(block.operations)
        if len(op.results) != 1 or len(body) != 2:
            raise UnsupportedProgram("metal: stablehlo.reduce requires a single binary reducer")
        reducer, ret = body
        kind = reducer.operation.name.removeprefix("stablehlo.")
        if (
            kind not in ("add", "maximum", "minimum", "and", "or")
            or list(reducer.operands) != list(block.arguments)
            or ret.operation.name != "stablehlo.return"
            or list(ret.operands) != list(reducer.results)
        ):
            raise UnsupportedProgram("metal: stablehlo.reduce has an unsupported reducer body")
        return {"dims": list(ir.DenseI64ArrayAttr(a["dimensions"])), "reducer": kind}
    if name == "broadcast_in_dim":
        return {"dims": list(ir.DenseI64ArrayAttr(a["broadcast_dimensions"]))}
    if name == "transpose":
        return {"perm": list(ir.DenseI64ArrayAttr(a["permutation"]))}
    if name == "iota":
        return {"dim": ir.IntegerAttr(a["iota_dimension"]).value}
    if name == "concatenate":
        return {"dim": ir.IntegerAttr(a["dimension"]).value}
    if name == "pad":
        return {
            key: list(ir.DenseI64ArrayAttr(a[attr]))
            for key, attr in (("low", "edge_padding_low"), ("interior", "interior_padding"))
        }
    if name == "slice":
        return {
            "start": list(ir.DenseI64ArrayAttr(a["start_indices"])),
            "strides": list(ir.DenseI64ArrayAttr(a["strides"])),
        }
    if name == "reverse":
        return {"dims": list(ir.DenseI64ArrayAttr(a["dimensions"]))}
    if name == "compare":
        # #stablehlo<comparison_direction LT>
        return {"direction": str(a["comparison_direction"]).split()[-1].rstrip(">").lower()}
    return {}


# Templates palladium's scalar table lacks, by operand ctype class.
_EXTRA = {
    "sign": {
        "float": "(isnan({a}) ? {a} : sign({a}))",
        "int": "int(({a} > 0) - ({a} < 0))",
        "uint": "uint({a} > 0u)",
    },
    "remainder": {
        "float": "fmod({a}, {b})",
        "int": "({b} == 0 ? {a} : ({b} == -1 ? 0 : {a} % {b}))",
        "uint": "({b} == 0u ? {a} : {a} % {b})",
    },
    "round_nearest_even": {"float": "rint({a})"},
    "round_nearest_afz": {"float": "round({a})"},
    "cbrt": {"float": "cbrt({a})"},
    "popcnt": {"int": "popcount({a})", "uint": "popcount({a})"},
    "count_leading_zeros": {"int": "clz({a})", "uint": "clz({a})"},
    "not": {"bool": "!{a}", "int": "~{a}", "uint": "~{a}"},
}

_SUPPORTED = {
    *_ELEMENTWISE,
    *_EXTRA,
    "compare",
    "select",
    "convert",
    "bitcast_convert",
    "broadcast_in_dim",
    "reshape",
    "transpose",
    "iota",
    "slice",
    "reverse",
    "concatenate",
    "pad",
    "dynamic_slice",
    "dynamic_update_slice",
    "reduce",
}


def _graph(module: ir.Module) -> tuple[list[_Parameter], list[_Node]]:
    """The main function as a DAG of nodes, calls and composites inlined."""
    functions = {
        ir.StringAttr(op.attributes["sym_name"]).value: op
        for op in module.body.operations
        if op.operation.name == "func.func"
    }

    def walk(function, args: list[_Node]) -> list[_Node]:
        block = function.regions[0].blocks[0]
        env: dict[ir.Value, _Node] = dict(zip(block.arguments, args, strict=True))
        for op in block.operations:
            name = op.operation.name
            if name == "func.return":
                return [env[v] for v in op.operands]
            if name in ("func.call", "stablehlo.composite"):
                key = "callee" if name == "func.call" else "decomposition"
                callee = functions[ir.FlatSymbolRefAttr(op.attributes[key]).value]
                results = walk(callee, [env[v] for v in op.operands])
                env.update(zip(op.results, results, strict=True))
                continue
            short = name.removeprefix("stablehlo.")
            if name == "stablehlo.constant":
                env[op.result] = _constant(op)
            elif name.startswith("stablehlo.") and short in _SUPPORTED:
                attributes = _attributes(op)
                dtype, shape = _tensor(op.result)
                operands = [env[v] for v in op.operands]
                env[op.result] = _Op(short, operands, dtype, shape, attributes)
            else:
                raise UnsupportedProgram(f"metal: {name} is not supported yet")
        raise UnsupportedProgram("metal: function without a return")

    main = functions["main"]
    block = main.regions[0].blocks[0]
    parameters = [_Parameter(i, *_tensor(v)) for i, v in enumerate(block.arguments)]
    return parameters, walk(main, list(parameters))


class _Kernel:
    """Generates one kernel computing the outputs of one shape."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.lines: list[str] = []
        self.helpers: dict[str, str] = {}
        self.buffers: list[_Node] = []  # read nodes, bound before the outputs
        self.values: dict[tuple[int, tuple[str, ...]], str] = {}
        self.next_value = 0
        self.reshaped: dict[tuple[tuple[str, ...], tuple[int, ...]], str] = {}
        self.coords = tuple(f"c{k}" for k in range(len(shape)))
        self.lines += [
            f"uint {c} = {e};" for c, e in zip(self.coords, _delinear("i", shape), strict=True)
        ]

    def require(self, name: str, source: str) -> None:
        self.helpers[name] = source

    def buffer(self, node: _Node) -> str:
        if node not in self.buffers:
            self.buffers.append(node)
        return f"b{self.buffers.index(node)}"

    def value(self, node: _Node, coords: tuple[str, ...]) -> str:
        key = (id(node), coords)
        if key not in self.values:
            expr = self._expression(node, coords)
            if _ATOM.fullmatch(expr):
                self.values[key] = expr
            else:
                name = self.temporary()
                self.lines.append(f"{_ctype(node.dtype)} {name} = {expr};")
                self.values[key] = name
        return self.values[key]

    def temporary(self) -> str:
        name = f"v{self.next_value}"
        self.next_value += 1
        return name

    def choose(self, dtype: np.dtype, branches: list[tuple[str, _Node, tuple[str, ...]]]) -> str:
        """Emit reads inside their guards; branch-local values must not escape.

        The final branch is unconditional. Parent-scope memoized values remain
        available, but neither values nor reshape shortcuts cross sibling scopes.
        """
        result = self.temporary()
        self.lines.append(f"{_ctype(dtype)} {result};")
        for k, (condition, node, coords) in enumerate(branches):
            last = k == len(branches) - 1
            prefix = "" if k == 0 else "else "
            self.lines.append(prefix + ("{" if last else f"if ({condition}) {{"))
            values, reshaped = self.values.copy(), self.reshaped.copy()
            value = self.value(node, coords)
            self.lines.append(f"{result} = {value};")
            self.values, self.reshaped = values, reshaped
            self.lines.append("}")
        return result

    def offsets(self, starts: list[_Node], limits: tuple[int, ...]) -> tuple[str, ...]:
        offsets = []
        for start, limit in zip(starts, limits, strict=True):
            # Preserve unsigned values above INT_MAX when clamping.
            ctype = _ctype(start.dtype)
            value = self.value(start, ())
            offsets.append(f"uint(clamp({ctype}({value}), {ctype}(0), {ctype}({limit})))")
        return tuple(offsets)

    def _flat(self, coords: tuple[str, ...], shape: tuple[int, ...]) -> str:
        return "i" if coords == self.coords and shape == self.shape else _linear(coords, shape)

    def _read(self, node: _Node, coords: tuple[str, ...]) -> str:
        flat = self.reshaped.get((coords, node.shape)) or self._flat(coords, node.shape)
        return f"{self.buffer(node)}[{flat}]"

    def _expression(self, node: _Node, coords: tuple[str, ...]) -> str:
        ctype = _ctype(node.dtype)
        if isinstance(node, _Parameter):
            return self._read(node, coords)
        if isinstance(node, _Constant):
            if node.splat is None:
                return self._read(node, coords)
            return _literal(node.splat, ctype)
        assert isinstance(node, _Op)
        a, operands = node.attributes, node.operands
        match node.name:
            case "reduce":
                return self._read(node, coords)
            case "concatenate":
                dim, offset, branches = a["dim"], 0, []
                for x in operands:
                    size = x.shape[dim]
                    if size:
                        inner = list(coords)
                        inner[dim] = f"({coords[dim]} - {offset}u)"
                        branches.append((f"{coords[dim]} < {offset + size}u", x, tuple(inner)))
                    offset += size
                # No thread runs for an empty output, but its source must be valid.
                return self.choose(node.dtype, branches) if branches else _literal(0, ctype)
            case "pad":
                x, padding = operands
                if not math.prod(x.shape):
                    return self.value(padding, ())
                inner, conditions = [], []
                for c, n, low, interior in zip(
                    coords, x.shape, a["low"], a["interior"], strict=True
                ):
                    shifted, step = f"(int({c}) - ({low}))", interior + 1
                    conditions += [
                        f"{shifted} >= 0",
                        f"{shifted} / {step} < {n}",
                        f"{shifted} % {step} == 0",
                    ]
                    inner.append(f"uint({shifted} / {step})")
                return self.choose(
                    node.dtype,
                    [
                        (" && ".join(conditions) or "true", x, tuple(inner)),
                        ("", padding, ()),
                    ],
                )
            case "dynamic_slice":
                x, *starts = operands
                limits = tuple(n - s for n, s in zip(x.shape, node.shape, strict=True))
                offsets = self.offsets(starts, limits)
                return self.value(
                    x, tuple(f"({c} + {s})" for c, s in zip(coords, offsets, strict=True))
                )
            case "dynamic_update_slice":
                x, update, *starts = operands
                if not math.prod(update.shape):
                    return self.value(x, coords)
                limits = tuple(n - s for n, s in zip(x.shape, update.shape, strict=True))
                offsets = self.offsets(starts, limits)
                condition = (
                    " && ".join(
                        f"({c} >= {s} && {c} - {s} < {n}u)"
                        for c, s, n in zip(coords, offsets, update.shape, strict=True)
                    )
                    or "true"
                )
                inner = tuple(f"({c} - {s})" for c, s in zip(coords, offsets, strict=True))
                return self.choose(node.dtype, [(condition, update, inner), ("", x, coords)])
            case "broadcast_in_dim":
                [x] = operands
                inner = tuple(
                    "0u" if n == 1 else coords[d] for n, d in zip(x.shape, a["dims"], strict=True)
                )
                return self.value(x, inner)
            case "reshape":
                [x] = operands
                flat = self._flat(coords, node.shape)
                inner = _delinear(flat, x.shape)
                self.reshaped[(inner, x.shape)] = flat  # a read there is at `flat`
                return self.value(x, inner)
            case "transpose":
                [x] = operands
                inner = [""] * len(coords)
                for k, p in enumerate(a["perm"]):
                    inner[p] = coords[k]
                return self.value(x, tuple(inner))
            case "slice":
                [x] = operands
                return self.value(
                    x,
                    tuple(
                        f"({s}u + {c} * {t}u)" if (s, t) != (0, 1) else c
                        for c, s, t in zip(coords, a["start"], a["strides"], strict=True)
                    ),
                )
            case "reverse":
                [x] = operands
                return self.value(
                    x,
                    tuple(
                        f"({n - 1}u - {c})" if k in a["dims"] else c
                        for k, (c, n) in enumerate(zip(coords, node.shape, strict=True))
                    ),
                )
            case "iota":
                return f"{ctype}({coords[a['dim']]})"
        values = [self.value(x, coords) for x in operands]
        match node.name:
            case "convert":
                return f"{ctype}({values[0]})"
            case "bitcast_convert":
                return f"as_type<{ctype}>({values[0]})"
            case "compare":
                return format_scalar(a["direction"], ctype, tuple(values))
            case "select":
                pred, on_true, on_false = values
                return format_scalar("select_n", ctype, (pred, on_false, on_true))
        operand_ctype = _ctype(operands[0].dtype)
        if node.name in _EXTRA:
            kind = "float" if operand_ctype in ("float", "half", "bfloat") else operand_ctype
            if kind not in _EXTRA[node.name]:
                raise UnsupportedProgram(f"metal: stablehlo.{node.name} on {operand_ctype}")
            template = _EXTRA[node.name][kind]
            return template.format(**dict(zip("ab", values, strict=False)))
        try:
            return format_scalar(_ELEMENTWISE[node.name], operand_ctype, tuple(values), self)
        except EmitError as e:
            raise UnsupportedProgram(f"metal: stablehlo.{node.name}: {e}") from e

    def source(self, outputs: list[_Node], name: str) -> str:
        stores = [f"o{k}[i] = {self.value(node, self.coords)};" for k, node in enumerate(outputs)]
        return self.render(outputs, name, stores)

    def render(self, outputs: list[_Node], name: str, stores: list[str], index: str = "i") -> str:
        params = [
            f"device const {_ctype(n.dtype)}* b{k} [[buffer({k})]]"
            for k, n in enumerate(self.buffers)
        ]
        params += [
            f"device {_ctype(n.dtype)}* o{k} [[buffer({len(self.buffers) + k})]]"
            for k, n in enumerate(outputs)
        ]
        params.append(f"uint {index} [[thread_position_in_grid]]")
        body = "\n".join(f"    {line}" for line in self.lines + stores)
        helpers = "".join(f"{s}\n\n" for s in self.helpers.values())
        return (
            "#include <metal_stdlib>\nusing namespace metal;\n\n"
            f"{helpers}kernel void {name}(\n    " + ",\n    ".join(params) + f") {{\n{body}\n}}\n"
        )


def _identity(kind: str, ctype: str) -> str:
    if kind in ("add", "or"):
        return f"{ctype}(0)"
    if kind == "and":
        return "true" if ctype == "bool" else f"{ctype}(~0u)"
    return extremum_identity(_ELEMENTWISE[kind], ctype)


def _reduction_kernel(node: _Op, output: _Node, name: str, chunks: int = 1) -> tuple[_Kernel, str]:
    """128 threads per output; large reductions can write independent partials."""
    x, init = node.operands
    dims = sorted(node.attributes["dims"])
    reduced_shape = tuple(x.shape[d] for d in dims)
    size = math.prod(reduced_shape)
    kind = node.attributes["reducer"]
    ctype = _ctype(node.dtype)
    acc_type = "float" if ctype in ("half", "bfloat") else ctype
    identity = _identity(kind, acc_type)
    intrinsic = {"add": "sum", "maximum": "max", "minimum": "min", "and": "and", "or": "or"}[kind]
    if ctype == "bool":
        intrinsic = "all" if kind in ("and", "minimum") else "any"

    def combine(a: str, b: str) -> str:
        return format_scalar(_ELEMENTWISE[kind], acc_type, (a, b))

    def collective() -> list[str]:
        if acc_type == "float" and kind in ("maximum", "minimum"):
            # Metal's extrema ignore NaNs and do not specify zero ties.
            sign = "" if kind == "minimum" else "!"
            zero, other = ("-0.0f", "0.0f") if kind == "minimum" else ("0.0f", "-0.0f")
            return [
                "{",
                "bool nan = simd_any(isnan(acc));",
                f"bool zero = simd_any(acc == 0.0f && {sign}signbit(acc));",
                f"acc = simd_{intrinsic}(acc);",
                f"acc = nan ? NAN : (acc == 0.0f ? (zero ? {zero} : {other}) : acc);",
                "}",
            ]
        return [f"acc = simd_{intrinsic}(acc);"]

    kernel = _Kernel(output.shape)
    kernel.lines.insert(0, "uint i = g / 128u;")
    kernel.lines += [f"{acc_type} acc = {identity};", "uint t = g % 128u;"]
    start = f"{kernel.coords[-1]} * 4096u + t" if chunks > 1 else "t"
    end = f"min(({kernel.coords[-1]} + 1u) * 4096u, {size}u)" if chunks > 1 else f"{size}u"
    # Zero-sized inputs have no read at all (nor zero divisors in coordinates).
    if size:
        kernel.lines.append(f"for (uint r = {start}; r < {end}; r += 128u) {{")
        red_coords = iter(_delinear("r", reduced_shape))
        out_coords = iter(kernel.coords)
        coords = tuple(
            next(red_coords) if d in dims else next(out_coords) for d in range(len(x.shape))
        )
        value = kernel.value(x, coords)
        kernel.lines += [f"acc = {combine('acc', f'{acc_type}({value})')};", "}"]
        kernel.values.clear()
        kernel.reshaped.clear()
    kernel.lines += collective() + [
        f"threadgroup {acc_type} partial[4];",
        "if (t % 32u == 0u) partial[t / 32u] = acc;",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "if (t < 32u) {",
        f"acc = t < 4u ? partial[t] : {identity};",
    ]
    kernel.lines += collective() + ["if (t == 0u) {"]
    if chunks == 1:
        initial = kernel.value(init, ())
        kernel.lines.append(f"acc = {combine('acc', f'{acc_type}({initial})')};")
    kernel.lines += [f"o0[i] = {_ctype(output.dtype)}(acc);", "}", "}"]
    return kernel, kernel.render([output], name, [], index="g")


def compile_program(code: bytes) -> Program:
    """Compile a serialized StableHLO portable artifact.

    Parameters
    ----------
    code : bytes
        The program as PJRT_Client_Compile receives it.

    Returns
    -------
    Program
        Slots, launches in order, and the output slots.

    Raises
    ------
    UnsupportedProgram
        An op or dtype is not supported yet.
    """
    with mlir.make_ir_context() as ctx:
        module = stablehlo.deserialize_portable_artifact(ctx, code)
        parameters, results = _graph(module)

    slots = [Slot("parameter", p.dtype, p.shape) for p in parameters]
    slot_of: dict[int, int] = {id(p): k for k, p in enumerate(parameters)}

    def add_constant(node: _Node) -> None:
        assert isinstance(node, _Constant) and node.array is not None
        slot_of[id(node)] = len(slots)
        slots.append(Slot("constant", node.dtype, node.shape, node.array.tobytes()))

    def fresh(node: _Node) -> None:
        _ctype(node.dtype)
        slot_of[id(node)] = len(slots)
        slots.append(Slot("fresh", node.dtype, node.shape))

    launches = []

    def launch(
        kernel: _Kernel, source: str, name: str, nodes: list[_Node], threads: int = 0
    ) -> None:
        for operand in kernel.buffers:
            if id(operand) not in slot_of:
                add_constant(operand)
        bound = [slot_of[id(n)] for n in kernel.buffers] + [slot_of[id(n)] for n in nodes]
        grid = math.prod(nodes[0].shape) * (threads or 1)
        launches.append(Launch(source, name, tuple(bound), grid, threads, 0 if threads else 2))

    visited: set[int] = set()
    temporaries: list[_Node] = []  # keep synthetic nodes alive while slot_of uses their ids

    def schedule(node: _Node) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        if not isinstance(node, _Op):
            return
        for operand in node.operands:
            schedule(operand)
        if node.name != "reduce":
            return
        x, init = node.operands
        size = math.prod(x.shape[d] for d in node.attributes["dims"])
        # Split long, low-output-count reductions so full reductions occupy the GPU.
        chunks = math.ceil(size / 4096) if size > 4096 and math.prod(node.shape) < 128 else 1
        reduction = node
        if chunks > 1:
            dtype = (
                np.dtype(np.float32)
                if node.dtype in (np.float16, ml_dtypes.bfloat16)
                else node.dtype
            )
            partial = _Parameter(-1, dtype, (*node.shape, chunks))
            temporaries.append(partial)
            fresh(partial)
            name = f"reduce_{len(launches)}"
            kernel, source = _reduction_kernel(node, partial, name, chunks)
            launch(kernel, source, name, [partial], 128)
            reduction = _Op(
                "reduce",
                [partial, init],
                node.dtype,
                node.shape,
                {"dims": [len(node.shape)], "reducer": node.attributes["reducer"]},
            )
        fresh(node)
        name = f"reduce_{len(launches)}"
        kernel, source = _reduction_kernel(reduction, node, name)
        launch(kernel, source, name, [node], 128)

    for node in results:
        schedule(node)

    computed: dict[tuple[int, ...], list[_Node]] = {}
    for node in results:
        if id(node) in slot_of:
            continue
        if isinstance(node, _Constant) and node.splat is None:
            add_constant(node)
            continue
        fresh(node)
        computed.setdefault(node.shape, []).append(node)

    for shape, nodes in computed.items():
        kernel = _Kernel(shape)
        name = f"fused_{len(launches)}"
        source = kernel.source(nodes, name)
        launch(kernel, source, name, nodes)
    return Program(slots, launches, [slot_of[id(n)] for n in results])
