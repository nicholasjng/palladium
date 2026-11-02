"""StableHLO programs to Metal kernels, for the `metal` PJRT plugin.

The plugin's PJRT_Client_Compile passes the portable artifact jax serializes to
`compile_program`, which parses it with jaxlib's MLIR bindings and returns a
`Program`: buffer slots and kernel launches, which `palladium._pjrt` hands to
the plugin. A program may hold elementwise and index-mapping ops
(broadcast, reshape, transpose, slices, concatenate, pad, reverse, iota), fused
into one kernel per output shape. Single-result reductions end a fusion and
materialize a buffer for subsequent kernels. Floating dots reuse Palladium's
Pallas GEMM emitter after layout normalization. Palladium kernels
(`pallas_call` lowered for `metal`, `metal_call`, `plmtl`) arrive as
`palladium_dispatch` custom calls and launch their own MSL. Kernels bind at
most 31 buffers, Metal's limit. Unsupported ops and reducer bodies raise
`UnsupportedProgram`.

Each kernel thread computes one output element at the coordinates of its flat
index. A value is generated at the coordinates it is read at, once per
coordinate tuple, so index-mapping ops cost index arithmetic, not memory.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import math
import re
from collections.abc import Callable, Sequence
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
    """One kernel over `grid` threads, its buffers bound to `slots` in order.

    `offsets`, when given, are the byte offsets of each binding (default 0).
    """

    source: str
    function: str
    slots: tuple[int, ...]
    grid: int | tuple[int, int, int]
    threadgroup: int | tuple[int, int, int] = 0  # threads, or 0: the pipeline chooses
    math_mode: int = 2  # runtime MathMode: Safe=0, Relaxed=1, Fast=2
    offsets: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class Program:
    slots: list[Slot]
    launches: list[Launch]
    outputs: list[int]
    aliases: tuple[tuple[int, int], ...] = ()  # parameter, output


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
    # PJRT-only types: do not broaden the shared Pallas emitter's type contract.
    if dtype == np.int64:
        return "long"
    if dtype == np.uint64:
        return "ulong"
    if dtype not in CTYPES:
        raise UnsupportedProgram(f"metal: kernels on {dtype} are not supported")
    return CTYPES[dtype]


def _literal(value: float, ctype: str) -> str:
    if ctype == "ulong":
        return f"{int(value) % (1 << 64)}ul"
    if ctype == "long":
        return "(-9223372036854775807l - 1l)" if int(value) == -(1 << 63) else f"{int(value)}l"
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


def _index(expr: str) -> int | None:
    """The value of a constant unsigned index expression (+, -, * only), else None."""
    try:
        tree = ast.parse(re.sub(r"\b(\d+)u\b", r"\1", expr), mode="eval").body
    except SyntaxError:
        return None

    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            a, b = evaluate(node.left), evaluate(node.right)
            return (
                a + b
                if isinstance(node.op, ast.Add)
                else a - b
                if isinstance(node.op, ast.Sub)
                else a * b
            )
        raise ValueError

    try:
        value = evaluate(tree)
    except ValueError:
        return None
    return value if 0 <= value < 1 << 32 else None


def _fold(expr: str) -> str:
    """A constant index expression as a literal; other expressions unchanged."""
    value = _index(expr)
    return expr if value is None else f"{value}u"


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


def _paren(expr: str) -> str:
    """`expr` as one operand: unchanged if an atom or already fully parenthesized."""
    if _ATOM.fullmatch(expr):
        return expr
    if expr.startswith("(") and expr.endswith(")"):
        depth = 0
        for k, ch in enumerate(expr):
            depth += {"(": 1, ")": -1}.get(ch, 0)
            if depth == 0 and k < len(expr) - 1:
                break
        else:
            return expr
    return f"({expr})"


def _delinear(flat: str, shape: Sequence[int]) -> tuple[str, ...]:
    """Coordinates of a flat row-major index into `shape`.

    One spelling per coordinate, so equal reads memoize together: every
    non-atom is parenthesized once, and the leading non-unit dimension needs
    no modulus (the flat index is below the size).
    """
    base = _paren(flat)
    coords = []
    leading = True
    for s, n in zip(_strides(shape), shape, strict=True):
        if n == 1:
            coords.append("0u")
            continue
        quotient = base if s == 1 else f"({base} / {s}u)"
        coords.append(quotient if leading else f"({quotient} % {n}u)")
        leading = False
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


@dataclasses.dataclass(eq=False)
class _While:
    inputs: list[_Node]  # carries followed by lexical captures
    parameters: list[_Parameter]
    results: list[_Node]  # updated carries
    trips: int
    counter: int  # carry index of the proven counter
    start: int  # its initial value


@dataclasses.dataclass(eq=False)
class _LoopResult(_Node):
    loop: _While
    index: int
    dtype: np.dtype
    shape: tuple[int, ...]


@dataclasses.dataclass(eq=False)
class _VariadicReduce:
    inputs: list[_Node]
    initial: list[_Node]
    parameters: list[_Parameter]
    body: list[_Node]
    dims: tuple[int, ...]


@dataclasses.dataclass(eq=False)
class _ReductionResult(_Node):
    reduction: _VariadicReduce
    index: int
    dtype: np.dtype
    shape: tuple[int, ...]


@dataclasses.dataclass(eq=False)
class _Dispatch:
    """A palladium kernel with its own launch: a `palladium_dispatch` custom call.

    `pallas_call` (lowered for `metal`) and `metal_call` both emit this call;
    its typed-FFI attributes carry the MSL and the launch geometry. Buffers
    bind inputs then results, in order. `aliases` maps a result to the input
    whose values it starts with.
    """

    inputs: list[_Node]
    results: list[_DispatchResult]
    source: str
    function: str
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]  # zeros: the pipeline chooses
    math_mode: int
    batch: int  # launches, each offset by `strides` bytes per buffer
    strides: tuple[int, ...]
    aliases: dict[int, int]


@dataclasses.dataclass(eq=False)
class _DispatchResult(_Node):
    dispatch: _Dispatch
    index: int
    dtype: np.dtype
    shape: tuple[int, ...]


def _dependencies(node: _Node) -> list[_Node]:
    if isinstance(node, _Op):
        return node.operands
    if isinstance(node, _LoopResult):
        return node.loop.inputs
    if isinstance(node, _ReductionResult):
        return node.reduction.inputs + node.reduction.initial
    if isinstance(node, _DispatchResult):
        return node.dispatch.inputs
    return []


def _trip_count(
    predicate: _Node, parameters: list[_Parameter], inputs: list[_Node], results: list[_Node]
) -> tuple[int, int, int]:
    """Prove a scalar integer counter starts at a constant and advances to a bound.

    Returns the trip count, the counter's carry index and its start value.
    """
    message = "metal: stablehlo.while requires a constant counter, LT bound and unit increment"

    def invariant(value: _Node) -> _Node:
        if isinstance(value, _Parameter) and value in parameters:
            k = parameters.index(value)
            if k >= len(results) or results[k] is value:
                return inputs[k]
        return value

    if (
        not isinstance(predicate, _Op)
        or predicate.name != "compare"
        or predicate.attributes["direction"] != "lt"
    ):
        raise UnsupportedProgram(message)
    counter, bound = predicate.operands
    if not isinstance(counter, _Parameter) or counter not in parameters[: len(results)]:
        raise UnsupportedProgram(message)
    index = parameters.index(counter)
    if counter.shape or counter.dtype not in (np.int32, np.uint32):
        raise UnsupportedProgram(message)
    increment = results[index]
    if not isinstance(increment, _Op) or increment.name != "add":
        raise UnsupportedProgram(message)
    a, b = increment.operands
    one = b if a is counter else a if b is counter else None
    if one is not None:
        one = invariant(one)
    if not isinstance(one, _Constant) or one.splat != 1:
        raise UnsupportedProgram(message)
    # An invariant carry or lexical capture can supply a constant loop bound.
    bound = invariant(bound)
    start = inputs[index]
    if (
        not isinstance(start, _Constant)
        or start.splat is None
        or not isinstance(bound, _Constant)
        or bound.splat is None
    ):
        raise UnsupportedProgram(message)
    return max(0, int(bound.splat) - int(start.splat)), index, int(start.splat)


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
    if name == "scatter":
        block = op.regions[0].blocks[0]
        body = list(block.operations)
        if (
            len(op.results) != 1
            or len(body) != 2
            or body[0].operation.name != "stablehlo.add"
            or list(body[0].operands) != list(block.arguments)
            or body[1].operation.name != "stablehlo.return"
            or list(body[1].operands) != list(body[0].results)
        ):
            raise UnsupportedProgram("metal: stablehlo.scatter requires a single add update")
        d = stablehlo.ScatterDimensionNumbers(a["scatter_dimension_numbers"])
        return {
            "window": tuple(d.update_window_dims),
            "inserted": tuple(d.inserted_window_dims),
            "map": tuple(d.scattered_dims_to_operand_dims),
            "vector": d.index_vector_dim,
            "batch": tuple(d.input_batching_dims),
            "index_batch": tuple(d.scatter_indices_batching_dims),
        }
    if name == "gather":
        d = stablehlo.GatherDimensionNumbers(a["dimension_numbers"])
        return {
            "offset": tuple(d.offset_dims),
            "collapsed": tuple(d.collapsed_slice_dims),
            "map": tuple(d.start_index_map),
            "vector": d.index_vector_dim,
            "batch": tuple(d.operand_batching_dims),
            "index_batch": tuple(d.start_indices_batching_dims),
            "sizes": tuple(ir.DenseI64ArrayAttr(a["slice_sizes"])),
        }
    if name == "dot_general":
        if "algorithm" in a:
            raise UnsupportedProgram(
                "metal: stablehlo.dot_general explicit algorithms are not supported"
            )
        dims = stablehlo.DotDimensionNumbers(a["dot_dimension_numbers"])
        return {
            "lhs_batch": tuple(dims.lhs_batching_dimensions),
            "rhs_batch": tuple(dims.rhs_batching_dimensions),
            "lhs_contract": tuple(dims.lhs_contracting_dimensions),
            "rhs_contract": tuple(dims.rhs_contracting_dimensions),
        }
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


_DISPATCH_TARGET = "palladium_dispatch"
# Reductions over at most this many elements are folded into their consumer's
# kernel, unrolled per output element, rather than given 128 threads per output.
_SHORT_REDUCTION = 16


def _short(node: _Op) -> bool:
    x = node.operands[0]
    size = math.prod(x.shape[d] for d in node.attributes["dims"])
    return node.dtype not in (np.int64, np.uint64) and size <= _SHORT_REDUCTION


# Buffer arguments one Metal kernel can bind.
_MAX_BUFFERS = 31


def _row_major(attributes, name: str, values) -> bool:
    """Whether the layouts attribute `name`, if any, is row-major for `values`."""
    if name not in attributes:
        return True
    return all(
        [int(d) for d in ir.DenseIntElementsAttr(layout)]
        == list(range(len(_tensor(v)[1]) - 1, -1, -1))
        for layout, v in zip(ir.ArrayAttr(attributes[name]), values, strict=True)
    )


def _dispatch(op, inputs: list[_Node]) -> _Dispatch:
    """A `palladium_dispatch` custom call as a launch description."""
    a = op.attributes
    target = ir.StringAttr(a["call_target_name"]).value
    if target != _DISPATCH_TARGET:
        raise UnsupportedProgram(f"metal: custom call {target!r} is not supported")
    if "mhlo.backend_config" not in a:
        raise UnsupportedProgram(f"metal: {target} requires typed-FFI attributes")
    if not _row_major(a, "operand_layouts", op.operands) or not _row_major(
        a, "result_layouts", op.results
    ):
        raise UnsupportedProgram(f"metal: {target} requires row-major layouts")
    config = ir.DictAttr(a["mhlo.backend_config"])

    def integer(name: str) -> int:
        return ir.IntegerAttr(config[name]).value

    aliases = {}
    if "output_operand_aliases" in a:
        for attr in ir.ArrayAttr(a["output_operand_aliases"]):
            alias = stablehlo.OutputOperandAlias(attr)
            output = list(alias.output_tuple_indices)
            if list(alias.operand_tuple_indices) or len(output) > 1:
                raise UnsupportedProgram(f"metal: {target} has a nested alias")
            aliases[output[0] if output else 0] = alias.operand_index
    strides = tuple(int(s) for s in ir.DenseIntElementsAttr(config["elem_strides"]))
    if len(strides) != len(op.operands) + len(op.results):
        raise UnsupportedProgram(f"metal: {target} needs one stride per buffer")
    dispatch = _Dispatch(
        inputs,
        [],
        ir.StringAttr(config["msl_source"]).value,
        ir.StringAttr(config["function_name"]).value,
        (integer("grid_x"), integer("grid_y"), integer("grid_z")),
        (integer("threadgroup_x"), integer("threadgroup_y"), integer("threadgroup_z")),
        integer("math_mode"),
        integer("batch_size"),
        strides,
        aliases,
    )
    dispatch.results = [
        _DispatchResult(dispatch, k, *_tensor(value)) for k, value in enumerate(op.results)
    ]
    for k, value in enumerate(dispatch.results):
        if k in aliases:
            source = inputs[aliases[k]]
            if (source.dtype, source.shape) != (value.dtype, value.shape):
                raise UnsupportedProgram(f"metal: {target} aliases mismatched buffers")
    return dispatch


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
    "dot_general",
    "gather",
    "scatter",
}


def _wide_scalar(name: str, ctype: str, values: list[str]) -> str:
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


def _graph(module: ir.Module) -> tuple[list[_Parameter], list[_Node]]:
    """The main function as a DAG of nodes, calls and composites inlined."""
    functions = {
        ir.StringAttr(op.attributes["sym_name"]).value: op
        for op in module.body.operations
        if op.operation.name == "func.func"
    }

    def walk(block, args: list[_Node], outer: dict | None = None) -> list[_Node]:
        env: dict[ir.Value, _Node] = dict(outer or {})
        env.update(zip(block.arguments, args, strict=True))
        for op in block.operations:
            name = op.operation.name
            if name in ("func.return", "stablehlo.return"):
                return [env[v] for v in op.operands]
            if name in ("func.call", "stablehlo.composite"):
                key = "callee" if name == "func.call" else "decomposition"
                callee = functions[ir.FlatSymbolRefAttr(op.attributes[key]).value]
                results = walk(callee.regions[0].blocks[0], [env[v] for v in op.operands])
                env.update(zip(op.results, results, strict=True))
                continue
            if name == "stablehlo.case":
                # Regions have no arguments and capture their enclosing values.
                # Parse every branch: unsupported/effectful operations must still
                # fail explicitly, even when a constant selector would skip them.
                branches = [walk(region.blocks[0], [], env) for region in op.regions]
                index = env[op.operands[0]]
                for k, value in enumerate(op.results):
                    env[value] = _Op(
                        "case", [index, *(b[k] for b in branches)], *_tensor(value), {}
                    )
                continue
            if name == "stablehlo.while":
                inputs = [env[v] for v in op.operands]
                parameters = [_Parameter(i, *_tensor(v)) for i, v in enumerate(op.operands)]
                # Replace lexical values by parameters, then keep only used captures.
                captures = {
                    v: n
                    if isinstance(n, _Constant)
                    else _Parameter(len(parameters) + i, n.dtype, n.shape)
                    for i, (v, n) in enumerate(env.items())
                }
                condition = walk(op.regions[0].blocks[0], list(parameters), captures)[0]
                body = walk(op.regions[1].blocks[0], list(parameters), captures)
                used: set[_Node] = set()
                pending = [condition, *body]
                while pending:
                    node = pending.pop()
                    if node not in used:
                        used.add(node)
                        pending.extend(_dependencies(node))
                for value, parameter in captures.items():
                    if isinstance(parameter, _Parameter) and parameter in used:
                        parameters.append(parameter)
                        inputs.append(env[value])
                trips, counter, start = _trip_count(condition, parameters, inputs, body)
                loop = _While(inputs, parameters, body, trips, counter, start)
                for i, value in enumerate(op.results):
                    env[value] = _LoopResult(loop, i, *_tensor(value))
                continue
            if name == "stablehlo.reduce" and len(op.results) > 1:
                count = len(op.results)
                block = op.regions[0].blocks[0]
                params = [_Parameter(i, *_tensor(v)) for i, v in enumerate(block.arguments)]
                body = walk(block, list(params), env)
                pending, seen = list(body), set()
                allowed = {
                    *_ELEMENTWISE,
                    *_EXTRA,
                    "compare",
                    "select",
                    "convert",
                    "bitcast_convert",
                }
                while pending:
                    value = pending.pop()
                    if value in seen:
                        continue
                    seen.add(value)
                    if (
                        value.shape
                        or (isinstance(value, _Parameter) and value not in params)
                        or (
                            not isinstance(value, (_Constant, _Parameter))
                            and (not isinstance(value, _Op) or value.name not in allowed)
                        )
                    ):
                        raise UnsupportedProgram(
                            "metal: variadic reduce requires a scalar elementwise body"
                        )
                    pending.extend(_dependencies(value))
                inputs = [env[v] for v in op.operands]
                reduction = _VariadicReduce(
                    inputs[:count],
                    inputs[count:],
                    params,
                    body,
                    tuple(ir.DenseI64ArrayAttr(op.attributes["dimensions"])),
                )
                for k, value in enumerate(op.results):
                    env[value] = _ReductionResult(reduction, k, *_tensor(value))
                continue
            if name == "stablehlo.custom_call":
                dispatch = _dispatch(op, [env[v] for v in op.operands])
                env.update(zip(op.results, dispatch.results, strict=True))
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
    return parameters, walk(block, list(parameters))


# Identifiers in generated expressions, for deciding where a value is defined.
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


@dataclasses.dataclass
class _Scope:
    """Memoized values of the kernel body or of one guarded branch.

    `names` are the variables the scope defines (temporaries and shifted
    coordinates). `insert` is the line index, in the enclosing scope, just
    before the branch chain: values hoisted out of this branch go there.
    """

    insert: int = -1
    hoist: bool = True  # may values leave this branch?
    names: set[str] = dataclasses.field(default_factory=set)
    values: dict[tuple[int, tuple[str, ...]], str] = dataclasses.field(default_factory=dict)
    reshaped: dict[tuple[tuple[str, ...], tuple[int, ...]], str] = dataclasses.field(
        default_factory=dict
    )


class _Kernel:
    """Generates one kernel computing the outputs of one shape."""

    def __init__(
        self,
        shape: tuple[int, ...],
        stored: set[int] | frozenset[int] = frozenset(),
        store: Callable[[_Node], None] | None = None,
    ):
        self.shape = shape
        self.stored = stored  # ids of ops already in buffers: read, not recomputed
        # Puts an op in a buffer (adding it to `stored`) before this kernel runs.
        self.store = store
        # Needs RELAXED math, not FAST: folds a float maximum or minimum.
        self.relaxed = False
        self.lines: list[str] = []
        self.helpers: dict[str, str] = {}
        self.buffers: list[_Node] = []  # read nodes, bound before the outputs
        self.scopes = [_Scope()]  # the body, then open guarded branches
        self.next_value = 0
        self.coords = tuple(f"c{k}" for k in range(len(shape)))
        # Expressions the coordinate variables stand for, so reads spelled
        # either way memoize together.
        self.named = dict(zip(_delinear("i", shape), self.coords, strict=True))
        self.lines += [
            f"uint {c} = {e};" for c, e in zip(self.coords, _delinear("i", shape), strict=True)
        ]

    def coords_of(self, shape: tuple[int, ...]) -> tuple[str, ...]:
        """Coordinates in `shape` of this thread's flat index.

        Expressions the kernel already names (its own coordinates) use those
        names, so a value read through another shape of equal size, such as
        an added unit dimension, hits the same memoized values.
        """
        if shape == self.shape:
            return self.coords
        return tuple(self.named.get(e, e) for e in _delinear("i", shape))

    def require(self, name: str, source: str) -> None:
        self.helpers[name] = source

    def buffer(self, node: _Node) -> str:
        if node not in self.buffers:
            self.buffers.append(node)
        return f"b{self.buffers.index(node)}"

    @property
    def values(self) -> dict[tuple[int, tuple[str, ...]], str]:
        """Values memoized in the kernel body (outside any guarded branch)."""
        return self.scopes[0].values

    @property
    def reshaped(self) -> dict[tuple[tuple[str, ...], tuple[int, ...]], str]:
        return self.scopes[0].reshaped

    def _level(self, coords: tuple[str, ...]) -> int:
        """The outermost open scope where a value at `coords` may be computed.

        Coordinates that use none of a branch's own names are computed from
        enclosing coordinates by index maps, which stay in bounds without the
        branch's guard; such a value can be computed once before the chain.
        """
        used = {name for c in coords for name in _IDENTIFIER.findall(c)}
        for depth in range(len(self.scopes) - 1, 0, -1):
            scope = self.scopes[depth]
            if not scope.hoist or used & scope.names:
                return depth
        return 0

    def value(self, node: _Node, coords: tuple[str, ...]) -> str:
        coords = tuple(self.named.get(c, c) for c in coords)
        key = (id(node), coords)
        for scope in reversed(self.scopes):
            if key in scope.values:
                return scope.values[key]
        level = self._level(coords)
        if level < len(self.scopes) - 1:
            # Generate it as if at `level`, then insert before the branch chain.
            inner, lines = self.scopes[level + 1 :], self.lines
            self.scopes, self.lines = self.scopes[: level + 1], []
            try:
                result = self.value(node, coords)
            finally:
                hoisted, self.lines = self.lines, lines
                self.scopes += inner
            index = inner[0].insert
            self.lines[index:index] = hoisted
            for scope in inner:
                scope.insert += len(hoisted)
            return result
        scope = self.scopes[-1]
        expr = self._expression(node, coords)
        if _ATOM.fullmatch(expr):
            scope.values[key] = expr
        else:
            name = self.temporary()
            self.lines.append(f"{_ctype(node.dtype)} {name} = {expr};")
            scope.values[key] = name
        return scope.values[key]

    def temporary(self) -> str:
        name = f"v{self.next_value}"
        self.next_value += 1
        self.scopes[-1].names.add(name)
        return name

    def choose(
        self,
        dtype: np.dtype,
        branches: list[tuple[str, _Node, tuple[str, ...]]],
        hoist: bool = True,
        parent: tuple[str, ...] = (),
    ) -> str:
        """Emit each branch's reads inside its guard; the final branch is unconditional.

        A branch's shifted coordinates (those differing from `parent`) and its
        temporaries are its own names, so its values never escape to siblings.
        Callers guarantee that unshifted coordinates index the branch operand
        within bounds. With `hoist`, a value that uses no branch name is
        computed once before the chain instead of in every branch. Without it
        (`case`), branches stay lazy.
        """
        result = self.temporary()
        self.lines.append(f"{_ctype(dtype)} {result};")
        insert = len(self.lines)
        for k, (condition, node, coords) in enumerate(branches):
            last = k == len(branches) - 1
            prefix = "" if k == 0 else "else "
            self.lines.append(prefix + ("{" if last else f"if ({condition}) {{"))
            scope = _Scope(insert, hoist)
            self.scopes.append(scope)
            try:
                bound = []
                for j, c in enumerate(coords):
                    if _ATOM.fullmatch(c) or (j < len(parent) and c == parent[j]):
                        bound.append(c)
                    else:
                        name = f"k{self.next_value}"
                        self.next_value += 1
                        scope.names.add(name)
                        self.lines.append(f"uint {name} = {c};")
                        bound.append(name)
                value = self.value(node, tuple(bound))
                self.lines.append(f"{result} = {value};")
            finally:
                self.scopes.pop()
            insert = scope.insert
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
        flat = next(
            (
                s.reshaped[(coords, node.shape)]
                for s in reversed(self.scopes)
                if (coords, node.shape) in s.reshaped
            ),
            None,
        ) or self._flat(coords, node.shape)
        return f"{self.buffer(node)}[{flat}]"

    def _expression(self, node: _Node, coords: tuple[str, ...]) -> str:
        ctype = _ctype(node.dtype)
        if id(node) in self.stored or isinstance(
            node, (_Parameter, _LoopResult, _ReductionResult, _DispatchResult)
        ):
            return self._read(node, coords)
        if isinstance(node, _Constant):
            if node.splat is None:
                return self._read(node, coords)
            return _literal(node.splat, ctype)
        assert isinstance(node, _Op)
        a, operands = node.attributes, node.operands
        match node.name:
            case "gather":
                x, indices = operands
                batch_coords = iter(c for d, c in enumerate(coords) if d not in a["offset"])
                index_coords = tuple(
                    "0u" if d == a["vector"] else next(batch_coords)
                    for d in range(len(indices.shape))
                )
                inner = ["0u"] * len(x.shape)
                for component, dim in enumerate(a["map"]):
                    read_coords = list(index_coords)
                    if a["vector"] < len(indices.shape):
                        read_coords[a["vector"]] = f"{component}u"
                    start = self.value(indices, tuple(read_coords))
                    itype = _ctype(indices.dtype)
                    inner[dim] = (
                        f"uint(clamp({start}, {itype}(0), {itype}({x.shape[dim] - a['sizes'][dim]})))"
                    )
                for dim, index_dim in zip(a["batch"], a["index_batch"], strict=True):
                    inner[dim] = index_coords[index_dim]
                window_dims = [
                    d for d in range(len(x.shape)) if d not in a["collapsed"] + a["batch"]
                ]
                for dim, offset in zip(window_dims, a["offset"], strict=True):
                    inner[dim] = f"({inner[dim]} + {coords[offset]})"
                return self.value(x, tuple(inner))
            case "reduce" if _short(node):
                if len(self.scopes) == 1 or self.store is None:
                    return self._fold_reduction(node, coords)
                # Branches name their shifted coordinates afresh, so a fold
                # at branch coordinates would be regenerated, producers and
                # all, in every branch that reads it: store it once instead.
                self.store(node)
                return self._read(node, coords)
            case "reduce" | "dot_general" | "scatter":
                return self._read(node, coords)
            case "case":
                index, *branches = operands
                selector = self.value(index, ())
                # StableHLO sends any out-of-range index to the last branch.
                # JAX switch inserts its own clamp before this operation.
                # Untaken branches stay unevaluated: no hoisting across them.
                return self.choose(
                    node.dtype,
                    [(f"{selector} == {k}", branch, coords) for k, branch in enumerate(branches)],
                    hoist=False,
                )
            case "concatenate":
                dim, offset, branches = a["dim"], 0, []
                position = _index(coords[dim])
                if position is not None:
                    # A constant coordinate selects its operand statically.
                    for x in operands:
                        if position < offset + x.shape[dim]:
                            inner = list(coords)
                            inner[dim] = f"{position - offset}u"
                            return self.value(x, tuple(inner))
                        offset += x.shape[dim]
                for x in operands:
                    size = x.shape[dim]
                    if size:
                        inner = list(coords)
                        inner[dim] = f"({coords[dim]} - {offset}u)"
                        branches.append((f"{coords[dim]} < {offset + size}u", x, tuple(inner)))
                    offset += size
                # No thread runs for an empty output, but its source must be valid.
                # Operands match the output in every other dimension.
                if not branches:
                    return _literal(0, ctype)
                return self.choose(node.dtype, branches, parent=coords)
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
                # The operand has the output's shape: its unshifted reads are in bounds.
                return self.choose(
                    node.dtype, [(condition, update, inner), ("", x, coords)], parent=coords
                )
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
                self.scopes[-1].reshaped[(inner, x.shape)] = flat  # a read there is at `flat`
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
                        _fold(f"({s}u + {c} * {t}u)") if (s, t) != (0, 1) else c
                        for c, s, t in zip(coords, a["start"], a["strides"], strict=True)
                    ),
                )
            case "reverse":
                [x] = operands
                return self.value(
                    x,
                    tuple(
                        _fold(f"({n - 1}u - {c})") if k in a["dims"] else c
                        for k, (c, n) in enumerate(zip(coords, node.shape, strict=True))
                    ),
                )
            case "iota":
                return f"{ctype}({coords[a['dim']]})"
        if node.name == "bitcast_convert" and node.dtype.itemsize != operands[0].dtype.itemsize:
            raise UnsupportedProgram(
                "metal: shape-changing stablehlo.bitcast_convert is not supported"
            )
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
        if operand_ctype in ("long", "ulong"):
            return _wide_scalar(node.name, operand_ctype, values)
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

    def _fold_reduction(self, node: _Op, coords: tuple[str, ...]) -> str:
        """A short reduction as a left fold from its initial value, unrolled.

        Like XLA's sequential reduce. Each step gets a temporary: the extremum
        templates repeat their operands. Under FAST math, producers such as
        `x * inf - inf` lose their NaNs and extrema their zero signs; RELAXED
        keeps both (measured on the M2) at FAST's speed, unlike SAFE.
        """
        x, init = node.operands
        dims = sorted(node.attributes["dims"])
        kind = node.attributes["reducer"]
        ctype = _ctype(node.dtype)
        acc_type = "float" if ctype in ("half", "bfloat") else ctype
        if acc_type == "float" and kind in ("maximum", "minimum"):
            self.relaxed = True
        acc = f"{acc_type}({self.value(init, ())})"
        for index in itertools.product(*(range(x.shape[d]) for d in dims)):
            fixed, outer = iter(index), iter(coords)
            inner = tuple(
                f"{next(fixed)}u" if d in dims else next(outer) for d in range(len(x.shape))
            )
            value = f"{acc_type}({self.value(x, inner)})"
            name = self.temporary()
            step = format_scalar(_ELEMENTWISE[kind], acc_type, (acc, value))
            self.lines.append(f"{acc_type} {name} = {step};")
            acc = name
        return f"{ctype}({acc})"

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


def _reduction_kernel(
    node: _Op,
    output: _Node,
    name: str,
    chunks: int = 1,
    new_kernel: Callable[[tuple[int, ...]], _Kernel] = _Kernel,
) -> tuple[_Kernel, str]:
    """128 threads per output; large reductions can write independent partials."""
    x, init = node.operands
    if node.dtype in (np.int64, np.uint64):
        raise UnsupportedProgram("metal: stablehlo.reduce on 64-bit integers is not supported")
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

    kernel = new_kernel(output.shape)
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
        main = next(
            op
            for op in module.body.operations
            if op.operation.name == "func.func"
            and ir.StringAttr(op.attributes["sym_name"]).value == "main"
        )
        aliases = []
        if "arg_attrs" in main.attributes:
            for i, attrs in enumerate(ir.ArrayAttr(main.attributes["arg_attrs"])):
                attrs = ir.DictAttr(attrs)
                if "tf.aliasing_output" in attrs:
                    output = ir.IntegerAttr(attrs["tf.aliasing_output"]).value
                    if (
                        not 0 <= output < len(results)
                        or parameters[i].dtype != results[output].dtype
                        or parameters[i].shape != results[output].shape
                        or output in [o for _, o in aliases]
                    ):
                        raise UnsupportedProgram("metal: incompatible input/output alias")
                    aliases.append((i, output))

    return dataclasses.replace(_compile_graph(parameters, results), aliases=tuple(aliases))


def _inductions(loop: _While) -> dict[int, tuple[int, int]]:
    """Integer scalar carries that start at a constant and add one each iteration.

    Maps carry index to (start, step); always includes the proven counter. Their
    value in every iteration is known, so no kernel needs to compute them.
    """
    found = {loop.counter: (loop.start, 1)}
    count = len(loop.results)

    def invariant(value: _Node) -> _Node:
        if isinstance(value, _Parameter) and value in loop.parameters:
            k = loop.parameters.index(value)
            if k >= count or loop.results[k] is value:
                return loop.inputs[k]
        return value

    for k, (parameter, start, result) in enumerate(
        zip(loop.parameters[:count], loop.inputs[:count], loop.results, strict=True)
    ):
        if (
            k in found
            or parameter.shape
            or parameter.dtype.kind not in "iu"
            or not isinstance(start, _Constant)
            or start.splat is None
            or not isinstance(result, _Op)
            or result.name != "add"
        ):
            continue
        a, b = result.operands
        step = invariant(b if a is parameter else a if b is parameter else parameter)
        if isinstance(step, _Constant) and step.splat is not None:
            found[k] = (int(start.splat), int(step.splat))
    return found


def _in_place_carries(loop: _While) -> dict[int, _Op]:
    """Carries updated only by a dynamic_update_slice of themselves.

    Such a carry can live in one buffer for the whole loop: each iteration
    writes just the update. Safe only if nothing else in the body reads the
    old carry or the updated value, so no read can observe a partial write.
    """
    uses: dict[int, int] = {}
    seen: set[int] = set()
    pending = list(loop.results)
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for dependency in _dependencies(node):
            uses[id(dependency)] = uses.get(id(dependency), 0) + 1
            pending.append(dependency)
    for result in loop.results:
        uses[id(result)] = uses.get(id(result), 0) + 1
    carries = {}
    for k, result in enumerate(loop.results):
        parameter = loop.parameters[k]
        if (
            isinstance(result, _Op)
            and result.name == "dynamic_update_slice"
            and result.operands[0] is parameter
            and uses.get(id(result)) == 1
            and uses.get(id(parameter)) == 1
            and math.prod(result.operands[1].shape)
        ):
            carries[k] = result
    return carries


def _compile_graph(
    parameters: list[_Parameter],
    results: list[_Node],
    updates: Sequence[tuple[_Op, _Parameter]] = (),
) -> Program:
    """Slots and launches computing `results` from `parameters`.

    Each of `updates` is a dynamic_update_slice whose operand is the paired
    parameter: its update is written into that parameter's buffer in place.
    """
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
    stored: set[int] = set()  # ops materialized for the buffer limit or for branches

    def store(node: _Node) -> None:
        materialize(node)
        stored.add(id(node))

    def new_kernel(shape: tuple[int, ...]) -> _Kernel:
        return _Kernel(shape, stored, store)

    def launch(
        kernel: _Kernel,
        source: str,
        name: str,
        nodes: list[_Node],
        threads: int = 0,
        size: int | None = None,
    ) -> None:
        for operand in kernel.buffers:
            if id(operand) not in slot_of:
                add_constant(operand)
        bound = [slot_of[id(n)] for n in kernel.buffers] + [slot_of[id(n)] for n in nodes]
        if len(bound) > _MAX_BUFFERS:
            raise UnsupportedProgram(
                f"metal: a kernel needs {len(bound)} buffers, over Metal's {_MAX_BUFFERS}"
            )
        size = math.prod(nodes[0].shape) if size is None else size
        grid = size * (threads or 1)
        # Reductions run SAFE; RELAXED keeps folded extrema's infinities and NaNs.
        mode = 0 if threads else 1 if kernel.relaxed else 2
        launches.append(Launch(source, name, tuple(bound), grid, threads, mode))

    visited: set[int] = set()
    temporaries: list[_Node] = []  # keep synthetic nodes alive while slot_of uses their ids
    loop_slots: dict[_While, list[int]] = {}
    reduction_slots: dict[_VariadicReduce, list[int]] = {}
    dispatch_slots: dict[_Dispatch, list[int]] = {}

    def schedule_variadic(node: _ReductionResult) -> int:
        reduction = node.reduction
        if reduction in reduction_slots:
            return reduction_slots[reduction][node.index]
        for operand in reduction.inputs + reduction.initial:
            schedule(operand)
        outputs: list[_Node] = [
            _ReductionResult(reduction, k, value.dtype, node.shape)
            for k, value in enumerate(reduction.body)
        ]
        temporaries.extend(outputs)
        for result in outputs:
            fresh(result)
        kernel = new_kernel(node.shape)
        accumulators = [f"acc{k}" for k in range(len(outputs))]
        for name, value in zip(accumulators, reduction.initial, strict=True):
            initial = kernel.value(value, ())
            kernel.lines.append(f"{_ctype(value.dtype)} {name} = {initial};")
        shape = reduction.inputs[0].shape
        dims = sorted(reduction.dims)
        reduced_shape = tuple(shape[d] for d in dims)
        size = math.prod(reduced_shape)
        if size:
            kernel.lines.append(f"for (uint r = 0u; r < {size}u; ++r) {{")
            red_coords, out_coords = iter(_delinear("r", reduced_shape)), iter(kernel.coords)
            coords = tuple(
                next(red_coords) if d in dims else next(out_coords) for d in range(len(shape))
            )
            values = [kernel.value(value, coords) for value in reduction.inputs]
            # Lexicographic left fold preserves JAX's first-index tie/NaN rule.
            # Bind the actual reducer body rather than guessing an argmax pattern.
            for param, value in zip(reduction.parameters, accumulators + values, strict=True):
                kernel.values[(id(param), ())] = value
            new_values = [kernel.value(value, ()) for value in reduction.body]
            # Snapshot all results before mutating any accumulator (parallel tuple update).
            snapshots = []
            for value, result in zip(new_values, outputs, strict=True):
                tmp = kernel.temporary()
                kernel.lines.append(f"{_ctype(result.dtype)} {tmp} = {value};")
                snapshots.append(tmp)
            kernel.lines.extend(f"{a} = {v};" for a, v in zip(accumulators, snapshots, strict=True))
            kernel.lines.append("}")
        name = f"variadic_reduce_{len(launches)}"
        stores = [f"o{k}[i] = {v};" for k, v in enumerate(accumulators)]
        launch(kernel, kernel.render(outputs, name, stores), name, outputs)
        launches[-1] = dataclasses.replace(launches[-1], math_mode=0)
        reduction_slots[reduction] = [slot_of[id(result)] for result in outputs]
        return reduction_slots[reduction][node.index]

    def schedule_dispatch(node: _DispatchResult) -> int:
        dispatch = node.dispatch
        if dispatch in dispatch_slots:
            return dispatch_slots[dispatch][node.index]
        inputs = [materialize(x) for x in dispatch.inputs]
        outputs = []
        for k, result in enumerate(dispatch.results):
            if k in dispatch.aliases:
                # The result starts as a copy of its input; the kernel updates it.
                source = dispatch.inputs[dispatch.aliases[k]]
                copy = _Op("reshape", [source], result.dtype, result.shape, {})
                temporaries.append(copy)
                slot_of[id(result)] = materialize(copy)
            else:
                fresh(result)
            outputs.append(slot_of[id(result)])
        threadgroup = dispatch.threadgroup if math.prod(dispatch.threadgroup) else 0
        for element in range(dispatch.batch):
            launches.append(
                Launch(
                    dispatch.source,
                    dispatch.function,
                    tuple(inputs + outputs),
                    dispatch.grid,
                    threadgroup,
                    dispatch.math_mode,
                    tuple(element * s for s in dispatch.strides) if element else (),
                )
            )
        dispatch_slots[dispatch] = outputs
        return outputs[node.index]

    def materialize(node: _Node) -> int:
        schedule(node)
        if id(node) not in slot_of:
            if isinstance(node, _Constant) and node.splat is None:
                add_constant(node)
            else:
                name = f"init_{len(launches)}"
                kernel = new_kernel(node.shape)
                source = kernel.source([node], name)
                if len(kernel.buffers) >= _MAX_BUFFERS:
                    shrink(node)
                    name = f"init_{len(launches)}"
                    kernel = new_kernel(node.shape)
                    source = kernel.source([node], name)
                fresh(node)
                launch(kernel, source, name, [node])
        return slot_of[id(node)]

    def reads(node: _Node) -> set[int]:
        """Buffers a kernel computing `node` binds."""
        kernel = new_kernel(node.shape)
        kernel.value(node, kernel.coords_of(node.shape))
        return {id(b) for b in kernel.buffers}

    def shrink(node: _Node) -> None:
        """Materialize subexpressions of `node` until one kernel can compute it.

        Metal binds at most _MAX_BUFFERS buffers per kernel. A fused value
        that reads more stores its largest operand that fits in one kernel,
        descending into operands too large themselves; a stored operand then
        costs a store and a read instead of index arithmetic.
        """
        while len(reads(node)) >= _MAX_BUFFERS:
            operands = [
                o
                for o in _dependencies(node)
                if isinstance(o, _Op) and id(o) not in slot_of and id(o) not in stored
            ]
            if not operands:
                raise UnsupportedProgram(
                    f"metal: a value reads {len(reads(node))} buffers, over Metal's {_MAX_BUFFERS}"
                )
            largest = max(operands, key=lambda o: len(reads(o)))
            if len(reads(largest)) < _MAX_BUFFERS:
                materialize(largest)
                stored.add(id(largest))
            else:
                shrink(largest)

    def constant_slot(array: np.ndarray) -> int:
        slots.append(Slot("constant", array.dtype, array.shape, array.tobytes()))
        return len(slots) - 1

    def schedule_loop(loop: _While) -> list[int]:
        if loop in loop_slots:
            return loop_slots[loop]
        count = len(loop.results)
        if not loop.trips:
            loop_slots[loop] = [materialize(n) for n in loop.inputs[:count]]
            return loop_slots[loop]
        inductions = _inductions(loop)
        in_place = _in_place_carries(loop)
        current = []
        for k, node in enumerate(loop.inputs):
            if k in inductions:
                current.append(-1)  # bound per iteration below
            elif k in in_place:
                # One buffer per execution, initialized once, updated in place.
                copy = _Op("reshape", [node], node.dtype, node.shape, {})
                temporaries.append(copy)
                current.append(materialize(copy))
            else:
                current.append(materialize(node))
        # Force every other changing carry into its own output slot, even swaps
        # and constants. Only true invariants, in-place carries and induction
        # variables alias a body parameter. This makes two output banks safe for
        # permutations with any cycle length.
        body_results: list[_Node] = [
            loop.parameters[k]
            if result is loop.parameters[k] or k in inductions or k in in_place
            else _Op("reshape", [result], result.dtype, result.shape, {})
            for k, result in enumerate(loop.results)
        ]
        body = _compile_graph(
            loop.parameters,
            body_results,
            [(update, loop.parameters[k]) for k, update in in_place.items()],
        )
        if len(launches) + loop.trips * len(body.launches) > 100_000:
            raise UnsupportedProgram(
                "metal: stablehlo.while exceeds the 100000-launch replay limit"
            )
        fixed: dict[int, int] = {}
        banks: list[dict[int, int]] = [{}, {}]
        output_set = set(body.outputs)
        for k, slot in enumerate(body.slots):
            if slot.kind == "parameter":
                continue
            fixed[k] = len(slots)
            slots.append(slot)
            if k in output_set and slot.kind == "fresh":
                banks[0][k] = fixed[k]
                banks[1][k] = len(slots)
                slots.append(slot)

        # Induction values are known: iteration k binds element k of one
        # constant array per variable instead of computing it with a kernel.
        # Values wrap like the integer arithmetic they replace.
        def values(k: int, first: int, length: int) -> np.ndarray:
            start, step = inductions[k]
            steps = np.arange(first, first + length, dtype=np.int64)
            # int64 arithmetic and narrowing casts wrap like the kernels would.
            return (start + step * steps).astype(loop.parameters[k].dtype)

        arrays = {k: constant_slot(values(k, 0, loop.trips)) for k in inductions}
        for iteration in range(loop.trips):
            for k, array in arrays.items():
                current[k] = array
            bindings = fixed | dict(enumerate(current)) | banks[iteration % 2]
            offset = {k: iteration * loop.parameters[k].dtype.itemsize for k in arrays}
            launches.extend(
                dataclasses.replace(
                    item,
                    slots=tuple(bindings[s] for s in item.slots),
                    offsets=tuple(
                        (item.offsets[j] if item.offsets else 0) + offset.get(s, 0)
                        for j, s in enumerate(item.slots)
                    ),
                )
                for item in body.launches
            )
            current = [bindings[s] for s in body.outputs] + current[count:]
        for k in inductions:
            current[k] = constant_slot(values(k, loop.trips, 1).reshape(()))
        loop_slots[loop] = current[:count]
        return loop_slots[loop]

    def schedule(node: _Node) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        if isinstance(node, _ReductionResult):
            slot_of[id(node)] = schedule_variadic(node)
            return
        if isinstance(node, _LoopResult):
            slot_of[id(node)] = schedule_loop(node.loop)[node.index]
            return
        if isinstance(node, _DispatchResult):
            slot_of[id(node)] = schedule_dispatch(node)
            return
        if not isinstance(node, _Op):
            return
        for operand in node.operands:
            schedule(operand)
        if node.name == "scatter":
            schedule_scatter(node)
            return
        if node.name == "dot_general":
            schedule_dot(node)
            return
        if node.name != "reduce" or _short(node):
            return  # short reductions are folded into the kernels that read them
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
            kernel, source = _reduction_kernel(node, partial, name, chunks, new_kernel)
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
        kernel, source = _reduction_kernel(reduction, node, name, new_kernel=new_kernel)
        launch(kernel, source, name, [node], 128)

    def schedule_scatter(node: _Op) -> None:
        x, indices, updates = node.operands
        narrow = node.dtype in (np.float16, ml_dtypes.bfloat16)
        if (
            not narrow and node.dtype not in (np.float32, np.int32, np.uint32)
        ) or indices.dtype not in (
            np.int32,
            np.uint32,
        ):
            raise UnsupportedProgram(
                "metal: stablehlo.scatter requires floating or int32/uint32 data and 32-bit indices"
            )
        fresh(node)
        kernel = new_kernel(node.shape)
        name = f"scatter_init_{len(launches)}"
        if not math.prod(updates.shape) or not math.prod(node.shape):
            launch(kernel, kernel.source([x], name), name, [node])
            return
        target: _Node = node
        if narrow:
            # One 32-bit atomic word per element, including odd-sized outputs.
            # Packing adjacent halves would require aligned/padded allocations.
            target = _Parameter(-1, np.dtype(np.uint32), node.shape)
            temporaries.append(target)
            fresh(target)
            value = kernel.value(x, kernel.coords)
            source = kernel.render([target], name, [f"o0[i] = uint(as_type<ushort>({value}));"])
        else:
            source = kernel.source([x], name)
        launch(kernel, source, name, [target])
        a = node.attributes
        kernel = new_kernel(updates.shape)
        batch_coords = iter(c for d, c in enumerate(kernel.coords) if d not in a["window"])
        index_coords = tuple(
            "0u" if d == a["vector"] else next(batch_coords) for d in range(len(indices.shape))
        )
        inner = ["0l"] * len(node.shape)
        for component, dim in enumerate(a["map"]):
            read_coords = list(index_coords)
            if a["vector"] < len(indices.shape):
                read_coords[a["vector"]] = f"{component}u"
            inner[dim] = f"long({kernel.value(indices, tuple(read_coords))})"
        for dim, index_dim in zip(a["batch"], a["index_batch"], strict=True):
            inner[dim] = f"long({index_coords[index_dim]})"
        window_dims = [d for d in range(len(node.shape)) if d not in a["inserted"] + a["batch"]]
        for dim, offset in zip(window_dims, a["window"], strict=True):
            inner[dim] = f"({inner[dim]} + long({kernel.coords[offset]}))"
        condition = (
            " && ".join(
                f"({c} >= 0l && {c} < {n}l)" for c, n in zip(inner, node.shape, strict=True)
            )
            or "true"
        )
        kernel.lines.append(f"if ({condition}) {{")
        value = kernel.value(updates, kernel.coords)
        flat = _linear(tuple(f"uint({c})" for c in inner), node.shape)
        ctype = _ctype(node.dtype)
        if narrow:
            # Round after every successful update, as a narrow scalar add does.
            kernel.lines += [
                f"device atomic_uint* address = reinterpret_cast<device atomic_uint*>(o0) + ({flat});",
                "uint expected = atomic_load_explicit(address, memory_order_relaxed);",
                "uint desired;",
                "do {",
                f"{ctype} next = {ctype}(float(as_type<{ctype}>(ushort(expected))) + float({value}));",
                "desired = uint(as_type<ushort>(next));",
                "} while (!atomic_compare_exchange_weak_explicit(address, &expected, desired, memory_order_relaxed, memory_order_relaxed));",
            ]
        else:
            kernel.lines.append(
                f"atomic_fetch_add_explicit(reinterpret_cast<device atomic_{ctype}*>(o0) + ({flat}), {value}, memory_order_relaxed);"
            )
        kernel.lines.append("}")
        name = f"scatter_add_{len(launches)}"
        source = kernel.render([target], name, [])
        # This grid covers updates, not output elements.
        launch(kernel, source, name, [target])
        launches[-1] = dataclasses.replace(launches[-1], grid=math.prod(updates.shape), math_mode=0)
        if narrow:
            kernel = new_kernel(node.shape)
            value = kernel.value(target, kernel.coords)
            name = f"scatter_unpack_{len(launches)}"
            source = kernel.render([node], name, [f"o0[i] = as_type<{ctype}>(ushort({value}));"])
            launch(kernel, source, name, [node])

    def schedule_dot(node: _Op) -> None:
        from palladium._stablehlo_gemm import gemm_source

        lhs, rhs = node.operands
        floats = (np.dtype(np.float32), np.dtype(np.float16), np.dtype(ml_dtypes.bfloat16))
        if lhs.dtype != rhs.dtype or lhs.dtype not in floats or node.dtype not in floats:
            raise UnsupportedProgram(
                "metal: stablehlo.dot_general requires matching floating operands"
            )
        if node.dtype != lhs.dtype and node.dtype != np.float32:
            raise UnsupportedProgram(
                "metal: stablehlo.dot_general only supports widening results to float32"
            )
        a = node.attributes
        lb, rb, lc, rc = (
            a[key] for key in ("lhs_batch", "rhs_batch", "lhs_contract", "rhs_contract")
        )
        lf = tuple(d for d in range(len(lhs.shape)) if d not in lb + lc)
        rf = tuple(d for d in range(len(rhs.shape)) if d not in rb + rc)
        batch = math.prod(lhs.shape[d] for d in lb)
        m, n = math.prod(lhs.shape[d] for d in lf), math.prod(rhs.shape[d] for d in rf)
        k = math.prod(lhs.shape[d] for d in lc)
        fresh(node)
        if not math.prod(node.shape):
            return
        if not k:
            zero = _Constant(node.dtype, node.shape, 0)
            kernel = new_kernel(node.shape)
            name = f"empty_dot_{len(launches)}"
            # Use the dot's output slot for the zero-producing kernel.
            launch(kernel, kernel.source([zero], name), name, [node])
            return

        def pack(value: _Node, perm: tuple[int, ...], shape: tuple[int, ...]) -> int:
            if perm != tuple(range(len(value.shape))):
                value = _Op(
                    "transpose",
                    [value],
                    value.dtype,
                    tuple(value.shape[d] for d in perm),
                    {"perm": perm},
                )
            if value.dtype != node.dtype:
                value = _Op("convert", [value], node.dtype, value.shape, {})
            # Buffer metadata may retain its original shape: a canonical reshape
            # changes neither the storage nor its row-major element order.
            if id(value) in slot_of:
                return slot_of[id(value)]
            value = _Op("reshape", [value], value.dtype, shape, {})
            temporaries.append(value)
            return materialize(value)

        left = pack(lhs, lb + lf + lc, (batch, m, k))
        right = pack(rhs, rb + rc + rf, (batch, k, n))
        try:
            source, grid = gemm_source(batch, m, n, k, node.dtype)
        except EmitError as e:
            raise UnsupportedProgram(f"metal: stablehlo.dot_general: {e}") from e
        launches.append(Launch(source, "gemm", (left, right, slot_of[id(node)]), grid, 128))

    for node in results:
        schedule(node)
    for update, _ in updates:
        for operand in update.operands[1:]:
            schedule(operand)

    # One kernel per element count: outputs of different shapes but equal size
    # share a launch, each read at the coordinates of the flat index.
    computed: dict[int, list[_Node]] = {}
    for node in results:
        if id(node) in slot_of:
            continue
        if isinstance(node, _Constant) and node.splat is None:
            add_constant(node)
            continue
        fresh(node)
        computed.setdefault(math.prod(node.shape), []).append(node)
    in_place: dict[int, list[tuple[_Op, _Parameter]]] = {}
    for update, target in updates:
        in_place.setdefault(math.prod(update.operands[1].shape), []).append((update, target))

    def fuse(shape: tuple[int, ...], items: list) -> tuple[_Kernel, list[str], list[_Node]]:
        """One kernel storing each item: a node, or an (update, target) pair."""
        kernel = new_kernel(shape)
        stores: list[str] = []
        targets: list[_Node] = []
        for item in items:
            if isinstance(item, _Node):
                coords = kernel.coords_of(item.shape)
                stores.append(f"o{len(targets)}[i] = {kernel.value(item, coords)};")
                targets.append(item)
                continue
            update, target = item
            x, value, *starts = update.operands
            limits = tuple(n - s for n, s in zip(x.shape, value.shape, strict=True))
            offsets = kernel.offsets(starts, limits)
            coords = kernel.coords_of(value.shape)
            flat = _linear(
                tuple(f"({c} + {s})" for c, s in zip(coords, offsets, strict=True)), x.shape
            )
            stores.append(f"o{len(targets)}[{flat}] = {kernel.value(value, coords)};")
            targets.append(target)
        return kernel, stores, targets

    def bins(shape: tuple[int, ...], items: list) -> list[list]:
        """Items in groups whose kernels bind at most _MAX_BUFFERS buffers."""
        kernel, _, targets = fuse(shape, items)
        if len(kernel.buffers) + len(targets) <= _MAX_BUFFERS:
            return [items]
        groups: list[list] = []
        reads: set[int] = set()
        for item in items:
            shrink(item if isinstance(item, _Node) else item[0].operands[1])
            needed = {id(b) for b in fuse(shape, [item])[0].buffers}
            if not groups or len(reads | needed) + len(groups[-1]) + 1 > _MAX_BUFFERS:
                groups.append([])
                reads = set()
            groups[-1].append(item)
            reads |= needed
        return groups

    for size in [*computed, *(s for s in in_place if s not in computed)]:
        nodes = computed.get(size, [])
        shape = nodes[0].shape if nodes else in_place[size][0][0].operands[1].shape
        for items in bins(shape, [*nodes, *in_place.get(size, [])]):
            kernel, stores, targets = fuse(shape, items)
            name = f"fused_{len(launches)}"
            launch(kernel, kernel.render(targets, name, stores), name, targets, size=size)
    return Program(slots, launches, [slot_of[id(n)] for n in results])
