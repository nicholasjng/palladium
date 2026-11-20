"""A StableHLO module as a DAG of nodes, with calls and composites inlined."""

from __future__ import annotations

import collections
import dataclasses
import math
from collections.abc import Callable, Iterable, Iterator, Sequence

import ml_dtypes
import numpy as np
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

from palladium.stablehlo.program import UnsupportedProgram
from palladium.stablehlo.scalar import SCALAR

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


_BOOL = np.dtype(np.bool_)


def tensor_type(value: ir.Value) -> tuple[np.dtype, tuple[int, ...]]:
    if str(value.type) == "!stablehlo.token":
        # The token of an ordered effect carries no data: an empty buffer.
        # Host steps run in program order without it.
        return np.dtype(np.bool_), (0,)
    t = ir.RankedTensorType(value.type)
    name = str(t.element_type)
    if name not in _DTYPES:
        raise UnsupportedProgram(f"metal: element type {name} is not supported")
    return _DTYPES[name], tuple(t.shape)


class Node:
    dtype: np.dtype
    shape: tuple[int, ...]


@dataclasses.dataclass(eq=False)
class Parameter(Node):
    dtype: np.dtype
    shape: tuple[int, ...]


@dataclasses.dataclass(eq=False)
class Buffer(Node):
    """A buffer the compiler adds between launches, such as partial sums."""

    dtype: np.dtype
    shape: tuple[int, ...]


@dataclasses.dataclass(eq=False)
class Constant(Node):
    dtype: np.dtype
    shape: tuple[int, ...]
    splat: float | None  # None: `array` holds the elements
    array: np.ndarray | None = None


@dataclasses.dataclass(eq=False)
class Op(Node):
    name: str  # StableHLO op name without the dialect
    operands: list[Node]
    dtype: np.dtype
    shape: tuple[int, ...]
    attributes: dict


# A `case` branch something lies in: the selector, the branch, and the number
# of branches. Kernels compute every branch and select; what runs on the host
# may raise, print or never end, so it runs only in the branch taken.
Guard = tuple[Node, int, int]


def guard_flag(guards: Sequence[Guard]) -> Node:
    """A boolean scalar: whether every branch of `guards` is the one taken."""

    def compare(index: Node, direction: str, value: int) -> Op:
        operands = [index, Constant(index.dtype, (), value)]
        return Op("compare", operands, _BOOL, (), {"direction": direction, "total": False})

    taken: Node | None = None
    for index, branch, count in guards:
        if branch < count - 1:
            here = compare(index, "eq", branch)
        else:
            # An index out of range selects the last branch.
            out_of_range: list[Node] = [
                compare(index, "lt", 0),
                compare(index, "ge", count - 1),
            ]
            here = Op("or", out_of_range, _BOOL, (), {})
        taken = here if taken is None else Op("and", [taken, here], _BOOL, (), {})
    assert taken is not None
    return taken


@dataclasses.dataclass(eq=False)
class While:
    inputs: list[Node]  # carries followed by lexical captures
    parameters: list[Parameter]
    results: list[Node]  # updated carries
    # Iterations, when the loop provably counts to a constant bound. None:
    # the host reads `condition` before every iteration.
    trips: int | None
    # Integer scalar carries with known values: carry index to (start, step).
    inductions: dict[int, tuple[int, int]]
    condition: Node
    # Host calls and loops in the body that run for their effect, in order.
    effects: list[Host | While] = dataclasses.field(default_factory=list)
    # Set for a loop the host drives or that calls it: it runs only if taken.
    guards: tuple[Guard, ...] = ()


@dataclasses.dataclass(eq=False)
class LoopResult(Node):
    loop: While
    index: int
    dtype: np.dtype
    shape: tuple[int, ...]


@dataclasses.dataclass(eq=False)
class Reducer:
    """A reducer or scatter body that is not one known binary op: the node
    combining its two scalar parameters, the accumulated value first."""

    parameters: list[Parameter]
    body: Node


@dataclasses.dataclass(eq=False)
class VariadicReduce:
    inputs: list[Node]
    initial: list[Node]
    parameters: list[Parameter]
    body: list[Node]
    dims: tuple[int, ...]


@dataclasses.dataclass(eq=False)
class ReductionResult(Node):
    reduction: VariadicReduce
    index: int
    dtype: np.dtype
    shape: tuple[int, ...]


@dataclasses.dataclass(eq=False)
class Sort:
    """A sort of `inputs` together along `dim`. `body` says whether the
    element in the first of each input's two `parameters` comes before the
    one in the second."""

    inputs: list[Node]
    parameters: list[Parameter]
    body: Node
    dim: int


@dataclasses.dataclass(eq=False)
class SortResult(Node):
    sort: Sort
    index: int
    dtype: np.dtype
    shape: tuple[int, ...]


@dataclasses.dataclass(eq=False)
class Dispatch:
    """A palladium kernel with its own launch: a `palladium_dispatch` custom call.

    `pallas_call` (lowered for `metal`) and `metal_call` both emit this call;
    its typed-FFI attributes carry the MSL and the launch geometry. Buffers
    bind inputs then results, in order. `aliases` maps a result to the input
    whose values it starts with.
    """

    inputs: list[Node]
    results: list[DispatchResult]
    source: str
    function: str
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]  # zeros: the pipeline chooses
    math_mode: int
    batch: int  # launches, each offset by `strides` bytes per buffer
    strides: tuple[int, ...]
    aliases: dict[int, int]


@dataclasses.dataclass(eq=False)
class DispatchResult(Node):
    dispatch: Dispatch
    index: int
    dtype: np.dtype
    shape: tuple[int, ...]


@dataclasses.dataclass(eq=False)
class Host:
    """A function the host runs on arrays: a Python callback, or an op without
    a kernel. With `effect`, it runs even if nothing reads its results."""

    inputs: list[Node]
    results: list[HostResult]
    function: Callable[..., Sequence[np.ndarray]]
    name: str
    effect: bool = False
    guards: tuple[Guard, ...] = ()


@dataclasses.dataclass(eq=False)
class HostResult(Node):
    host: Host
    index: int
    dtype: np.dtype
    shape: tuple[int, ...]


def host_call(
    name: str,
    function: Callable[..., Sequence[np.ndarray]],
    inputs: list[Node],
    types: Sequence[tuple[np.dtype, tuple[int, ...]]],
    effect: bool = False,
) -> Host:
    """A call of `function` on the host, its results of the given types."""
    host = Host(inputs, [], function, name, effect)
    host.results = [HostResult(host, k, dtype, shape) for k, (dtype, shape) in enumerate(types)]
    return host


def copy_of(node: Node) -> Op:
    """A distinct node with the values of `node`, for a buffer of its own."""
    return Op("reshape", [node], node.dtype, node.shape, {})


def dependencies(node: Node) -> list[Node]:
    if isinstance(node, Op):
        return node.operands
    if isinstance(node, LoopResult):
        return node.loop.inputs
    if isinstance(node, ReductionResult):
        return node.reduction.inputs + node.reduction.initial
    if isinstance(node, DispatchResult):
        return node.dispatch.inputs
    if isinstance(node, HostResult):
        return node.host.inputs
    if isinstance(node, SortResult):
        return node.sort.inputs
    return []


def reachable(roots: Iterable[Node]) -> Iterator[Node]:
    """Each node `roots` depend on, the roots included, once."""
    seen: set[Node] = set()
    pending = list(roots)
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            yield node
            pending.extend(dependencies(node))


def _invariant(
    value: Node, parameters: list[Parameter], inputs: list[Node], results: list[Node]
) -> Node:
    """A loop parameter that every iteration passes on unchanged, as its input."""
    if isinstance(value, Parameter) and value in parameters:
        k = parameters.index(value)
        if k >= len(results) or results[k] is value:
            return inputs[k]
    return value


def _inductions(
    parameters: list[Parameter], inputs: list[Node], results: list[Node]
) -> dict[int, tuple[int, int]]:
    """Integer scalar carries that start at a constant and add a constant each iteration.

    Maps carry index to (start, step). Their value in every iteration is
    known, so no kernel needs to compute them.
    """
    found = {}
    for k, (parameter, start, result) in enumerate(
        zip(parameters[: len(results)], inputs, results, strict=False)
    ):
        if (
            parameter.shape
            or parameter.dtype.kind not in "iu"
            or not isinstance(start, Constant)
            or start.splat is None
            or not isinstance(result, Op)
            or result.name != "add"
        ):
            continue
        a, b = result.operands
        step = b if a is parameter else a if b is parameter else parameter
        step = _invariant(step, parameters, inputs, results)
        if isinstance(step, Constant) and step.splat is not None:
            found[k] = (int(start.splat), int(step.splat))
    return found


def _trip_count(
    predicate: Node,
    parameters: list[Parameter],
    inputs: list[Node],
    results: list[Node],
    inductions: dict[int, tuple[int, int]],
) -> int | None:
    """The iterations of a loop that provably runs a counter from a constant
    start, by one, up to a constant bound; None for any other loop."""
    if (
        not isinstance(predicate, Op)
        or predicate.name != "compare"
        or predicate.attributes["direction"] != "lt"
        or predicate.attributes["total"]
    ):
        return None
    counter, bound = predicate.operands
    # An invariant carry or lexical capture can supply a constant loop bound.
    bound = _invariant(bound, parameters, inputs, results)
    in_loop = isinstance(counter, Parameter) and counter in parameters
    k = parameters.index(counter) if in_loop else -1
    if (
        counter.dtype.kind not in "iu"
        or inductions.get(k, (0, 0))[1] != 1
        or not isinstance(bound, Constant)
        or bound.splat is None
    ):
        return None
    return max(0, int(bound.splat) - inductions[k][0])


def _constant(op) -> Constant:
    dtype, shape = tensor_type(op.result)
    attr = ir.DenseElementsAttr(op.attributes["value"])
    if attr.is_splat:
        value = attr.get_splat_value()
        for kind in (ir.BoolAttr, ir.FloatAttr, ir.IntegerAttr):
            if isinstance(value, kind):
                return Constant(dtype, shape, value.value)
    if dtype == ml_dtypes.bfloat16:
        fp = ir.DenseFPElementsAttr(attr)
        array = np.array([fp[i] for i in range(len(fp))], dtype).reshape(shape)
    else:
        array = np.array(attr, dtype).reshape(shape)
    return Constant(dtype, shape, None, array)


def _array(a, name: str, default: int, rank: int) -> list[int]:
    """An optional dense i64 array attribute, `default` in every dimension when absent."""
    return list(ir.DenseI64ArrayAttr(a[name])) if name in a else [default] * rank


def _pairs(a, name: str, rank: int) -> list[tuple[int, int]]:
    """An optional (rank, 2) padding attribute as (low, high) pairs, zero when absent."""
    if name not in a:
        return [(0, 0)] * rank
    values = ir.DenseIntElementsAttr(a[name])
    if values.is_splat:
        v = ir.IntegerAttr(values.get_splat_value()).value
        return [(v, v)] * rank
    return [(values[2 * d], values[2 * d + 1]) for d in range(rank)]


REDUCERS = ("add", "multiply", "maximum", "minimum", "and", "or", "xor")


def _reducer(op, region: int = 0) -> str | None:
    """The op of a body that applies one known binary op to its two
    arguments, or None for any other body."""
    block = op.regions[region].blocks[0]
    body = list(block.operations)
    if len(op.results) != 1 or len(body) != 2:
        return None
    reducer, ret = body
    kind = reducer.operation.name.removeprefix("stablehlo.")
    if (
        kind not in REDUCERS
        or list(reducer.operands) != list(block.arguments)
        or ret.operation.name != "stablehlo.return"
        or list(ret.operands) != list(reducer.results)
    ):
        return None
    return kind


def _window(op) -> dict:
    """The geometry of a reduce_window."""
    a = op.attributes
    rank = len(tensor_type(op.operands[0])[1])
    return {
        "window": _array(a, "window_dimensions", 1, rank),
        "strides": _array(a, "window_strides", 1, rank),
        "base": _array(a, "base_dilations", 1, rank),
        "dilations": _array(a, "window_dilations", 1, rank),
        "padding": _pairs(a, "padding", rank),
    }


# Cumulative reductions along more elements than this run in blocks.
CUMULATIVE_DIRECT = 128


def _cumulative(x: Node, init: Node, a: dict, shape: tuple[int, ...]) -> Node | None:
    """A reduce_window that is a cumulative reduction along one long axis, in
    blocks; None for any other window.

    JAX lowers `cumsum` and its relatives to a window as long as the axis,
    padded by its length less one before (or, reversed, after). Folding each
    window alone would read the square of the length. Instead: the total of
    each block of the axis, then for each block the total of the blocks
    before it, then each element from its block's start. The reducer must be
    associative and the initial value its identity, as StableHLO asks.
    """
    moving = [
        d
        for d in range(len(x.shape))
        if (a["window"][d], a["strides"][d], a["base"][d], *a["padding"][d]) != (1, 1, 1, 0, 0)
    ]
    if len(moving) != 1 or shape != x.shape:
        return None
    [axis] = moving
    n = x.shape[axis]
    padding = tuple(a["padding"][axis])
    if (
        n <= CUMULATIVE_DIRECT
        or (a["window"][axis], a["strides"][axis], a["base"][axis]) != (n, 1, 1)
        or padding not in ((n - 1, 0), (0, n - 1))
    ):
        return None
    # Work per row: n * block / 2 within blocks, (n / block)^2 across them.
    block = max(16, round((4 * n) ** (1 / 3)))
    common = {"axis": axis, "block": block, "reducer": a["reducer"], "reverse": padding[0] == 0}
    blocks = (*x.shape[:axis], -(-n // block), *x.shape[axis + 1 :])
    totals = Op("fold_blocks", [x, init], x.dtype, blocks, common)
    before = Op("fold_before", [totals, init], x.dtype, blocks, common)
    return Op("fold_within", [x, before], x.dtype, x.shape, common)


# Internal ops that fold with a loop per output element: computed once into a
# buffer, never refolded by each kernel that reads them.
STORED = ("fold_blocks", "fold_before", "fold_within")


def _attributes(op) -> dict:
    a = op.attributes
    name = op.operation.name.removeprefix("stablehlo.")
    if name == "scatter":
        if len(op.results) != 1:
            raise UnsupportedProgram("metal: stablehlo.scatter of several operands")
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
        return {"dims": list(ir.DenseI64ArrayAttr(a["dimensions"])), "reducer": _reducer(op)}
    if name == "select_and_scatter":
        select, scatter = (list(r.blocks[0].operations) for r in op.regions)
        arguments = list(op.regions[0].blocks[0].arguments)
        direction = (
            stablehlo.ComparisonDirectionAttr(select[0].attributes["comparison_direction"]).value
            if select[0].operation.name == "stablehlo.compare"
            else ""
        ).lower()
        if (
            len(select) != 2
            or direction not in ("ge", "gt", "le", "lt")
            or list(select[0].operands) != arguments
            or list(select[1].operands) != list(select[0].results)
            or len(scatter) != 2
            or scatter[0].operation.name != "stablehlo.add"
            or list(scatter[0].operands) != list(op.regions[1].blocks[0].arguments)
            or list(scatter[1].operands) != list(scatter[0].results)
        ):
            raise UnsupportedProgram(
                "metal: stablehlo.select_and_scatter requires a comparison and an add"
            )
        rank = len(tensor_type(op.operands[0])[1])
        return {
            "direction": direction,
            "window": _array(a, "window_dimensions", 1, rank),
            "strides": _array(a, "window_strides", 1, rank),
            "padding": _pairs(a, "padding", rank),
        }
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
    if name == "reduce_precision":
        return {
            "exponent": ir.IntegerAttr(a["exponent_bits"]).value,
            "mantissa": ir.IntegerAttr(a["mantissa_bits"]).value,
        }
    if name == "compare":
        # A total order ranks NaNs and the two zeros, which MSL's operators do not.
        total = (
            "compare_type" in a
            and stablehlo.ComparisonTypeAttr(a["compare_type"]).value == "TOTALORDER"
        )
        direction = stablehlo.ComparisonDirectionAttr(a["comparison_direction"]).value
        return {"direction": direction.lower(), "total": total}
    return {}


_DISPATCH_TARGET = "palladium_dispatch"


# Elements one reduce_window or select_and_scatter output element may read,
# each unrolled: pooling windows, not sliding sums over whole axes.
WINDOW_LIMIT = 64


# Reductions over at most this many elements are folded into their consumer's
# kernel, unrolled per output element, rather than given 128 threads per output.
SHORT_REDUCTION = 16


def is_short(node: Op) -> bool:
    x = node.operands[0]
    size = math.prod(x.shape[d] for d in node.attributes["dims"])
    return node.dtype not in (np.int64, np.uint64) and size <= SHORT_REDUCTION


def is_looped(node: Op) -> bool:
    """Whether a reduce_window folds with a loop instead of unrolled: its
    window is long, or its reducer a body of its own."""
    a = node.attributes
    return not isinstance(a["reducer"], str) or math.prod(a["window"]) > WINDOW_LIMIT


def is_stored(node: Node) -> bool:
    """Whether scheduling `node` computes it once into a buffer for all readers."""
    return isinstance(node, Op) and (
        node.name in STORED or (node.name == "reduce_window" and is_looped(node))
    )


def owns_slot(node: Node) -> bool:
    """Whether scheduling `node` computes it into a fresh buffer of its own."""
    if isinstance(node, (ReductionResult, DispatchResult, HostResult, SortResult)):
        return True
    return isinstance(node, Op) and (
        node.name in ("dot_general", "scatter") or (node.name == "reduce" and not is_short(node))
    )


def _dispatch(op, inputs: list[Node]) -> Dispatch:
    """A `palladium_dispatch` custom call as a launch description."""
    a = op.attributes
    target = ir.StringAttr(a["call_target_name"]).value
    if target != _DISPATCH_TARGET:
        raise UnsupportedProgram(f"metal: custom call {target!r} is not supported")
    if "mhlo.backend_config" not in a:
        raise UnsupportedProgram(f"metal: {target} requires typed-FFI attributes")
    for name, values in (("operand_layouts", op.operands), ("result_layouts", op.results)):
        if name not in a:
            continue
        for layout, value in zip(ir.ArrayAttr(a[name]), values, strict=True):
            rank = len(tensor_type(value)[1])
            if [int(d) for d in ir.DenseIntElementsAttr(layout)] != list(range(rank - 1, -1, -1)):
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
    dispatch = Dispatch(
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
        DispatchResult(dispatch, k, *tensor_type(value)) for k, value in enumerate(op.results)
    ]
    for k, value in enumerate(dispatch.results):
        if k in aliases:
            source = inputs[aliases[k]]
            if (source.dtype, source.shape) != (value.dtype, value.shape):
                raise UnsupportedProgram(f"metal: {target} aliases mismatched buffers")
    return dispatch


_SUPPORTED = {
    *SCALAR,
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
    "select_and_scatter",
    "reduce_precision",
}


def _grouped_product(
    patches: Op, kernel: Node, dtype: np.dtype, rank: int, feature_groups: int, batch_groups: int
) -> Op:
    """The product of a grouped convolution's patches, (batch, output
    spatial..., window..., feature), with its kernel, (window..., input
    feature, output feature), as (batch, output spatial..., output feature).

    Each group is one product of a batch: feature groups split the patches'
    features and batch groups their batch, and both split the kernel's output
    features. A group contracting few elements, as in a depthwise
    convolution, is a sum the reading kernel unrolls instead of a GEMM.
    """
    groups = feature_groups * batch_groups
    batch, *outputs = patches.shape[: 1 + rank]
    window, features = patches.shape[1 + rank : 1 + 2 * rank], patches.shape[-1]
    *_, inputs, out_features = kernel.shape
    per_group = out_features // groups
    if feature_groups > 1:
        split = (batch, *outputs, *window, groups, features // groups)
        lhs_batch = (1 + 2 * rank,)
        lhs_contract = (*range(1 + rank, 1 + 2 * rank), 2 + 2 * rank)
    else:
        split = (groups, batch // groups, *outputs, *window, features)
        lhs_batch = (0,)
        lhs_contract = tuple(range(2 + rank, 3 + 2 * rank))
    lhs = Op("reshape", [patches], patches.dtype, split, {})
    rhs = Op("reshape", [kernel], kernel.dtype, (*window, inputs, groups, per_group), {})
    rows = batch // batch_groups
    product = Op(
        "dot_general",
        [lhs, rhs],
        dtype,
        (groups, rows, *outputs, per_group),
        {
            "lhs_batch": lhs_batch,
            "rhs_batch": (rank + 1,),
            "lhs_contract": lhs_contract,
            "rhs_contract": tuple(range(rank + 1)),
        },
    )
    if math.prod(window) * inputs <= SHORT_REDUCTION:
        product = _summed_products(product)
    order = (*range(1, 2 + rank), 0, 2 + rank)
    grouped = Op(
        "transpose",
        [product],
        dtype,
        tuple(product.shape[k] for k in order),
        {"perm": order},
    )
    return Op("reshape", [grouped], dtype, (rows, *outputs, out_features), {})


def _convolution(op, lhs: Node, rhs: Node) -> Node:
    """A convolution as a product of its input's patches with the kernel.

    `conv_patches` is an index map from the input to (batch, output spatial...,
    window..., input feature), with zeros where the window falls on padding or
    between dilated input elements; reversed window dimensions read the
    window backwards. The product with the kernel, transposed to (window...,
    input feature, output feature), runs on the GEMM, so every form JAX
    emits, gradients included, needs only these two operations.
    """
    a = op.attributes
    feature_groups = ir.IntegerAttr(a["feature_group_count"]).value
    batch_groups = ir.IntegerAttr(a["batch_group_count"]).value
    groups = feature_groups * batch_groups
    d = stablehlo.ConvDimensionNumbers(a["dimension_numbers"])
    spatial = list(d.input_spatial_dimensions)
    kernel_spatial = list(d.kernel_spatial_dimensions)
    output_spatial = list(d.output_spatial_dimensions)
    rank = len(spatial)
    dtype, shape = tensor_type(op.result)
    window = tuple(rhs.shape[k] for k in kernel_spatial)
    outputs = tuple(shape[o] for o in output_spatial)
    reversal = (
        [bool(x) for x in ir.DenseBoolArrayAttr(a["window_reversal"])]
        if "window_reversal" in a
        else [False] * rank
    )
    strides = _array(a, "window_strides", 1, rank)
    padding = _pairs(a, "padding", rank)
    lhs_dilation = _array(a, "lhs_dilation", 1, rank)
    rhs_dilation = _array(a, "rhs_dilation", 1, rank)
    batch, features = lhs.shape[d.input_batch_dimension], lhs.shape[d.input_feature_dimension]
    attributes = {
        "batch": d.input_batch_dimension,
        "feature": d.input_feature_dimension,
        "spatial": spatial,
        "strides": strides,
        "padding": padding,
        "lhs_dilation": lhs_dilation,
        "rhs_dilation": rhs_dilation,
        "reversal": reversal,
    }
    perm = (*kernel_spatial, d.kernel_input_feature_dimension, d.kernel_output_feature_dimension)
    kernel = rhs
    if perm != tuple(range(len(perm))):
        kernel = Op(
            "transpose", [rhs], rhs.dtype, tuple(rhs.shape[p] for p in perm), {"perm": perm}
        )
    phased = (
        groups == 1
        and any(n > 1 for n in lhs_dilation)
        and all(
            n == 1 or (s, r, rev, w % n, o % n) == (1, 1, False, 0, 0)
            for n, s, r, rev, w, o in zip(
                lhs_dilation, strides, rhs_dilation, reversal, window, outputs, strict=True
            )
        )
    )
    if groups > 1:
        product = _grouped_product(
            Op("conv_patches", [lhs], lhs.dtype, (batch, *outputs, *window, features), attributes),
            kernel,
            dtype,
            rank,
            feature_groups,
            batch_groups,
        )
    elif not phased:
        patches = Op(
            "conv_patches", [lhs], lhs.dtype, (batch, *outputs, *window, features), attributes
        )
        product = Op(
            "dot_general",
            [patches, kernel],
            dtype,
            (batch, *outputs, kernel.shape[-1]),
            {
                "lhs_batch": (),
                "rhs_batch": (),
                "lhs_contract": tuple(range(1 + rank, 2 + 2 * rank)),
                "rhs_contract": tuple(range(rank + 1)),
            },
        )
    else:
        # Dilating the input by n puts real elements at every n-th position:
        # output phase o % n meets the kernel taps congruent to one residue
        # mod n, and only real elements. Each phase is a convolution of the
        # undilated input with its window / n taps, so the phases form a
        # batched product without the dilation's zeros.
        phases, taps = lhs_dilation, tuple(w // n for w, n in zip(window, lhs_dilation))
        reduced = tuple(o // n for o, n in zip(outputs, lhs_dilation))
        tapped = Op(
            "conv_taps",
            [kernel],
            kernel.dtype,
            (*phases, *taps, *kernel.shape[-2:]),
            {"lhs_dilation": lhs_dilation, "padding": padding},
        )
        # Where each phase's first tap lands in the undilated input, relative
        # to its reduced output position.
        shifts = [
            {(p + (low - p) % n - low) // n for p in range(n)}
            for n, (low, _) in zip(lhs_dilation, padding, strict=True)
        ]
        shared = all(len(shift) == 1 for shift in shifts)
        if shared:
            # Every phase reads the same patches: one product against the
            # taps of all phases, (tap..., in feature, phase..., out feature).
            patches = Op(
                "conv_phases",
                [lhs],
                lhs.dtype,
                (batch, *reduced, *taps, features),
                {**attributes, "shared": True},
            )
            order = (*range(rank, 2 * rank + 1), *range(rank), 2 * rank + 1)
            tapped = Op(
                "transpose",
                [tapped],
                kernel.dtype,
                tuple(tapped.shape[k] for k in order),
                {"perm": order},
            )
            phased_product = Op(
                "dot_general",
                [patches, tapped],
                dtype,
                (batch, *reduced, *phases, kernel.shape[-1]),
                {
                    "lhs_batch": (),
                    "rhs_batch": (),
                    "lhs_contract": tuple(range(1 + rank, 2 + 2 * rank)),
                    "rhs_contract": tuple(range(rank + 1)),
                },
            )
            interleave = (0, *(k for i in range(rank) for k in (1 + i, 1 + rank + i)), 2 * rank + 1)
        else:
            patches = Op(
                "conv_phases",
                [lhs],
                lhs.dtype,
                (*phases, batch, *reduced, *taps, features),
                attributes,
            )
            phased_product = Op(
                "dot_general",
                [patches, tapped],
                dtype,
                (*phases, batch, *reduced, kernel.shape[-1]),
                {
                    "lhs_batch": tuple(range(rank)),
                    "rhs_batch": tuple(range(rank)),
                    "lhs_contract": tuple(range(2 * rank + 1, 3 * rank + 2)),
                    "rhs_contract": tuple(range(rank, 2 * rank + 1)),
                },
            )
            interleave = (
                rank,
                *(k for i in range(rank) for k in (rank + 1 + i, i)),
                2 * rank + 1,
            )
        # To (batch, reduced, phase, ..., feature): output position o is
        # reduced * n + phase.
        interleaved = Op(
            "transpose",
            [phased_product],
            dtype,
            tuple(phased_product.shape[k] for k in interleave),
            {"perm": interleave},
        )
        product = Op("reshape", [interleaved], dtype, (batch, *outputs, kernel.shape[-1]), {})
    # The product is (batch, spatial..., feature); the result's layout may differ.
    order = [0] * len(shape)
    order[d.output_batch_dimension] = 0
    order[d.output_feature_dimension] = rank + 1
    for k, o in enumerate(output_spatial):
        order[o] = 1 + k
    if order == list(range(len(shape))):
        return product
    return Op("transpose", [product], dtype, shape, {"perm": order})


def _summed_products(node: Op) -> Op:
    """A dot_general as a sum over the products of its operands, broadcast
    against each other: for integers and booleans, since the GEMM kernels are
    floating, and for contractions too short to be worth one."""
    lhs, rhs = node.operands
    a = node.attributes
    lb, rb, lc, rc = (a[key] for key in ("lhs_batch", "rhs_batch", "lhs_contract", "rhs_contract"))
    lf = tuple(d for d in range(len(lhs.shape)) if d not in lb + lc)
    rf = tuple(d for d in range(len(rhs.shape)) if d not in rb + rc)
    contracted = tuple(lhs.shape[d] for d in lc)
    full = (*node.shape, *contracted)
    batch, left, right = len(lb), len(lf), len(rf)

    def spread(x: Node, perm: tuple[int, ...], positions: list[int]) -> Op:
        if x.dtype != node.dtype:
            x = Op("convert", [x], node.dtype, x.shape, {})
        shape = tuple(x.shape[d] for d in perm)
        x = Op("transpose", [x], node.dtype, shape, {"perm": list(perm)})
        return Op("broadcast_in_dim", [x], node.dtype, full, {"dims": positions})

    last = list(range(batch + left + right, len(full)))
    products = Op(
        "and" if node.dtype == np.bool_ else "multiply",
        [
            spread(lhs, lb + lf + lc, [*range(batch + left), *last]),
            spread(
                rhs,
                rb + rf + rc,
                [*range(batch), *range(batch + left, batch + left + right), *last],
            ),
        ],
        node.dtype,
        full,
        {},
    )
    return Op(
        "reduce",
        [products, Constant(node.dtype, (), 0)],
        node.dtype,
        node.shape,
        {"dims": last, "reducer": "or" if node.dtype == np.bool_ else "add"},
    )


def _rng_bits(op, state: Node) -> list[Node]:
    """rng_bit_generator: the new state and the random bits.

    StableHLO leaves the generator to the implementation. This one is
    Threefry-2x32 keyed by the state, counting output elements; the next
    state is the state plus one in its first word.
    """
    if state.dtype not in (np.uint32, np.uint64) or len(state.shape) != 1:
        raise UnsupportedProgram(
            "metal: stablehlo.rng_bit_generator needs a uint32 or uint64 state"
        )
    dtype, shape = tensor_type(op.results[1])
    if dtype.kind not in "iu" or dtype.itemsize not in (4, 8):
        raise UnsupportedProgram(f"metal: stablehlo.rng_bit_generator of {dtype}")
    step = np.zeros(state.shape, state.dtype)
    step[:1] = 1
    advanced = Op(
        "add", [state, Constant(state.dtype, state.shape, None, step)], state.dtype, state.shape, {}
    )
    return [advanced, Op("rng_bits", [state], dtype, shape, {})]


def _scalar_body(values: list[Node], parameters: list[Parameter], what: str) -> None:
    """Reject a reducer or comparator body that is not scalar elementwise code."""
    for value in reachable(values):
        if (
            value.shape
            or (isinstance(value, Parameter) and value not in parameters)
            or (
                not isinstance(value, (Constant, Parameter))
                and (not isinstance(value, Op) or value.name not in SCALAR)
            )
        ):
            raise UnsupportedProgram(f"metal: {what} requires a scalar elementwise body")


_HOST_TARGET = "palladium_host"


def _host_callback(op, inputs: list[Node]) -> Host:
    """A `palladium_host` custom call: the function `palladium.host` registered
    under the call's index, run on the host."""
    from palladium import host

    a = op.attributes
    config = ir.DictAttr(a["mhlo.backend_config"])
    name, function = host.registered(ir.IntegerAttr(config["index"]).value)
    types = [tensor_type(value) for value in op.results]
    effect = "has_side_effect" in a and ir.BoolAttr(a["has_side_effect"]).value
    return host_call(name, function, inputs, types, effect)


def import_graph(
    module: ir.Module,
) -> tuple[ir.OpView, list[Parameter], list[Node], list[Host | While], int]:
    """The main function, and it as a DAG of nodes, calls and composites
    inlined: its parameters, results and the host calls that run for their
    effect, with the number of ops that took, inlined ones counted per call."""
    functions = {
        ir.StringAttr(op.attributes["sym_name"]).value: op
        for op in module.body.operations
        if op.operation.name == "func.func"
    }

    # Host calls with effects, per region being walked; calls inline into
    # their caller's.
    effects: list[list[Host | While]] = [[]]
    # The `case` branches around the op being walked, and the host calls seen.
    guards: list[Guard] = []
    hosts = 0

    def region(block, args: list[Node], outer: dict, what: str) -> list[Node]:
        """The results of a region that may not call the host for an effect."""
        effects.append([])
        results = walk(block, args, outer)
        if effects.pop():
            raise UnsupportedProgram(f"metal: a host callback with effects inside {what}")
        return results

    def walk(block, args: list[Node], outer: dict | None = None) -> list[Node]:
        env: dict[ir.Value, Node] = dict(outer or {})
        env.update(zip(block.arguments, args, strict=True))
        nonlocal ops, hosts
        for op in block.operations:
            ops += 1
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
                # Parse every branch: unsupported operations must still fail
                # explicitly, even when a constant selector would skip them.
                index = env[op.operands[0]]
                branches = []
                for k, r in enumerate(op.regions):
                    guards.append((index, k, len(op.regions)))
                    branches.append(walk(r.blocks[0], [], env))
                    guards.pop()
                for k, value in enumerate(op.results):
                    env[value] = Op(
                        "case", [index, *(b[k] for b in branches)], *tensor_type(value), {}
                    )
                continue
            if name == "stablehlo.while":
                inputs = [env[v] for v in op.operands]
                parameters = [Parameter(*tensor_type(v)) for v in op.operands]
                # Replace lexical values by parameters, then keep only used captures.
                captures = {
                    v: n if isinstance(n, Constant) else Parameter(n.dtype, n.shape)
                    for v, n in env.items()
                }
                # Inside, the loop's own conditionals guard; the ones around
                # it guard the loop as a whole.
                around, before = guards[:], hosts
                guards.clear()
                [condition] = region(
                    op.regions[0].blocks[0], list(parameters), captures, "a loop condition"
                )
                effects.append([])
                body = walk(op.regions[1].blocks[0], list(parameters), captures)
                body_effects = effects.pop()
                guards[:] = around
                roots = [condition, *body, *(x for host in body_effects for x in host.inputs)]
                used = set(reachable(roots))
                for value, parameter in captures.items():
                    if isinstance(parameter, Parameter) and parameter in used:
                        parameters.append(parameter)
                        inputs.append(env[value])
                inductions = _inductions(parameters, inputs, body)
                trips = _trip_count(condition, parameters, inputs, body, inductions)
                if trips is None:
                    inductions = {}
                loop = While(inputs, parameters, body, trips, inductions, condition, body_effects)
                if trips is None or hosts > before:
                    loop.guards = tuple(guards)
                results: list[Node] = [
                    LoopResult(loop, i, *tensor_type(value)) for i, value in enumerate(op.results)
                ]
                env.update(zip(op.results, results, strict=True))
                if body_effects:
                    # The loop runs for its effects even if no carry is read.
                    effects[-1].append(loop)
                continue
            if name == "stablehlo.reduce" and (
                # Any body, folded in order by one thread per output; so are
                # 64-bit integers, which the SIMD-group collectives lack.
                _reducer(op) is None
                or tensor_type(op.results[0])[0].itemsize == 8
                or (tensor_type(op.results[0])[0] == np.bool_ and _reducer(op) == "xor")
            ):
                count = len(op.results)
                block = op.regions[0].blocks[0]
                params = [Parameter(*tensor_type(v)) for v in block.arguments]
                body = region(block, list(params), env, "a reducer")
                _scalar_body(body, params, "stablehlo.reduce")
                inputs = [env[v] for v in op.operands]
                reduction = VariadicReduce(
                    inputs[:count],
                    inputs[count:],
                    params,
                    body,
                    tuple(ir.DenseI64ArrayAttr(op.attributes["dimensions"])),
                )
                for k, value in enumerate(op.results):
                    env[value] = ReductionResult(reduction, k, *tensor_type(value))
                continue
            if name in ("stablehlo.reduce_window", "stablehlo.scatter"):
                # The body: a known binary op, or a `Reducer` of its own
                # (scatter also assigns, when the body returns the update).
                block = op.regions[0].blocks[0]
                kind: str | Reducer | None = _reducer(op)
                body_ops = list(block.operations)
                if len(op.results) != 1:
                    raise UnsupportedProgram(f"metal: {name} of several operands")
                if (
                    name == "stablehlo.scatter"
                    and len(body_ops) == 1
                    and list(body_ops[0].operands) == [block.arguments[1]]
                ):
                    kind = "set"
                if kind is None:
                    params = [Parameter(*tensor_type(v)) for v in block.arguments]
                    [combined] = region(block, list(params), env, "a reducer")
                    _scalar_body([combined], params, name)
                    kind = Reducer(params, combined)
                operands = [env[v] for v in op.operands]
                dtype, shape = tensor_type(op.result)
                if name == "stablehlo.scatter":
                    attributes = {**_attributes(op), "update": kind}
                    env[op.result] = Op("scatter", operands, dtype, shape, attributes)
                else:
                    attributes = {**_window(op), "reducer": kind}
                    env[op.result] = _cumulative(*operands, attributes, shape) or Op(
                        "reduce_window", operands, dtype, shape, attributes
                    )
                continue
            if name == "stablehlo.sort":
                inputs = [env[v] for v in op.operands]
                block = op.regions[0].blocks[0]
                params = [Parameter(*tensor_type(v)) for v in block.arguments]
                [before] = region(block, list(params), env, "a comparator")
                _scalar_body([before], params, name)
                rank = len(inputs[0].shape)
                dim = ir.IntegerAttr(op.attributes["dimension"]).value % max(rank, 1)
                sort = Sort(inputs, params, before, dim)
                for k, value in enumerate(op.results):
                    env[value] = SortResult(sort, k, *tensor_type(value))
                continue
            if name == "stablehlo.rng_bit_generator":
                env.update(zip(op.results, _rng_bits(op, env[op.operands[0]]), strict=True))
                continue
            if name == "stablehlo.create_token":
                env[op.result] = Constant(np.dtype(np.bool_), (0,), 0)
                continue
            if name == "stablehlo.convolution":
                env[op.result] = _convolution(op, env[op.operands[0]], env[op.operands[1]])
                continue
            if name in ("sdy.sharding_constraint", "stablehlo.optimization_barrier"):
                # Identities on values: one device has nothing to shard, and
                # the barrier (from jax.checkpoint) only orders XLA's passes.
                env.update(zip(op.results, (env[v] for v in op.operands), strict=True))
                continue
            if name == "stablehlo.custom_call":
                target = ir.StringAttr(op.attributes["call_target_name"]).value
                operands = [env[v] for v in op.operands]
                if target == _HOST_TARGET:
                    host = _host_callback(op, operands)
                    host.guards = tuple(guards)
                    hosts += 1
                    if host.effect:
                        effects[-1].append(host)
                    results = host.results
                else:
                    results = _dispatch(op, operands).results
                env.update(zip(op.results, results, strict=True))
                continue
            short = name.removeprefix("stablehlo.")
            if name == "stablehlo.constant":
                env[op.result] = _constant(op)
            elif name.startswith("stablehlo.") and short in _SUPPORTED:
                attributes = _attributes(op)
                dtype, shape = tensor_type(op.result)
                operands = [env[v] for v in op.operands]
                node = Op(short, operands, dtype, shape, attributes)
                if short == "dot_general" and dtype.kind in "iub":
                    node = _summed_products(node)
                env[op.result] = node
            else:
                raise UnsupportedProgram(f"metal: {name} is not supported yet")
        raise UnsupportedProgram("metal: function without a return")

    ops = 0
    main = functions["main"]
    block = main.regions[0].blocks[0]
    parameters = [Parameter(*tensor_type(v)) for v in block.arguments]
    results = walk(block, list(parameters))
    return main, parameters, results, effects[0], ops


def in_place_carries(loop: While) -> dict[int, Op]:
    """Carries updated only by a dynamic_update_slice of themselves.

    Such a carry can live in one buffer for the whole loop: each iteration
    writes just the update. Safe only if nothing else in the body reads the
    old carry or the updated value, so no read can observe a partial write.
    """
    uses = collections.Counter(
        dependency for node in reachable(loop.results) for dependency in dependencies(node)
    )
    uses.update(result for result in loop.results)
    carries = {}
    for k, result in enumerate(loop.results):
        parameter = loop.parameters[k]
        if (
            isinstance(result, Op)
            and result.name == "dynamic_update_slice"
            and result.operands[0] is parameter
            and uses[result] == 1
            and uses[parameter] == 1
            and math.prod(result.operands[1].shape)
        ):
            carries[k] = result
    return carries
