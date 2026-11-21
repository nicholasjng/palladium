"""Code generation for one fused kernel."""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Callable, Collection, Sequence

import numpy as np

from palladium.emit.numeric import FMOD, extremum_identity, format_scalar
from palladium.errors import EmitError
from palladium.stablehlo.graph import (
    WINDOW_LIMIT,
    Buffer,
    Constant,
    DispatchResult,
    HostResult,
    LoopResult,
    Node,
    Op,
    Parameter,
    Reducer,
    ReductionResult,
    Sort,
    SortResult,
    VariadicReduce,
    is_looped,
    is_short,
    reachable,
)
from palladium.stablehlo.program import UnsupportedProgram
from palladium.stablehlo.scalar import (
    COMPARISONS,
    ELEMENTWISE,
    EXTRA,
    NARROW,
    THREEFRY,
    TOTAL_ORDER,
    accumulator,
    ctype_of,
    literal,
    narrow_scalar,
    wide_scalar,
)
from palladium.stablehlo.term import ZERO, Coords, Term, delinear, linear

I = Term.name("i")  # a thread's flat index


def index_positions(
    coords: Sequence[Term], window: Sequence[int], vector: int, rank: int, components: int
) -> tuple[Coords, list[Coords]]:
    """Where a gather result or scatter update at `coords` finds its index.

    Its coordinates outside the `window` dimensions select the index vector,
    which lies along dimension `vector` of the rank-`rank` index tensor.
    Returns that position, zero along `vector`, and the coordinates of each of
    the vector's `components`.
    """
    batch = iter(c for d, c in enumerate(coords) if d not in window)
    position = tuple(ZERO if d == vector else next(batch) for d in range(rank))
    return position, [
        tuple(Term.constant(k) if d == vector else c for d, c in enumerate(position))
        for k in range(components)
    ]


@dataclasses.dataclass
class Scope:
    """Memoized values of the kernel body or of one guarded branch.

    `names` are the variables the scope defines (temporaries and guarded
    coordinates). `insert` is the index in `lines`, the list the branch chain
    was emitted into, just before the chain: values hoisted out of this
    branch go there.
    """

    lines: list[str] = dataclasses.field(default_factory=list)
    insert: int = -1
    hoist: bool = True  # may values leave this branch?
    names: set[str] = dataclasses.field(default_factory=set)
    values: dict[tuple[Node, Coords], Term] = dataclasses.field(default_factory=dict)
    # The flat index of a buffer read at coordinates a reshape produced.
    reshaped: dict[tuple[Coords, tuple[int, ...]], Term] = dataclasses.field(default_factory=dict)
    indices: dict[Term, Term] = dataclasses.field(default_factory=dict)  # expression to name

    def forget(self) -> None:
        """Drop what was memoized, when the block that defined it closes."""
        self.values.clear()
        self.reshaped.clear()
        self.indices.clear()


class Kernel:
    """Generates one kernel computing the outputs of one shape."""

    def __init__(
        self,
        shape: tuple[int, ...],
        stored: set[Node] | frozenset[Node] = frozenset(),
        store: Callable[[Node], None] | None = None,
    ):
        self.shape = shape
        self.stored = stored  # ops already in buffers: read, not recomputed
        # Puts an op in a buffer (adding it to `stored`) before this kernel runs.
        self.store = store
        self.lines: list[str] = []
        self.helpers: dict[str, str] = {}
        self.buffers: dict[Node, int] = {}  # read nodes, bound in this order before the outputs
        self.scopes = [Scope()]  # the body, then open guarded branches
        # Every open branch, including those a hoist in progress has set aside.
        self.branches: list[Scope] = []
        self.next_value = 0
        # Expressions the coordinate variables stand for, so reads spelled
        # either way memoize together. A unit dimension's coordinate is zero.
        self.named: dict[Term, Term] = {
            e: Term.name(f"c{k}") for k, e in enumerate(delinear(I, shape)) if e.value is None
        }
        self.coords = self.coords_of(shape)
        self.lines += [f"uint {c} = {e.text};" for e, c in self.named.items()]

    def coords_of(self, shape: tuple[int, ...]) -> Coords:
        """Coordinates in `shape` of this thread's flat index.

        Expressions the kernel already names (its own coordinates) use those
        names, so a value read through another shape of equal size, such as
        an added unit dimension, hits the same memoized values.
        """
        return tuple(self.named.get(e, e) for e in delinear(I, shape))

    def require(self, name: str, source: str) -> None:
        self.helpers[name] = source

    def buffer(self, node: Node) -> str:
        return f"b{self.buffers.setdefault(node, len(self.buffers))}"

    @property
    def body(self) -> Scope:
        """The scope of the kernel body, outside any guarded branch."""
        return self.scopes[0]

    def _level(self, coords: Coords) -> int:
        """The outermost open scope where a value at `coords` may be computed.

        Coordinates that use none of a branch's own names are computed from
        enclosing coordinates by index maps, which stay in bounds without the
        branch's guard; such a value can be computed once before the chain.
        """
        if len(self.scopes) == 1:
            return 0
        used = frozenset().union(*(c.names for c in coords))
        for depth in range(len(self.scopes) - 1, 0, -1):
            scope = self.scopes[depth]
            if not scope.hoist or used & scope.names:
                return depth
        return 0

    def value(self, node: Node, coords: Coords) -> Term:
        key = (node, coords)
        for scope in reversed(self.scopes):
            if (found := scope.values.get(key)) is not None:
                return found
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
            self._insert_before(inner[0], hoisted)
            return result
        scope = self.scopes[-1]
        result = self._expression(node, coords)
        if not isinstance(result, Term):
            # An expression gets a variable; a term stands for itself.
            name = self.temporary()
            ctype = ctype_of(node.dtype)
            if ctype == "bfloat":
                # MSL's math functions return float for bfloat operands, and
                # bfloat has no implicit conversion from float.
                result = f"bfloat({result})"
            self.lines.append(f"{ctype} {name} = {result};")
            result = name
        scope.values[key] = result
        return result

    def _insert_before(self, branch: Scope, lines: list[str]) -> None:
        """Put `lines` just before the chain of `branch`.

        The chain may be in another list than the current one: a value
        generated for a hoist can itself hoist further out.
        """
        target, index = branch.lines, branch.insert
        target[index:index] = lines
        for scope in self.branches:
            if scope.lines is target and scope.insert >= index:
                scope.insert += len(lines)

    def _named_index(self, expr: Term) -> Term:
        """A variable holding the index `expr`, defined once, as far out as
        values at it may be computed, so they still hoist and memoize together."""
        if expr.simple:
            return expr
        level = self._level((expr,))
        for scope in self.scopes[: level + 1]:
            if expr in scope.indices:
                return scope.indices[expr]
        name = self._variable("k", self.scopes[level])
        self.scopes[level].indices[expr] = name
        line = f"uint {name} = {expr.text};"
        if level == len(self.scopes) - 1:
            self.lines.append(line)
        else:
            self._insert_before(self.scopes[level + 1], [line])
        return name

    def _variable(self, prefix: str, scope: Scope) -> Term:
        """A new variable name, owned by `scope`."""
        name = f"{prefix}{self.next_value}"
        self.next_value += 1
        scope.names.add(name)
        return Term.name(name)

    def temporary(self) -> Term:
        return self._variable("v", self.scopes[-1])

    def choose(
        self,
        dtype: np.dtype,
        branches: list[tuple[str, Node, Coords, Collection[int]]],
        hoist: bool = True,
    ) -> Term:
        """Emit each branch's reads inside its guard; the final branch is unconditional.

        A branch is (condition, node, coordinates, guarded dimensions). Its
        coordinates in the guarded dimensions are in bounds only under the
        guard: they become the branch's own names, like its temporaries, so
        values at them never escape to siblings. The others index the node
        within bounds anyway. With `hoist`, a value that uses no branch name
        is computed once before the chain instead of in every branch. Without
        it (`case`), branches stay lazy.
        """
        result = self.temporary()
        self.lines.append(f"{ctype_of(dtype)} {result};")
        insert = len(self.lines)
        for k, (condition, node, coords, guarded) in enumerate(branches):
            last = k == len(branches) - 1
            prefix = "" if k == 0 else "else "
            self.lines.append(prefix + ("{" if last else f"if ({condition}) {{"))
            scope = Scope(self.lines, insert, hoist)
            self.scopes.append(scope)
            self.branches.append(scope)
            try:
                bound = list(coords)
                for j in guarded:
                    if coords[j].value is None:
                        bound[j] = self._variable("k", scope)
                        self.lines.append(f"uint {bound[j]} = {coords[j].text};")
                value = self.value(node, tuple(bound))
                self.lines.append(f"{result} = {value};")
            finally:
                self.scopes.pop()
                self.branches.pop()
            insert = scope.insert
            self.lines.append("}")
        return result

    def offsets(self, starts: list[Node], limits: tuple[int, ...]) -> Coords:
        """The start indices of a dynamic slice, clamped so it stays inside its operand."""
        offsets = []
        for start, limit in zip(starts, limits, strict=True):
            # Clamped in the start's own type, widened: unsigned values above
            # INT_MAX stay large, and the limit may not fit 8 bits.
            ctype = accumulator(ctype_of(start.dtype))
            value = self.value(start, ())
            clamped = f"uint(clamp({ctype}({value}), {ctype}(0), {ctype}({limit})))"
            offsets.append(Term.call(clamped, value))
        return tuple(offsets)

    def _flat(self, coords: Coords, shape: tuple[int, ...]) -> Term:
        return I if coords == self.coords and shape == self.shape else linear(coords, shape)

    def _read(self, node: Node, coords: Coords) -> str:
        key = (coords, node.shape)
        flat = next((s.reshaped[key] for s in reversed(self.scopes) if key in s.reshaped), None)
        return f"{self.buffer(node)}[{(flat or self._flat(coords, node.shape)).text}]"

    def _expression(self, node: Node, coords: Coords) -> str | Term:
        """`node` at `coords`: an expression, or the term it is already spelled as."""
        ctype = ctype_of(node.dtype)
        if node in self.stored or isinstance(
            node,
            (
                Parameter,
                Buffer,
                LoopResult,
                ReductionResult,
                DispatchResult,
                HostResult,
                SortResult,
            ),
        ):
            return self._read(node, coords)
        if isinstance(node, Constant):
            if node.splat is None:
                return self._read(node, coords)
            return Term(literal(node.splat, ctype))
        assert isinstance(node, Op)
        a, operands = node.attributes, node.operands
        match node.name:
            case "gather":
                x, indices = operands
                index_coords, components = index_positions(
                    coords, a["offset"], a["vector"], len(indices.shape), len(a["map"])
                )
                inner = [ZERO] * len(x.shape)
                for dim, read_coords in zip(a["map"], components, strict=True):
                    start = self.value(indices, read_coords)
                    itype = accumulator(ctype_of(indices.dtype))  # the limit may not fit 8 bits
                    limit = x.shape[dim] - a["sizes"][dim]
                    clamped = f"uint(clamp({itype}({start}), {itype}(0), {itype}({limit})))"
                    inner[dim] = Term.call(clamped, start)
                for dim, index_dim in zip(a["batch"], a["index_batch"], strict=True):
                    inner[dim] = index_coords[index_dim]
                window_dims = [
                    d for d in range(len(x.shape)) if d not in a["collapsed"] + a["batch"]
                ]
                for dim, offset in zip(window_dims, a["offset"], strict=True):
                    inner[dim] += coords[offset]
                return self.value(x, tuple(inner))
            case "conv_patches":
                return self._conv_patches(node, coords)
            case "conv_phases":
                return self._conv_phases(node, coords)
            case "conv_taps":
                [kernel] = operands
                rank = len(a["lhs_dilation"])
                inner = []
                for k, n in enumerate(a["lhs_dilation"]):
                    phase, tap = coords[k], coords[rank + k]
                    # The tap of this phase's residue that `tap` counts from.
                    first = (a["padding"][k][0] % n + n - phase) % n
                    inner.append(tap if n == 1 else first + tap * n)
                return self.value(kernel, (*inner, *coords[2 * rank :]))
            case "reduce_window" | "select_and_scatter" | "reduce" if (
                node.name != "reduce" or is_short(node)
            ):
                if self.store is not None and len(self.scopes) > 1:
                    # Each branch that reads a fold would regenerate it,
                    # producers and all, under its own coordinate names:
                    # store it once instead.
                    self.store(node)
                    return self._read(node, coords)
                fold = {
                    "reduce": self._fold_reduction,
                    "reduce_window": self._reduce_window,
                    "select_and_scatter": self._select_and_scatter,
                }[node.name]
                return fold(node, coords)
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
                    [
                        (f"{selector} == {k}", branch, coords, ())
                        for k, branch in enumerate(branches)
                    ],
                    hoist=False,
                )
            case "concatenate":
                dim, offset, branches = a["dim"], 0, []
                position = coords[dim].value
                for x in operands:
                    size = x.shape[dim]
                    inner = (*coords[:dim], coords[dim] - offset, *coords[dim + 1 :])
                    if position is not None and position < offset + size:
                        # A constant coordinate selects its operand statically.
                        return self.value(x, inner)
                    if size:
                        # Operands match the output in every other dimension.
                        branches.append((f"{coords[dim]} < {offset + size}u", x, inner, (dim,)))
                    offset += size
                # No thread runs for an empty output, but its source must be valid.
                if not branches:
                    return Term(literal(0, ctype))
                return self.choose(node.dtype, branches)
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
                    inner.append(Term.call(f"uint({shifted} / {step})", c))
                return self.choose(
                    node.dtype,
                    [
                        (" && ".join(conditions) or "true", x, tuple(inner), range(len(inner))),
                        ("", padding, (), ()),
                    ],
                )
            case "dynamic_slice":
                x, *starts = operands
                limits = tuple(n - s for n, s in zip(x.shape, node.shape, strict=True))
                offsets = self.offsets(starts, limits)
                return self.value(x, tuple(c + s for c, s in zip(coords, offsets, strict=True)))
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
                inner = tuple(c - s for c, s in zip(coords, offsets, strict=True))
                # The operand has the output's shape: its reads are in bounds unguarded.
                return self.choose(
                    node.dtype, [(condition, update, inner, range(len(inner))), ("", x, coords, ())]
                )
            case "broadcast_in_dim":
                [x] = operands
                inner = tuple(
                    ZERO if n == 1 else coords[d] for n, d in zip(x.shape, a["dims"], strict=True)
                )
                return self.value(x, inner)
            case "reshape":
                [x] = operands
                flat = self._flat(coords, node.shape)
                if sum(n != 1 for n in x.shape) > 1:
                    # Each coordinate repeats the flat index: named, a chain
                    # of reshapes grows by a line each, not by the rank each.
                    # A single coordinate is the index itself and keeps its
                    # spelling, to memoize with other reads.
                    flat = self._named_index(flat)
                inner = self.coords_of(x.shape) if flat is I else delinear(flat, x.shape)
                self.scopes[-1].reshaped[(inner, x.shape)] = flat
                return self.value(x, inner)
            case "transpose":
                [x] = operands
                inner = list(coords)
                for k, p in enumerate(a["perm"]):
                    inner[p] = coords[k]
                return self.value(x, tuple(inner))
            case "slice":
                [x] = operands
                return self.value(
                    x,
                    tuple(
                        c * t + s for c, s, t in zip(coords, a["start"], a["strides"], strict=True)
                    ),
                )
            case "reverse":
                [x] = operands
                return self.value(
                    x,
                    tuple(
                        n - 1 - c if k in a["dims"] else c
                        for k, (c, n) in enumerate(zip(coords, node.shape, strict=True))
                    ),
                )
            case "iota":
                return f"{ctype}({coords[a['dim']].text})"
            case "fold_blocks" | "fold_before" | "fold_within":
                return self._block_fold(node, coords)
            case "rng_bits":
                return self._rng_bits(node, coords)
        if node.name == "bitcast_convert" and node.dtype.itemsize != operands[0].dtype.itemsize:
            return self._regrouped_bits(node, coords)
        # A scalar operand of a tensor op (select's predicate, clamp's bounds)
        # applies to every element.
        values = [str(self.value(x, coords if x.shape else ())) for x in operands]
        match node.name:
            case "convert":
                return f"{ctype}({values[0]})"
            case "bitcast_convert":
                return f"as_type<{ctype}>({values[0]})"
            case "reduce_precision":
                # A round trip through the narrower format; same-format is a no-op.
                formats = {(5, 10): "half", (8, 7): "bfloat", (8, 23): "float"}
                narrow = formats.get((a["exponent"], a["mantissa"]))
                if narrow is None or ctype not in formats.values():
                    raise UnsupportedProgram("metal: stablehlo.reduce_precision to this format")
                wider = ctype == "float" and narrow != "float"
                return f"{ctype}({narrow}({values[0]}))" if wider else values[0]
            case "compare" if a["total"] and operands[0].dtype.kind == "f":
                self.require(*TOTAL_ORDER)
                ranks = [f"total_order(float({v}))" for v in values]
                return f"({ranks[0]} {COMPARISONS[a['direction']]} {ranks[1]})"
            case "compare":
                # JAX spells isnan as x != x, which RELAXED and FAST math fold
                # to false; isnan survives every math mode.
                if (
                    operands[0] is operands[1]
                    and operands[0].dtype.kind == "f"
                    and a["direction"] in ("ne", "eq")
                ):
                    return f"{'' if a['direction'] == 'ne' else '!'}isnan({values[0]})"
                return format_scalar(a["direction"], ctype, tuple(values))
            case "select":
                pred, on_true, on_false = values
                return format_scalar("select_n", ctype, (pred, on_false, on_true))
        operand_ctype = ctype_of(operands[0].dtype)
        if operand_ctype in ("long", "ulong"):
            return wide_scalar(node.name, operand_ctype, values)
        if operand_ctype in NARROW:
            return narrow_scalar(node.name, operand_ctype, values, self)
        if node.name in EXTRA:
            kind = "float" if operand_ctype in ("float", "half", "bfloat") else operand_ctype
            if kind not in EXTRA[node.name]:
                raise UnsupportedProgram(f"metal: stablehlo.{node.name} on {operand_ctype}")
            if node.name == "remainder" and kind == "float":
                self.require(*FMOD)
            expr = EXTRA[node.name][kind].format(**dict(zip("abc", values, strict=False)))
            # Templates that compute in float narrow explicitly: bfloat has no
            # implicit conversion from float.
            return expr if operand_ctype in ("float", kind) else f"{operand_ctype}({expr})"
        try:
            return format_scalar(ELEMENTWISE[node.name], operand_ctype, tuple(values), self)
        except EmitError as e:
            raise UnsupportedProgram(f"metal: stablehlo.{node.name}: {e}") from e

    def _fold_reduction(self, node: Op, coords: Coords) -> str:
        """A short reduction as a left fold from its initial value, unrolled,
        in the order XLA's sequential reduce takes."""
        x, init = node.operands
        dims = sorted(node.attributes["dims"])
        ctype = ctype_of(node.dtype)
        acc_type = accumulator(ctype)
        acc = f"{acc_type}({self.value(init, ())})"
        for index in itertools.product(*(range(x.shape[d]) for d in dims)):
            fixed, outer = iter(index), iter(coords)
            inner = tuple(
                Term.constant(next(fixed)) if d in dims else next(outer)
                for d in range(len(x.shape))
            )
            acc = self._fold_step(node, acc_type, acc, str(self.value(x, inner)))
        return f"{ctype}({acc})"

    def _fold_step(self, node: Op, acc_type: str, acc: str, value: str) -> str:
        """A variable holding `acc` combined with `value` by the reducer of
        `node`; the extremum templates repeat their operands."""
        name = self.temporary()
        reducer = ELEMENTWISE[node.attributes["reducer"]]
        step = format_scalar(reducer, acc_type, (acc, f"{acc_type}({value})"))
        self.lines.append(f"{acc_type} {name} = {step};")
        return name.text

    def _guarded(
        self,
        x: Node,
        coords: Sequence[Term],
        positions: dict[int, str],
        dilations: Sequence[int] | None = None,
    ) -> tuple[str, Term]:
        """Read `x` at positions that may fall outside it.

        Dimensions in `positions` hold int expressions in base-dilated
        coordinates, divided by `dilations[d]`; the others take `coords[d]`
        as is. Returns the condition that every position is a real element,
        and the value read at positions clamped into `x`, which is safe to
        compute either way.
        """
        conditions, inner = [], list(coords)
        for d, p in positions.items():
            n, step = x.shape[d], dilations[d] if dilations else 1
            q = self.temporary()
            self.lines.append(f"int {q} = {p};")
            conditions.append(f"{q} >= 0")
            element = str(q)
            if step != 1:
                conditions.append(f"{q} % {step} == 0")
                element = f"({q} / {step})"
            conditions.append(f"{element} < {n}")
            inner[d] = Term.call(f"uint(clamp({element}, 0, {n - 1}))", q)
        return " && ".join(conditions) or "true", self.value(x, tuple(inner))

    def _conv_patches(self, node: Op, coords: Coords) -> str:
        """Element (batch, output spatial..., window..., feature) of a convolution's patches."""
        [x] = node.operands
        a = node.attributes
        rank = len(a["spatial"])
        plain = [ZERO] * len(x.shape)
        plain[a["batch"]], plain[a["feature"]] = coords[0], coords[-1]
        positions, dilations = {}, [1] * len(x.shape)
        for k, dim in enumerate(a["spatial"]):
            out, w = coords[1 + k], coords[1 + rank + k]
            if a["reversal"][k]:
                w = node.shape[1 + rank + k] - 1 - w
            positions[dim] = (
                f"int({out}) * {a['strides'][k]} + int({w}) * {a['rhs_dilation'][k]}"
                f" - {a['padding'][k][0]}"
            )
            dilations[dim] = a["lhs_dilation"][k]
        return self._patch(node, plain, positions, dilations)

    def _conv_phases(self, node: Op, coords: Coords) -> str:
        """Element (phase..., batch, reduced output..., tap..., feature) of a
        convolution whose input is dilated; see `graph._convolution`."""
        [x] = node.operands
        a = node.attributes
        rank = len(a["spatial"])
        if a.get("shared"):
            # Every phase reads these patches; phase 0 stands for all.
            coords = (ZERO,) * rank + coords
        plain = [ZERO] * len(x.shape)
        plain[a["batch"]], plain[a["feature"]] = coords[rank], coords[-1]
        positions = {}
        for k, dim in enumerate(a["spatial"]):
            phase, out, tap = coords[k], coords[rank + 1 + k], coords[2 * rank + 1 + k]
            n, low = a["lhs_dilation"][k], a["padding"][k][0]
            if n == 1:
                if a["reversal"][k]:
                    tap = node.shape[len(node.shape) - 1 - rank + k] - 1 - tap
                positions[dim] = (
                    f"int({out}) * {a['strides'][k]} + int({tap}) * {a['rhs_dilation'][k]} - {low}"
                )
                continue
            # The phase's first tap, and where its first tap lands in the input.
            first = f"(({low % n + n} - int({phase})) % {n})"
            positions[dim] = f"int({out}) + int({tap}) + (int({phase}) + {first} - {low}) / {n}"
        return self._patch(node, plain, positions)

    def _patch(
        self,
        node: Op,
        coords: Sequence[Term],
        positions: dict[int, str],
        dilations: Sequence[int] | None = None,
    ) -> str:
        """An element of a convolution's patches: its input where `_guarded`
        finds a real element, zero on padding and between dilated elements."""
        [x] = node.operands
        if self.store is not None and isinstance(x, Op) and x not in self.stored:
            # Windows overlap and every patch would recompute the producers.
            self.store(x)
        condition, value = self._guarded(x, coords, positions, dilations)
        return f"({condition} ? {value} : {literal(0, ctype_of(node.dtype))})"

    def _loop_fold(
        self,
        dtype: np.dtype,
        reducer: str | Reducer,
        seed: str,
        start: str,
        end: str,
        element: Callable[[Term], str | Term],
    ) -> str:
        """`seed` combined, in a loop over [start, end), with `element` of the
        loop variable. A `Reducer` binds its body to the two values."""
        ctype = ctype_of(dtype)
        general = isinstance(reducer, Reducer)
        acc_type = ctype if general else accumulator(ctype)
        acc = self.temporary()
        self.lines.append(f"{acc_type} {acc} = {acc_type}({seed});")
        # A body's values are scalars without coordinates, which would look
        # loop-invariant: nothing leaves a loop that binds one.
        scope = Scope(self.lines, len(self.lines), hoist=not general)
        r = self._variable("r", scope)
        self.lines.append(f"for (uint {r} = {start}; {r} < {end}; ++{r}) {{")
        self.scopes.append(scope)
        self.branches.append(scope)
        try:
            value = element(r)
            if isinstance(reducer, Reducer):
                bound = self.temporary()
                self.lines.append(f"{ctype} {bound} = {value};")
                scope.values[(reducer.parameters[0], ())] = acc
                scope.values[(reducer.parameters[1], ())] = bound
                combined = str(self.value(reducer.body, ()))
            else:
                operands = (str(acc), f"{acc_type}({value})")
                combined = format_scalar(ELEMENTWISE[reducer], acc_type, operands)
            self.lines.append(f"{acc} = {combined};")
        finally:
            self.scopes.pop()
            self.branches.pop()
        self.lines.append("}")
        return f"{ctype}({acc})"

    def _stored_operand(self, x: Node) -> None:
        """Compute `x` into a buffer before a fold that reads each of its
        elements many times, where the kernel can."""
        if self.store is not None and isinstance(x, Op) and x not in self.stored:
            self.store(x)

    def _block_fold(self, node: Op, coords: Coords) -> str:
        """One element of a step of a cumulative reduction in blocks; see
        `graph._cumulative`."""
        a = node.attributes
        axis, block, reverse = a["axis"], a["block"], a["reverse"]
        x, other = node.operands
        n, c = x.shape[axis], coords[axis]
        self._stored_operand(x)

        def at(position: Term) -> Coords:
            return (*coords[:axis], position, *coords[axis + 1 :])

        if node.name == "fold_blocks":
            # The total of block c.
            seed, start, end = self.value(other, ()), c * block, f"min({(c + 1) * block}, {n}u)"
        elif node.name == "fold_before":
            # The total of the blocks before block c, or after it in reverse.
            seed = self.value(other, ())
            start, end = (c + 1, f"{n}u") if reverse else (ZERO, c)
        else:
            # Element c: what precedes its block, then its block up to it.
            first = c // block
            seed = self.value(other, at(first))
            start, end = (
                (c, f"min({(first + 1) * block}, {n}u)") if reverse else (first * block, c + 1)
            )
        return self._loop_fold(
            node.dtype,
            a["reducer"],
            str(seed),
            str(start),
            str(end),
            lambda r: self.value(x, at(r)),
        )

    def _rng_bits(self, node: Op, coords: Coords) -> str:
        """Random bits for one element: Threefry keyed by the state's words,
        on the element's flat index."""
        [state] = node.operands
        self.require(*THREEFRY)
        words = []
        for k in range(state.shape[0]):
            value = self.value(state, (Term.constant(k),))
            if state.dtype.itemsize == 8:
                words += [f"uint({value})", f"uint({value} >> 32)"]
            else:
                words.append(str(value))
        words += ["0u"] * (len(words) % 2)
        key = self.temporary()
        self.lines.append(f"uint2 {key} = uint2({words[0]}, {words[1]});")
        for k in range(2, len(words), 2):
            self.lines.append(f"{key} = threefry2x32({key}, uint2({words[k]}, {words[k + 1]}));")
        bits = self.temporary()
        flat = self._flat(coords, node.shape)
        self.lines.append(f"uint2 {bits} = threefry2x32({key}, uint2({flat.text}, 0u));")
        ctype = ctype_of(node.dtype)
        if node.dtype.itemsize == 8:
            return f"as_type<{ctype}>((ulong({bits}.y) << 32) | ulong({bits}.x))"
        return f"as_type<{ctype}>({bits}.x)"

    def _regrouped_bits(self, node: Op, coords: Coords) -> str:
        """A bitcast between element widths: the last dimension of the
        narrower side holds the parts of each wider element, lowest first."""
        [x] = node.operands
        unsigned = {1: "uchar", 2: "ushort", 4: "uint", 8: "ulong"}
        if np.bool_ in (node.dtype, x.dtype):
            raise UnsupportedProgram("metal: shape-changing stablehlo.bitcast_convert of booleans")
        ctype, wide, narrow = ctype_of(node.dtype), node.dtype.itemsize, x.dtype.itemsize
        if narrow > wide:
            whole = unsigned[narrow]
            bits = f"as_type<{whole}>({self.value(x, coords[:-1])})"
            part = f"{unsigned[wide]}({bits} >> {whole}({coords[-1]} * {8 * wide}u))"
            return f"as_type<{ctype}>({part})"
        whole = unsigned[wide]
        parts = [
            f"({whole}(as_type<{unsigned[narrow]}>({self.value(x, (*coords, Term.constant(k)))}))"
            f" << {8 * narrow * k}u)"
            for k in range(wide // narrow)
        ]
        return f"as_type<{ctype}>({' | '.join(parts)})"

    def _reduce_window(self, node: Op, coords: Coords) -> str:
        """A window of the base-dilated, padded input folded from the initial
        value; positions on padding or dilation holes contribute the initial
        value, as StableHLO pads with it. A window of at most `WINDOW_LIMIT`
        elements unrolls; a longer one, or a reducer body, folds in a loop."""
        x, init = node.operands
        a = node.attributes
        ctype = ctype_of(node.dtype)
        acc_type = accumulator(ctype)
        start = self.value(init, ())
        acc = f"{acc_type}({start})"
        # Dimensions the window does not move along read the output coordinate.
        moving = [
            d
            for d in range(len(x.shape))
            if (a["window"][d], a["strides"][d], a["base"][d], *a["padding"][d]) != (1, 1, 1, 0, 0)
        ]
        if is_looped(node):
            self._stored_operand(x)
            sizes = [a["window"][d] for d in moving]

            def element(r: Term) -> str | Term:
                positions = {
                    d: f"int({coords[d]}) * {a['strides'][d]} + int({e}) * {a['dilations'][d]}"
                    f" - {a['padding'][d][0]}"
                    for d, e in zip(moving, delinear(r, sizes), strict=True)
                }
                condition, value = self._guarded(x, coords, positions, a["base"])
                return value if condition == "true" else f"({condition} ? {value} : {start})"

            return self._loop_fold(
                node.dtype, a["reducer"], str(start), "0u", f"{math.prod(sizes)}u", element
            )
        for offsets in itertools.product(*(range(a["window"][d]) for d in moving)):
            positions = {
                d: f"int({coords[d]}) * {a['strides'][d]} + {e * a['dilations'][d]}"
                f" - {a['padding'][d][0]}"
                for d, e in zip(moving, offsets, strict=True)
            }
            condition, value = self._guarded(x, coords, positions, a["base"])
            element = str(value) if condition == "true" else f"({condition} ? {value} : {start})"
            acc = self._fold_step(node, acc_type, acc, element)
        return f"{ctype}({acc})"

    def _select_and_scatter(self, node: Op, coords: Coords) -> str:
        """One element of a select_and_scatter, gathered rather than scattered.

        For each window that contains this element (at most ceil(window /
        stride) per dimension), recompute the window's selection: the first
        in-bounds element, replaced by each later one that the comparison
        does not keep. Where the selection is this element, add the window's
        source value. Padding is never selected. Without atomics, the result
        is deterministic.
        """
        x, source, init = node.operands
        a = node.attributes
        window, strides, padding = a["window"], a["strides"], a["padding"]
        if any(low < 0 or high < 0 for low, high in padding):
            raise UnsupportedProgram("metal: stablehlo.select_and_scatter with negative padding")
        moving = [
            d for d in range(len(x.shape)) if (window[d], strides[d], *padding[d]) != (1, 1, 0, 0)
        ]
        candidates = [-(-window[d] // strides[d]) for d in moving]
        if math.prod(candidates) * math.prod(window) > WINDOW_LIMIT:
            raise UnsupportedProgram(
                f"metal: stablehlo.select_and_scatter reading over {WINDOW_LIMIT} elements"
            )
        ctype = ctype_of(node.dtype)
        acc_type = accumulator(ctype)
        operator = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<"}[a["direction"]]
        acc = self.temporary()
        self.lines.append(f"{acc_type} {acc} = {acc_type}({self.value(init, ())});")
        for choice in itertools.product(*(range(c) for c in candidates)):
            # This candidate window's index and this element's offset in it.
            starts, offsets, valid = {}, {}, []
            for d, c in zip(moving, choice, strict=True):
                w, low = self.temporary(), padding[d][0]
                self.lines.append(f"int {w} = int({coords[d]} + {low}u) / {strides[d]} - {c};")
                offsets[d] = f"(int({coords[d]}) + {low} - {w} * {strides[d]})"
                valid += [f"{w} >= 0", f"{w} < {source.shape[d]}", f"{offsets[d]} < {window[d]}"]
                starts[d] = w
            selected, best = self.temporary(), self.temporary()
            self.lines += [f"int {selected} = -1;", f"{acc_type} {best} = {acc_type}(0);"]
            for flat, element in enumerate(itertools.product(*(range(window[d]) for d in moving))):
                positions = {
                    d: f"{starts[d]} * {strides[d]} + {e} - {padding[d][0]}"
                    for d, e in zip(moving, element, strict=True)
                }
                condition, value = self._guarded(x, coords, positions)
                take, replaces = self.temporary(), f"!({best} {operator} {acc_type}({value}))"
                self.lines += [
                    f"bool {take} = ({condition}) && ({selected} < 0 || {replaces});",
                    f"{selected} = {take} ? {flat} : {selected};",
                    f"{best} = {take} ? {acc_type}({value}) : {best};",
                ]
            here = " + ".join(
                f"{offsets[d]} * {math.prod(window[e] for e in moving[k + 1 :])}"
                for k, d in enumerate(moving)
            )
            inner = tuple(
                Term.call(f"uint(clamp({starts[d]}, 0, {source.shape[d] - 1}))", starts[d])
                if d in starts
                else c
                for d, c in enumerate(coords)
            )
            value = self.value(source, inner)
            condition = " && ".join([*valid, f"{selected} == {here or '0'}"])
            self.lines.append(f"if ({condition}) {acc} += {acc_type}({value});")
        return f"{ctype}({acc})"

    def source(self, outputs: list[Node], name: str) -> str:
        stores = [f"o{k}[i] = {self.value(node, self.coords)};" for k, node in enumerate(outputs)]
        return self.render(outputs, name, stores)

    def render(self, outputs: list[Node], name: str, stores: list[str], index: str = "i") -> str:
        params = [
            f"device const {ctype_of(n.dtype)}* b{k} [[buffer({k})]]"
            for k, n in enumerate(self.buffers)
        ]
        params += [
            f"device {ctype_of(n.dtype)}* o{k} [[buffer({len(self.buffers) + k})]]"
            for k, n in enumerate(outputs)
        ]
        params.append(f"uint {index} [[thread_position_in_grid]]")
        body = "\n".join(f"    {line}" for line in self.lines + stores)
        helpers = "".join(f"{s}\n\n" for s in self.helpers.values())
        return (
            "#include <metal_stdlib>\nusing namespace metal;\n\n"
            f"{helpers}kernel void {name}(\n    " + ",\n    ".join(params) + f") {{\n{body}\n}}\n"
        )


def reduction_source(kernel: Kernel, node: Op, output: Node, name: str, chunks: int = 1) -> str:
    """128 threads per output; large reductions can write independent partials."""
    x, init = node.operands
    if node.dtype in (np.int64, np.uint64):
        raise UnsupportedProgram("metal: stablehlo.reduce on 64-bit integers is not supported")
    dims = sorted(node.attributes["dims"])
    reduced_shape = tuple(x.shape[d] for d in dims)
    size = math.prod(reduced_shape)
    kind = node.attributes["reducer"]
    ctype = ctype_of(node.dtype)
    acc_type = accumulator(ctype)
    if kind in ("add", "or", "xor"):
        identity = f"{acc_type}(0)"
    elif kind == "multiply":
        identity = "true" if acc_type == "bool" else f"{acc_type}(1)"
    elif kind == "and":
        identity = "true" if acc_type == "bool" else f"{acc_type}(~0u)"
    else:
        identity = extremum_identity(ELEMENTWISE[kind], acc_type)
    intrinsic = {
        "add": "sum",
        "multiply": "product",
        "maximum": "max",
        "minimum": "min",
        "and": "and",
        "or": "or",
        "xor": "xor",
    }[kind]
    if ctype == "bool":
        intrinsic = "all" if kind in ("and", "minimum", "multiply") else "any"

    def combine(a: str, b: str) -> str:
        return format_scalar(ELEMENTWISE[kind], acc_type, (a, b))

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

    kernel.lines.insert(0, "uint i = g / 128u;")
    kernel.lines += [f"{acc_type} acc = {identity};", "uint t = g % 128u;"]
    start = f"{kernel.coords[-1]} * 4096u + t" if chunks > 1 else "t"
    end = f"min(({kernel.coords[-1]} + 1u) * 4096u, {size}u)" if chunks > 1 else f"{size}u"
    # Zero-sized inputs have no read at all (nor zero divisors in coordinates).
    if size:
        kernel.lines.append(f"for (uint r = {start}; r < {end}; r += 128u) {{")
        red_coords = iter(delinear(Term.name("r"), reduced_shape))
        out_coords = iter(kernel.coords)
        coords = tuple(
            next(red_coords) if d in dims else next(out_coords) for d in range(len(x.shape))
        )
        value = kernel.value(x, coords)
        kernel.lines += [f"acc = {combine('acc', f'{acc_type}({value})')};", "}"]
        kernel.body.forget()  # the loop's variables are out of scope
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
    kernel.lines += [f"o0[i] = {ctype_of(output.dtype)}(acc);", "}", "}"]
    return kernel.render([output], name, [], index="g")


def variadic_reduction_source(
    kernel: Kernel, reduction: VariadicReduce, outputs: list[Node], name: str
) -> str:
    """One thread per output element folds the reducer body over the reduced
    elements in order, which keeps first-index ties and NaN rules."""
    accumulators = [Term.name(f"acc{k}") for k in range(len(outputs))]
    for acc, value in zip(accumulators, reduction.initial, strict=True):
        initial = kernel.value(value, ())
        kernel.lines.append(f"{ctype_of(value.dtype)} {acc} = {initial};")
    shape = reduction.inputs[0].shape
    dims = sorted(reduction.dims)
    reduced_shape = tuple(shape[d] for d in dims)
    size = math.prod(reduced_shape)
    if size:
        kernel.lines.append(f"for (uint r = 0u; r < {size}u; ++r) {{")
        red_coords = iter(delinear(Term.name("r"), reduced_shape))
        out_coords = iter(kernel.coords)
        coords = tuple(
            next(red_coords) if d in dims else next(out_coords) for d in range(len(shape))
        )
        values = [kernel.value(value, coords) for value in reduction.inputs]
        # Lexicographic left fold preserves JAX's first-index tie/NaN rule.
        # Bind the actual reducer body rather than guessing an argmax pattern.
        for param, value in zip(reduction.parameters, accumulators + values, strict=True):
            kernel.body.values[(param, ())] = value
        new_values = [kernel.value(value, ()) for value in reduction.body]
        # Snapshot all results before mutating any accumulator (parallel tuple update).
        snapshots = []
        for value, result in zip(new_values, outputs, strict=True):
            tmp = kernel.temporary()
            kernel.lines.append(f"{ctype_of(result.dtype)} {tmp} = {value};")
            snapshots.append(tmp)
        kernel.lines.extend(f"{a} = {v};" for a, v in zip(accumulators, snapshots, strict=True))
        kernel.lines.append("}")
    stores = [f"o{k}[i] = {v};" for k, v in enumerate(accumulators)]
    return kernel.render(outputs, name, stores)


def pack_word(ctype: str, value: str) -> str:
    """`value` as a 32-bit word, the unit Metal's atomics work on."""
    if ctype in ("half", "bfloat"):
        return f"uint(as_type<ushort>({value}))"
    return f"as_type<uint>({value})" if ctype in ("float", "int") else f"uint({value})"


def unpack_word(ctype: str, word: str) -> str:
    """The element of `ctype` that `pack_word` put in `word`."""
    if ctype in ("half", "bfloat"):
        return f"as_type<{ctype}>(ushort({word}))"
    if ctype == "bool":
        return f"({word} != 0u)"
    return f"as_type<{ctype}>({word})" if ctype in ("float", "int") else f"{ctype}({word})"


def words_source(kernel: Kernel, x: Node, words: Node, name: str) -> str:
    """Each element of `x` in a 32-bit word of its own."""
    value = kernel.value(x, kernel.coords)
    return kernel.render([words], name, [f"o0[i] = {pack_word(ctype_of(x.dtype), str(value))};"])


def unpacked_source(kernel: Kernel, words: Node, output: Node, name: str) -> str:
    """The elements held in the 32-bit `words`."""
    value = kernel.value(words, kernel.coords)
    return kernel.render(
        [output], name, [f"o0[i] = {unpack_word(ctype_of(output.dtype), str(value))};"]
    )


def scatter_source(kernel: Kernel, node: Op, target: Node, name: str) -> str:
    """One thread per update element combines it into `target`, where its
    index lies inside the operand.

    An assignment stores. An addition of 32-bit elements is an atomic add.
    Any other update is a compare-and-exchange loop on the element's 32-bit
    word; `target` then holds one word per element for narrower types.
    Updates of one element apply in no particular order.
    """
    _, indices, updates = node.operands
    words = target is not node
    update = node.attributes["update"]
    a = node.attributes
    index_coords, components = index_positions(
        kernel.coords, a["window"], a["vector"], len(indices.shape), len(a["map"])
    )
    inner = ["0l"] * len(node.shape)
    for dim, read_coords in zip(a["map"], components, strict=True):
        inner[dim] = f"long({kernel.value(indices, read_coords)})"
    for dim, index_dim in zip(a["batch"], a["index_batch"], strict=True):
        inner[dim] = f"long({index_coords[index_dim]})"
    window_dims = [d for d in range(len(node.shape)) if d not in a["inserted"] + a["batch"]]
    for dim, offset in zip(window_dims, a["window"], strict=True):
        inner[dim] = f"({inner[dim]} + long({kernel.coords[offset]}))"
    condition = (
        " && ".join(f"({c} >= 0l && {c} < {n}l)" for c, n in zip(inner, node.shape, strict=True))
        or "true"
    )
    kernel.lines.append(f"if ({condition}) {{")
    value = kernel.value(updates, kernel.coords)
    flat = linear(tuple(Term(f"uint({c})") for c in inner), node.shape).text
    ctype = ctype_of(node.dtype)
    if update == "set":
        kernel.lines.append(f"o0[{flat}] = {value};")
    elif update == "add" and not words:
        kernel.lines.append(
            f"atomic_fetch_add_explicit(reinterpret_cast<device atomic_{ctype}*>(o0) + ({flat}), {value}, memory_order_relaxed);"
        )
    else:
        kernel.lines += [
            f"device atomic_uint* address = reinterpret_cast<device atomic_uint*>(o0) + ({flat});",
            "uint expected = atomic_load_explicit(address, memory_order_relaxed);",
            "uint desired;",
            "do {",
            f"{ctype} current = {unpack_word(ctype, 'expected')};",
        ]
        if isinstance(update, Reducer):
            kernel.body.values[(update.parameters[0], ())] = Term("current")
            kernel.body.values[(update.parameters[1], ())] = value
            combined = str(kernel.value(update.body, ()))
        else:
            # Round after every update, as a scalar of the element type does.
            wide = accumulator(ctype)
            operands = (f"{wide}(current)", f"{wide}({value})")
            combined = f"{ctype}({format_scalar(ELEMENTWISE[update], wide, operands)})"
        kernel.lines += [
            f"{ctype} next = {combined};",
            f"desired = {pack_word(ctype, 'next')};",
            "} while (!atomic_compare_exchange_weak_explicit(address, &expected, desired, memory_order_relaxed, memory_order_relaxed));",
        ]
    kernel.lines.append("}")
    return kernel.render([target], name, [])


def sort_pass_source(
    kernel: Kernel,
    sort: Sort,
    sources: Sequence[Node],
    width: Node,
    outputs: Sequence[Node],
    name: str,
) -> str:
    """One pass of a merge sort along `sort.dim`: sorted runs of the width in
    `width` merge in pairs.

    One thread per element finds its rank in the merged pair: its position in
    its own run plus the elements of the other run that go before it, found
    by binary search with the comparator. An element of the left run goes
    before equal elements of the right, so the sort is stable. Every thread
    writes a different position.
    """
    shape = sort.inputs[0].shape
    n, stride = shape[sort.dim], math.prod(shape[sort.dim + 1 :])
    buffers = [kernel.buffer(source) for source in sources]
    kernel.lines += [
        f"uint w = {kernel.buffer(width)}[0];",
        f"uint j = {kernel.coords[sort.dim].text};",
        "uint base = j / (2u * w) * (2u * w);",
        "bool right = j - base >= w;",
        # The other run of the pair: where it starts and how many it holds.
        "uint other = right ? base : base + w;",
        f"uint count = right ? w : (base + w < {n}u ? min(w, {n}u - (base + w)) : 0u);",
        f"uint row = i - j * {stride}u;",
    ]
    for k, (source, buffer) in enumerate(zip(sources, buffers, strict=True)):
        kernel.lines.append(f"{ctype_of(source.dtype)} m{k} = {buffer}[i];")
    kernel.lines += [
        "uint lo = 0u, hi = count;",
        "while (lo < hi) {",
        "uint mid = (lo + hi) / 2u;",
        f"uint at = row + (other + mid) * {stride}u;",
    ]
    # The comparator on (other, this) for an element of the left run and on
    # (this, other) for one of the right: one evaluation serves both.
    used = set(reachable([sort.body]))
    for k, (source, buffer) in enumerate(zip(sources, buffers, strict=True)):
        ctype = ctype_of(source.dtype)
        for side, parameter in enumerate(sort.parameters[2 * k : 2 * k + 2]):
            if parameter in used:
                mine = "right" if side == 0 else "!right"
                kernel.lines.append(f"{ctype} p{k}_{side} = {mine} ? m{k} : {buffer}[at];")
                kernel.body.values[(parameter, ())] = Term(f"p{k}_{side}")
    before = kernel.value(sort.body, ())
    kernel.lines += [
        # From `lo` on, the other run's elements go after this one.
        f"if (right ? {before} : !{before}) hi = mid; else lo = mid + 1u;",
        "}",
        f"uint to = row + (base + (right ? j - base - w : j - base) + lo) * {stride}u;",
    ]
    stores = [f"o{k}[to] = m{k};" for k in range(len(sources))]
    return kernel.render(list(outputs), name, stores)


def fused_stores(kernel: Kernel, items: Sequence[Node | tuple[Op, Node]]) -> list[str]:
    """The statement storing each item into its output, in order.

    A node is stored at the thread's flat index. An (update, target) pair is
    a dynamic_update_slice written in place: only its update, at its offset.
    """
    stores = []
    for output, item in enumerate(items):
        if isinstance(item, Node):
            coords = kernel.coords_of(item.shape)
            stores.append(f"o{output}[i] = {kernel.value(item, coords)};")
            continue
        x, value, *starts = item[0].operands
        limits = tuple(n - s for n, s in zip(x.shape, value.shape, strict=True))
        offsets = kernel.offsets(starts, limits)
        coords = kernel.coords_of(value.shape)
        flat = linear(tuple(c + s for c, s in zip(coords, offsets, strict=True)), x.shape)
        stores.append(f"o{output}[{flat.text}] = {kernel.value(value, coords)};")
    return stores
