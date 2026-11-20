"""Buffer slots and launches for a graph: which values get a kernel of their own."""

from __future__ import annotations

import collections
import dataclasses
import math
from collections.abc import Sequence

import ml_dtypes
import numpy as np

from palladium.errors import EmitError
from palladium.stablehlo.attention import Attention, attention_source, match
from palladium.stablehlo.gemm import gemm_source
from palladium.stablehlo.graph import (
    SHORT_REDUCTION,
    Buffer,
    Constant,
    Dispatch,
    DispatchResult,
    Host,
    HostResult,
    LoopResult,
    Node,
    Op,
    Parameter,
    ReductionResult,
    Sort,
    SortResult,
    VariadicReduce,
    While,
    copy_of,
    dependencies,
    guard_flag,
    in_place_carries,
    is_short,
    is_stored,
    owns_slot,
    reachable,
)
from palladium.stablehlo.kernel import (
    Kernel,
    fused_stores,
    reduction_source,
    scatter_source,
    sort_pass_source,
    unpacked_source,
    variadic_reduction_source,
    words_source,
)
from palladium.stablehlo.program import (
    RELAXED,
    SAFE,
    Branch,
    HostCall,
    Launch,
    Program,
    Slot,
    Step,
    UnsupportedProgram,
)
from palladium.stablehlo.scalar import ctype_of

# Buffer arguments one Metal kernel can bind.
MAX_BUFFERS = 31


# A reduction stores an operand that other nodes read too, if computing it
# reads more buffers than this: a normalization reads what it took the mean
# of twice more, and a fused residual sum can bind dozens of buffers.
STORE_READS = 2


# Split K when a product's output has fewer GEMM tiles than this: one
# threadgroup per 64x64 tile leaves most of the GPU idle, and one float32
# register per output accumulates all of K in sequence.
_SPLIT_TILES = 32


_MIN_CHUNK = 256


def _k_splits(batch: int, m: int, n: int, k: int, dtype: np.dtype) -> int:
    """K chunks for a float32 product, a power of two up to `SHORT_REDUCTION`
    so the partial sums fold into one kernel; 1 to leave it whole.

    Chunks stay multiples of 16, the `simdgroup_matrix` kernel's K step, and
    at least `_MIN_CHUNK` long. Half types are not split: their partials
    would round to the output type.
    """
    tiles = batch * -(-m // 64) * -(-n // 64)
    if dtype != np.float32 or tiles >= _SPLIT_TILES:
        return 1
    splits = 1
    while (
        splits * 2 <= SHORT_REDUCTION
        and tiles * splits < _SPLIT_TILES
        and k % (splits * 2 * 16) == 0
        and k // (splits * 2) >= _MIN_CHUNK
    ):
        splits *= 2
    return splits


@dataclasses.dataclass(frozen=True)
class Product:
    """A dot_general as the batched matrix product (batch, M, K) by (batch, K, N).

    `left` and `right` permute the operands to (batch..., free...,
    contracted...) and (batch..., contracted..., free...), keeping the
    order in which the op pairs its batch and contracting dimensions.
    """

    batch: int
    m: int
    n: int
    k: int
    left: tuple[int, ...]
    right: tuple[int, ...]

    @classmethod
    def of(cls, node: Op) -> Product:
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
        return cls(
            math.prod(lhs.shape[d] for d in lb),
            math.prod(lhs.shape[d] for d in lf),
            math.prod(rhs.shape[d] for d in rf),
            math.prod(lhs.shape[d] for d in lc),
            lb + lf + lc,
            rb + rc + rf,
        )


def permuted(value: Node, perm: tuple[int, ...], dtype: np.dtype) -> Node:
    """`value` with its dimensions in the order `perm`, converted to `dtype`."""
    if perm != tuple(range(len(value.shape))):
        shape = tuple(value.shape[d] for d in perm)
        value = Op("transpose", [value], value.dtype, shape, {"perm": perm})
    if value.dtype != dtype:
        value = Op("convert", [value], dtype, value.shape, {})
    return value


def loop_body(loop: While, in_place: dict[int, Op]) -> Program:
    """The program of one iteration, from the loop's parameters to its carries.

    Every changing carry gets an output slot of its own, even swaps and
    constants, so two banks of outputs are safe for permutations with any
    cycle length. Only invariants, in-place carries and induction variables
    alias a parameter. A carry computed by launches of its own already has
    such a slot and needs no copy into another.
    """
    results: list[Node] = []
    for k, result in enumerate(loop.results):
        if result is loop.parameters[k] or k in loop.inductions or k in in_place:
            results.append(loop.parameters[k])
        elif owns_slot(result) and (loop.trips is not None or result not in results):
            results.append(result)
        else:
            # A dynamic loop writes each carry back to one place, so there
            # two carries with one value need a slot each.
            results.append(copy_of(result))
    updates = [(update, loop.parameters[k]) for k, update in in_place.items()]
    return compile_graph(loop.parameters, results, updates, loop.effects)


def induction_values(start: int, step: int, steps: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """An induction variable after each of `steps` iterations.

    Computed in int64 and narrowed, which wraps like the kernels' integer
    arithmetic would.
    """
    return (start + step * steps.astype(np.int64)).astype(dtype)


def rebound(launch: Step, bindings: dict[int, int], offsets: dict[int, int]) -> Step:
    """A loop body's step for one iteration: its slots as `bindings` map
    them, and the slots in `offsets` read at that byte offset. Branches skip
    relative to themselves, so they hold wherever the body is replayed."""
    if isinstance(launch, Branch):
        slot = None if launch.slot is None else bindings[launch.slot]
        return dataclasses.replace(launch, slot=slot)
    return dataclasses.replace(
        launch,
        slots=tuple(bindings[s] for s in launch.slots),
        offsets=tuple(
            (launch.offsets[j] if launch.offsets else 0) + offsets.get(s, 0)
            for j, s in enumerate(launch.slots)
        ),
    )


def compile_graph(
    parameters: list[Parameter],
    results: list[Node],
    updates: Sequence[tuple[Op, Parameter]] = (),
    effects: Sequence[Host | While] = (),
) -> Program:
    """Slots and steps computing `results` from `parameters`.

    Each of `updates` is a dynamic_update_slice whose operand is the paired
    parameter: its update is written into that parameter's buffer in place.
    `effects` are the host calls, and the loops holding some, that run even
    if no result depends on them, in order.
    """
    return Scheduler(parameters).compile(results, updates, effects)


def copy_source(dtype: np.dtype, gated: bool = False) -> tuple[str, str]:
    """A kernel copying one buffer to another, and its name. Gated, it binds
    a boolean scalar first and copies only if that is true."""
    ctype = ctype_of(dtype)
    name = f"copy_{'if_' if gated else ''}{ctype}"
    flag = "    device const bool* flag [[buffer(0)]],\n" if gated else ""
    first = int(gated)
    return (
        "#include <metal_stdlib>\nusing namespace metal;\n\n"
        f"kernel void {name}(\n{flag}    device const {ctype}* b0 [[buffer({first})]],\n"
        f"    device {ctype}* o0 [[buffer({first + 1})]],\n"
        "    uint i [[thread_position_in_grid]]) {\n"
        f"    {'if (flag[0]) ' if gated else ''}o0[i] = b0[i];\n}}\n"
    ), name


# A loop the host drives runs this many iterations between two reads of its
# condition, after 1, 1, 2 and 4: a short loop wastes little, and a long one
# waits for the GPU once in eight iterations.
SPECULATED = (1, 1, 2, 4, 8)


class Scheduler:
    """Assigns the values of one graph to buffer slots and kernel launches.

    Values that need a buffer of their own (products, long reductions,
    scatters, loops, kernel dispatches) are scheduled in dependency order;
    everything else fuses into the kernels that read it.
    """

    def __init__(self, parameters: list[Parameter]):
        self.slots = [Slot("parameter", p.dtype, p.shape) for p in parameters]
        self.slot_of: dict[Node, int] = {p: k for k, p in enumerate(parameters)}
        self.launches: list[Step] = []
        # Ops materialized for the buffer limit or for branches.
        self.stored: set[Node] = set()
        self.visited: set[Node] = set()
        self.loop_slots: dict[While, list[int]] = {}
        self.reduction_slots: dict[VariadicReduce, list[int]] = {}
        self.dispatch_slots: dict[Dispatch, list[int]] = {}
        self.host_slots: dict[Host, list[int]] = {}
        # Products recognized as attention, or None for those that are not.
        self.attentions: dict[Op, Attention | None] = {}
        # How many nodes read each node, among those `compile` schedules.
        self.uses: collections.Counter[Node] = collections.Counter()
        self.sort_slots: dict[Sort, list[int]] = {}
        # Buffers read by node, for one size of `stored`.
        self.read_sets: dict[Node, tuple[int, set[Node]]] = {}

    def add_constant(self, node: Node) -> None:
        """A slot uploaded once with the elements of a constant array."""
        assert isinstance(node, Constant) and node.array is not None
        self.slot_of[node] = len(self.slots)
        self.slots.append(Slot("constant", node.dtype, node.shape, node.array.tobytes()))

    def fresh(self, node: Node) -> None:
        """A slot of its own for `node`, allocated per execution."""
        ctype_of(node.dtype)
        self.slot_of[node] = len(self.slots)
        self.slots.append(Slot("fresh", node.dtype, node.shape))

    def buffer(self, dtype: np.dtype, shape: tuple[int, ...]) -> Buffer:
        """A fresh slot that no value of the graph stands for, such as partial sums."""
        node = Buffer(dtype, shape)
        self.fresh(node)
        return node

    def store(self, node: Node) -> None:
        """Compute `node` into a buffer now; kernels made from here on read it."""
        self.materialize(node)
        self.stored.add(node)

    def new_kernel(self, shape: tuple[int, ...]) -> Kernel:
        """A kernel that reads stored ops from their buffers and can store more."""
        return Kernel(shape, self.stored, self.store)

    def launch(
        self,
        kernel: Kernel,
        source: str,
        name: str,
        nodes: list[Node],
        threads: int = 0,
        size: int | None = None,
        grid: int | None = None,
        math_mode: int | None = None,
    ) -> None:
        """Launch `source`, binding the buffers `kernel` reads, then `nodes` as outputs.

        The grid is one thread per element of the first output, or of `size`,
        times `threads`, the threadgroup size of a reduction; `grid` overrides it.
        """
        for operand in kernel.buffers:
            if operand not in self.slot_of:
                self.add_constant(operand)
        bound = [self.slot_of[n] for n in kernel.buffers] + [self.slot_of[n] for n in nodes]
        if len(bound) > MAX_BUFFERS:
            raise UnsupportedProgram(
                f"metal: a kernel needs {len(bound)} buffers, over Metal's {MAX_BUFFERS}"
            )
        size = math.prod(nodes[0].shape) if size is None else size
        grid = size * (threads or 1) if grid is None else grid
        if math_mode is None:
            # Reductions run SAFE. Everything else runs RELAXED: FAST assumes
            # finite operands, so `x - x` and `x * 0` lose their NaNs and
            # folded extrema their infinities and zero signs.
            math_mode = SAFE if threads else RELAXED
        self.launches.append(Launch(source, name, tuple(bound), grid, threads, math_mode))

    def schedule_variadic(self, node: ReductionResult) -> int:
        """The slot of one result of a multi-result reduction, launched once for all."""
        reduction = node.reduction
        if reduction in self.reduction_slots:
            return self.reduction_slots[reduction][node.index]
        outputs: list[Node] = [
            ReductionResult(reduction, k, value.dtype, node.shape)
            for k, value in enumerate(reduction.body)
        ]
        for result in outputs:
            self.fresh(result)
        self.bound([*reduction.inputs, *reduction.initial], len(outputs))
        kernel = self.new_kernel(node.shape)
        name = f"variadic_reduce_{len(self.launches)}"
        source = variadic_reduction_source(kernel, reduction, outputs, name)
        self.launch(kernel, source, name, outputs, math_mode=SAFE)
        self.reduction_slots[reduction] = [self.slot_of[result] for result in outputs]
        return self.reduction_slots[reduction][node.index]

    def schedule_dispatch(self, node: DispatchResult) -> int:
        """The slot of one result of a Palladium kernel, launched once per batch element."""
        dispatch = node.dispatch
        if dispatch in self.dispatch_slots:
            return self.dispatch_slots[dispatch][node.index]
        inputs = [self.materialize(x) for x in dispatch.inputs]
        outputs = []
        for k, result in enumerate(dispatch.results):
            if k in dispatch.aliases:
                # The result starts as a copy of its input; the kernel updates it.
                self.slot_of[result] = self.materialize(
                    copy_of(dispatch.inputs[dispatch.aliases[k]])
                )
            else:
                self.fresh(result)
            outputs.append(self.slot_of[result])
        for element in range(dispatch.batch):
            self.launches.append(
                Launch(
                    dispatch.source,
                    dispatch.function,
                    tuple(inputs + outputs),
                    dispatch.grid,
                    dispatch.threadgroup,
                    dispatch.math_mode,
                    tuple(element * s for s in dispatch.strides) if element else (),
                )
            )
        self.dispatch_slots[dispatch] = outputs
        return outputs[node.index]

    def schedule_host(self, host: Host) -> list[int]:
        """The slots of a host call's results, after the step that runs it."""
        if host in self.host_slots:
            return self.host_slots[host]
        inputs = [self.materialize(x) for x in host.inputs]
        if host.guards:
            # In a conditional branch: skip the call unless the branch is taken.
            self.launches.append(Branch(self.materialize(guard_flag(host.guards)), 1))
        for result in host.results:
            self.fresh(result)
        outputs = [self.slot_of[result] for result in host.results]
        types = tuple((n.dtype, n.shape) for n in (*host.inputs, *host.results))
        self.launches.append(
            HostCall(host.function, tuple(inputs + outputs), len(inputs), types, host.name)
        )
        self.host_slots[host] = outputs
        return outputs

    def materialize(self, node: Node) -> int:
        """The slot holding `node`, computed by a kernel of its own if nothing holds it."""
        self.schedule(node)
        if node not in self.slot_of:
            if isinstance(node, Constant) and node.splat is None:
                self.add_constant(node)
            else:
                name = f"init_{len(self.launches)}"
                kernel = self.new_kernel(node.shape)
                source = kernel.source([node], name)
                if len(kernel.buffers) >= MAX_BUFFERS:
                    self.shrink(node)
                    name = f"init_{len(self.launches)}"
                    kernel = self.new_kernel(node.shape)
                    source = kernel.source([node], name)
                self.fresh(node)
                self.launch(kernel, source, name, [node])
        return self.slot_of[node]

    def reads(self, node: Node) -> set[Node]:
        """Buffers a kernel computing `node` binds, which only storing an op changes."""
        cached = self.read_sets.get(node)
        if cached is not None and cached[0] == len(self.stored):
            return cached[1]
        before = len(self.stored)
        kernel = self.new_kernel(node.shape)
        kernel.value(node, kernel.coords_of(node.shape))
        found = set(kernel.buffers)
        if len(self.stored) == before:  # generating it may itself store an op
            self.read_sets[node] = (before, found)
        return found

    def shrink(self, node: Node) -> None:
        """Materialize subexpressions of `node` until one kernel can compute it.

        Metal binds at most MAX_BUFFERS buffers per kernel. A fused value
        that self.reads more stores its largest operand that fits in one kernel,
        descending into operands too large themselves; a self.stored operand then
        costs a self.store and a read instead of index arithmetic.
        """
        while len(self.reads(node)) >= MAX_BUFFERS:
            operands = [
                o
                for o in dependencies(node)
                if isinstance(o, Op) and o not in self.slot_of and o not in self.stored
            ]
            if not operands:
                raise UnsupportedProgram(
                    f"metal: a value reads {len(self.reads(node))} buffers, over Metal's {MAX_BUFFERS}"
                )
            largest = max(operands, key=lambda o: len(self.reads(o)))
            if len(self.reads(largest)) < MAX_BUFFERS:
                self.store(largest)
            else:
                self.shrink(largest)

    def bound(self, nodes: Sequence[Node], outputs: int = 1) -> None:
        """Store the values among `nodes` that read the most buffers, until
        one kernel can read all of `nodes` and write `outputs` buffers."""
        while sum(len(self.reads(n)) for n in nodes) > MAX_BUFFERS - outputs:
            fused = [n for n in nodes if isinstance(n, Op) and n not in self.stored]
            if not fused:
                raise UnsupportedProgram(f"metal: a kernel reads over {MAX_BUFFERS} buffers")
            self.store(max(fused, key=lambda n: len(self.reads(n))))

    def schedule_loop(self, loop: While) -> list[int]:
        """The slots of the loop's results, after the steps that run it.

        A loop in a conditional branch that the host drives, or that calls
        the host, runs only if the branch is taken: a branch step skips it.
        Its inputs are computed before, since values outside the loop may
        read them too.
        """
        if loop in self.loop_slots:
            return self.loop_slots[loop]
        if not loop.guards:
            return self.replay_loop(loop)
        flag = self.materialize(guard_flag(loop.guards))
        for node in loop.inputs:
            self.materialize(node)
        start = len(self.launches)
        self.launches.append(Branch(flag, 0))
        slots = self.replay_loop(loop)
        self.launches[start] = Branch(flag, len(self.launches) - start - 1)
        return slots

    def replay_loop(self, loop: While) -> list[int]:
        """The slots of the loop's results, after replaying its body `trips` times."""
        count = len(loop.results)
        if loop.trips is None:
            self.loop_slots[loop] = self.schedule_dynamic_loop(loop)
            return self.loop_slots[loop]
        if not loop.trips:
            self.loop_slots[loop] = [self.materialize(n) for n in loop.inputs[:count]]
            return self.loop_slots[loop]
        inductions = loop.inductions
        in_place = in_place_carries(loop)
        current = []
        for k, node in enumerate(loop.inputs):
            if k in inductions:
                current.append(-1)  # bound per iteration below
            elif k in in_place:
                # One buffer per execution, initialized once, updated in place.
                current.append(self.materialize(copy_of(node)))
            else:
                current.append(self.materialize(node))
        body = loop_body(loop, in_place)
        if len(self.launches) + loop.trips * len(body.steps) > 100_000:
            raise UnsupportedProgram(
                "metal: stablehlo.while exceeds the 100000-launch replay limit"
            )
        fixed, banks = self.adopt(body)

        # Induction values are known: iteration k binds element k of one
        # constant array per variable instead of computing it with a kernel.
        def induction(k: int, steps: np.ndarray) -> int:
            array = induction_values(*inductions[k], steps, loop.parameters[k].dtype)
            self.slots.append(Slot("constant", array.dtype, array.shape, array.tobytes()))
            return len(self.slots) - 1

        arrays = {k: induction(k, np.arange(loop.trips)) for k in inductions}
        for iteration in range(loop.trips):
            for k, array in arrays.items():
                current[k] = array
            bindings = fixed | dict(enumerate(current)) | banks[iteration % 2]
            offsets = {k: iteration * loop.parameters[k].dtype.itemsize for k in arrays}
            self.launches.extend(rebound(item, bindings, offsets) for item in body.steps)
            current = [bindings[s] for s in body.outputs] + current[count:]
        for k in inductions:
            current[k] = induction(k, np.array(loop.trips))
        self.loop_slots[loop] = current[:count]
        return self.loop_slots[loop]

    def schedule_dynamic_loop(self, loop: While) -> list[int]:
        """The slots of the results of a loop whose condition the host reads.

        The carries start as copies in slots of their own, which hold the
        results in the end. Reading the condition costs a wait for the GPU,
        so a body made of launches alone runs several iterations between two
        reads: `speculate`. A body with host steps of its own takes one read
        per iteration: `alternate`.
        """
        count = len(loop.results)
        body = loop_body(loop, {})
        condition = compile_graph(loop.parameters, [loop.condition])
        changing = [k for k in range(count) if body.outputs[k] != k]
        home = {
            k: self.materialize(copy_of(node) if k in changing else node)
            for k, node in enumerate(loop.inputs)
        }
        fixed, _ = self.adopt(body, second_bank=False)
        away = {k: fixed[body.outputs[k]] for k in changing}
        condition_slots, _ = self.adopt(condition, second_bank=False)

        def steps(program: Program, extra: dict[int, int], carries: dict[int, int]) -> list[Step]:
            bindings = extra | home | carries
            return [rebound(step, bindings, {}) for step in program.steps]

        def flag(carries: dict[int, int]) -> int:
            return (condition_slots | home | carries)[condition.outputs[0]]

        def copies(gate: int | None) -> list[Step]:
            """The second bank copied into the carries' slots, if `gate` holds."""
            launches: list[Step] = []
            for k in changing:
                slot = self.slots[home[k]]
                if size := math.prod(slot.shape):
                    source, name = copy_source(slot.dtype, gate is not None)
                    slots = (away[k], home[k]) if gate is None else (gate, away[k], home[k])
                    launches.append(Launch(source, name, slots, size, 0, RELAXED))
            return launches

        there = fixed | {body.outputs[k]: away[k] for k in changing}
        test_home = steps(condition, condition_slots, {})
        first = steps(body, there, {})
        if all(isinstance(step, Launch) for step in body.steps):
            self.speculate([*test_home, *first, *copies(flag({}))], flag({}))
        else:
            back = fixed | {body.outputs[k]: home[k] for k in changing}
            second = steps(body, back, away)
            test_away = steps(condition, condition_slots, away)
            self.alternate(test_home, flag({}), first, test_away, flag(away), second, copies(None))
        return [home[k] for k in range(count)]

    def speculate(self, iteration: list[Step], flag: int) -> None:
        """Lay out a loop whose `iteration` is the condition, the body into a
        second bank, and copies back that only run if the condition held.

        Once the condition is false the carries stop changing, so it stays
        false and further iterations change nothing: they may run without
        the host having looked. Groups of `SPECULATED` iterations end in a
        branch out of the loop; the last group repeats.
        """
        if len(self.launches) + sum(SPECULATED) * len(iteration) > 100_000:
            raise UnsupportedProgram(
                "metal: stablehlo.while exceeds the 100000-launch replay limit"
            )
        *leading, repeated = SPECULATED
        tail: list[Step] = [*iteration * repeated, Branch(flag, 1)]
        tail.append(Branch(None, -(len(tail) + 1)))
        groups = [[*iteration * n, Branch(flag, 0)] for n in leading]
        remaining = len(tail) + sum(len(group) for group in groups)
        for group in groups:
            remaining -= len(group)
            group[-1] = Branch(flag, remaining)
            self.launches += group
        self.launches += tail

    def alternate(
        self,
        test_home: list[Step],
        flag_home: int,
        first: list[Step],
        test_away: list[Step],
        flag_away: int,
        second: list[Step],
        copies: list[Step],
    ) -> None:
        """Lay out a loop with one read of the condition per iteration.

        The body is laid out twice, from the carries' slots to a second bank
        and back, each behind the condition on the carries it reads; leaving
        after the first copy moves the second bank into the carries' slots:

            L:  condition(A); if false go to END
                body(A to B)
                condition(B); if false go to E
                body(B to A); go to L
            E:  copy B to A
            END:
        """
        tail = [*second, Branch(None, 0)]
        rest = [*first, *test_away, Branch(flag_away, len(tail)), *tail]
        # Back to the first condition: over everything laid out before the jump.
        tail[-1] = Branch(None, -(len(test_home) + 1 + len(rest)))
        rest = [*first, *test_away, Branch(flag_away, len(tail)), *tail]
        self.launches += [*test_home, Branch(flag_home, len(rest) + len(copies)), *rest, *copies]

    def adopt(
        self, body: Program, second_bank: bool = True
    ) -> tuple[dict[int, int], list[dict[int, int]]]:
        """Slots here for the slots a loop body adds to its parameters.

        Returns the slot of each, and two banks of slots for the body's
        outputs: iterations alternate between them, so none writes a carry it
        still reads. Without `second_bank`, the caller places the outputs.
        """
        fixed: dict[int, int] = {}
        banks: list[dict[int, int]] = [{}, {}]
        outputs = set(body.outputs)
        for k, slot in enumerate(body.slots):
            if slot.kind == "parameter":
                continue
            fixed[k] = len(self.slots)
            self.slots.append(slot)
            if second_bank and k in outputs and slot.kind == "fresh":
                banks[0][k] = fixed[k]
                banks[1][k] = len(self.slots)
                self.slots.append(slot)
        return fixed, banks

    def schedule(self, root: Node) -> None:
        """Give `root` and everything it depends on their launches, dependencies first.

        An explicit stack, not recursion: dependency chains are as deep as
        the program is long.
        """
        pending = [(root, False)]
        while pending:
            node, ready = pending.pop()
            if ready:
                self.launch_node(node)
            elif node not in self.visited:
                self.visited.add(node)
                pending.append((node, True))
                # Attention needs its operands only, not the scores between.
                found = self.attention(node)
                needed = [found.q, found.k, found.v] if found else dependencies(node)
                pending.extend((n, False) for n in reversed(needed))

    def launch_node(self, node: Node) -> None:
        """The launches of `node` itself; its dependencies are scheduled."""
        if isinstance(node, ReductionResult):
            self.slot_of[node] = self.schedule_variadic(node)
        elif isinstance(node, LoopResult):
            self.slot_of[node] = self.schedule_loop(node.loop)[node.index]
        elif isinstance(node, DispatchResult):
            self.slot_of[node] = self.schedule_dispatch(node)
        elif isinstance(node, HostResult):
            self.slot_of[node] = self.schedule_host(node.host)[node.index]
        elif isinstance(node, SortResult):
            self.slot_of[node] = self.schedule_sort(node.sort)[node.index]
        elif is_stored(node):
            self.store(node)
        elif isinstance(node, Op):
            if node.name == "scatter":
                self.schedule_scatter(node)
            elif node.name == "dot_general" and self.attention(node):
                self.schedule_attention(node)
            elif node.name == "dot_general":
                self.schedule_dot(node)
            elif node.name == "reduce" and not is_short(node):
                self.schedule_reduce(node)
            # Every other op, short reductions included, fuses into the kernels reading it.

    def schedule_reduce(self, node: Op) -> None:
        """128 threads per output element; a long reduction to few outputs
        first reduces 4096-element chunks, so it occupies the GPU."""
        x, init = node.operands
        if (
            isinstance(x, Op)
            and x not in self.stored
            and self.uses[x] > 1
            and len(self.reads(x)) > STORE_READS
        ):
            # Something else reads the operand too, as a normalization reads
            # what it took the mean of: compute it once, not per reader.
            self.store(x)
        self.bound([x, init])
        size = math.prod(x.shape[d] for d in node.attributes["dims"])
        chunks = math.ceil(size / 4096) if size > 4096 and math.prod(node.shape) < 128 else 1
        reduction = node
        if chunks > 1:
            dtype = (
                np.dtype(np.float32)
                if node.dtype in (np.float16, ml_dtypes.bfloat16)
                else node.dtype
            )
            partial = self.buffer(dtype, (*node.shape, chunks))
            name = f"reduce_{len(self.launches)}"
            kernel = self.new_kernel(partial.shape)
            source = reduction_source(kernel, node, partial, name, chunks)
            self.launch(kernel, source, name, [partial], 128)
            reduction = Op(
                "reduce",
                [partial, init],
                node.dtype,
                node.shape,
                {"dims": [len(node.shape)], "reducer": node.attributes["reducer"]},
            )
        self.fresh(node)
        name = f"reduce_{len(self.launches)}"
        kernel = self.new_kernel(node.shape)
        self.launch(kernel, reduction_source(kernel, reduction, node, name), name, [node], 128)

    def schedule_scatter(self, node: Op) -> None:
        """A scatter: a copy of the operand, then one thread per update
        element combining it in. Elements narrower than the 32-bit words of
        Metal's atomics get a word each while they are updated."""
        x, indices, updates = node.operands
        update = node.attributes["update"]
        if indices.dtype not in (np.int32, np.uint32):
            raise UnsupportedProgram("metal: stablehlo.scatter requires 32-bit indices")
        if update != "set" and node.dtype.itemsize == 8:
            raise UnsupportedProgram("metal: stablehlo.scatter combining 64-bit elements")
        words = update != "set" and node.dtype.itemsize < 4
        self.bound([x])
        self.bound([indices, updates])
        self.fresh(node)
        kernel = self.new_kernel(node.shape)
        name = f"scatter_init_{len(self.launches)}"
        if not math.prod(updates.shape) or not math.prod(node.shape):
            self.launch(kernel, kernel.source([x], name), name, [node])
            return
        target: Node = node
        if words:
            target = self.buffer(np.dtype(np.uint32), node.shape)
            source = words_source(kernel, x, target, name)
        else:
            source = kernel.source([x], name)
        self.launch(kernel, source, name, [target])
        kernel = self.new_kernel(updates.shape)
        name = f"scatter_{len(self.launches)}"
        # This grid covers updates, not output elements.
        self.launch(
            kernel,
            scatter_source(kernel, node, target, name),
            name,
            [target],
            grid=math.prod(updates.shape),
            math_mode=SAFE,
        )
        if words:
            kernel = self.new_kernel(node.shape)
            name = f"scatter_unpack_{len(self.launches)}"
            self.launch(kernel, unpacked_source(kernel, target, node, name), name, [node])

    def schedule_sort(self, sort: Sort) -> list[int]:
        """The slots of a sort's results: a merge sort along the dimension,
        one launch per doubling of the sorted runs, between two banks of
        buffers so the last pass writes the results."""
        if sort in self.sort_slots:
            return self.sort_slots[sort]
        inputs = [self.materialize(x) for x in sort.inputs]
        shape = sort.inputs[0].shape
        n = shape[sort.dim] if shape else 1
        passes = (n - 1).bit_length() if math.prod(shape) else 0
        if not passes:
            self.sort_slots[sort] = inputs
            return inputs
        results = [self.buffer(x.dtype, x.shape) for x in sort.inputs]
        spare = [self.buffer(x.dtype, x.shape) for x in sort.inputs] if passes > 1 else []
        banks = [[self.slot_of[b] for b in bank] for bank in (results, spare)]
        widths = Constant(
            np.dtype(np.uint32), (passes,), None, 1 << np.arange(passes, dtype=np.uint32)
        )
        self.add_constant(widths)
        kernel = self.new_kernel(shape)
        name = f"sort_pass_{len(self.launches)}"
        source = sort_pass_source(kernel, sort, list(sort.inputs), widths, results, name)
        # The sources bind first, in order, then the widths.
        assert list(kernel.buffers) == [*dict.fromkeys(sort.inputs), widths]
        reads = inputs
        for k in range(passes):
            writes = banks[(passes - 1 - k) % 2]
            bound = {**dict(zip(sort.inputs, reads, strict=True)), widths: self.slot_of[widths]}
            slots = (*(bound[node] for node in kernel.buffers), *writes)
            offsets = tuple(4 * k if node is widths else 0 for node in kernel.buffers)
            offsets += (0,) * len(writes)
            self.launches.append(Launch(source, name, slots, math.prod(shape), 0, SAFE, offsets))
            reads = writes
        self.sort_slots[sort] = banks[0]
        return banks[0]

    def attention(self, node: Node) -> Attention | None:
        """`node` as attention, if it is one the kernel can run."""
        if not isinstance(node, Op) or node.name != "dot_general":
            return None
        if node not in self.attentions:
            found = match(node, self.uses)
            if found:
                try:
                    attention_source(
                        found.heads, found.queries, found.keys, found.head_dim, found.scale
                    )
                except EmitError:
                    found = None
            self.attentions[node] = found
        return self.attentions[node]

    def schedule_attention(self, node: Op) -> None:
        """Attention as one kernel: a threadgroup per tile of queries, with
        each head a batch, so operands bind in the layout they are stored."""
        found = self.attentions[node]
        assert found is not None
        heads, queries, keys, head_dim = found.heads, found.queries, found.keys, found.head_dim
        source, name, grid, threadgroup = attention_source(
            heads, queries, keys, head_dim, found.scale
        )
        self.fresh(node)
        slots = (
            self.packed(found.q, (heads, queries, 1, head_dim)),
            self.packed(found.k, (heads, keys, 1, head_dim)),
            self.packed(found.v, (heads, keys, 1, head_dim)),
            self.slot_of[node],
        )
        self.launches.append(Launch(source, name, slots, grid, threadgroup))

    def schedule_dot(self, node: Op) -> None:
        """A product on the GEMM kernel, its operands packed to (batch, M, K)
        and (batch, K, N) where they are not stored that way already."""
        product = Product.of(node)
        batch, m, n, k = product.batch, product.m, product.n, product.k
        self.fresh(node)
        if not math.prod(node.shape):
            return
        if not k:
            # Nothing to contract: the product is zero, and reads neither operand.
            kernel = self.new_kernel(node.shape)
            name = f"empty_dot_{len(self.launches)}"
            zero = Constant(node.dtype, node.shape, 0)
            self.launch(kernel, kernel.source([zero], name), name, [node])
            return
        lhs, rhs = node.operands
        left = permuted(lhs, product.left, node.dtype)
        right = permuted(rhs, product.right, node.dtype)
        splits = _k_splits(batch, m, n, k, node.dtype)
        if splits > 1:
            # Each batch's K in `splits` chunks, as a batched product into
            # float32 partials that a short reduction then sums.
            left = Op("reshape", [left], node.dtype, (batch, m, splits, k // splits), {})
            left = permuted(left, (0, 2, 1, 3), node.dtype)
            batch, k = batch * splits, k // splits
        operands = self.packed(left, (batch, m, k)), self.packed(right, (batch, k, n))
        target = self.buffer(node.dtype, (batch // splits, splits, m, n)) if splits > 1 else node
        try:
            source, grid = gemm_source(batch, m, n, k, node.dtype)
        except EmitError as e:
            raise UnsupportedProgram(f"metal: stablehlo.dot_general: {e}") from e
        slots = (*operands, self.slot_of[target])
        self.launches.append(Launch(source, "gemm", slots, grid, 128))
        if splits > 1:
            total = Op(
                "reduce",
                [target, Constant(node.dtype, (), 0)],
                node.dtype,
                (batch // splits, m, n),
                {"dims": [1], "reducer": "add"},
            )
            total = Op("reshape", [total], node.dtype, node.shape, {})
            kernel = self.new_kernel(node.shape)
            name = f"split_k_sum_{len(self.launches)}"
            self.launch(kernel, kernel.source([total], name), name, [node])

    def packed(self, value: Node, shape: tuple[int, ...]) -> int:
        """The slot holding `value` in row-major order, read as `shape`.

        A value that has a slot binds it directly: a slot's own shape does
        not matter to a kernel that reads it flat.
        """
        if value in self.slot_of:
            return self.slot_of[value]
        return self.materialize(Op("reshape", [value], value.dtype, shape, {}))

    def compile(
        self,
        results: list[Node],
        updates: Sequence[tuple[Op, Parameter]] = (),
        effects: Sequence[Host | While] = (),
    ) -> Program:
        """The program computing `results`, with each update written in place.

        Each of `updates` is a dynamic_update_slice whose operand is the paired
        parameter: only the update is written, into that parameter's buffer.
        `effects` run first, in order, whether or not a result depends on them.
        """
        roots = [*results, *(x for effect in effects for x in effect.inputs)]
        self.uses = collections.Counter(d for n in reachable(roots) for d in dependencies(n))
        for effect in effects:
            for node in effect.inputs:
                self.schedule(node)
            if isinstance(effect, Host):
                self.schedule_host(effect)
            else:
                self.schedule_loop(effect)
        for node in results:
            self.schedule(node)
        for update, _ in updates:
            for operand in update.operands[1:]:
                self.schedule(operand)
        self.fuse_outputs(results, updates)
        return Program(self.slots, self.launches, [self.slot_of[n] for n in results])

    def fuse_outputs(self, results: list[Node], updates: Sequence[tuple[Op, Parameter]]) -> None:
        """Launch the kernels storing the results that no launch has computed.

        One kernel per element count: outputs of different shapes but equal
        size share a launch, each read at the coordinates of the flat index,
        unless together they bind more buffers than a kernel can.
        """
        computed: dict[int, list[Node]] = {}
        for node in results:
            if node in self.slot_of:
                continue
            if isinstance(node, Constant) and node.splat is None:
                self.add_constant(node)
                continue
            self.fresh(node)
            computed.setdefault(math.prod(node.shape), []).append(node)
        in_place: dict[int, list[tuple[Op, Parameter]]] = {}
        for update, target in updates:
            in_place.setdefault(math.prod(update.operands[1].shape), []).append((update, target))

        for size in [*computed, *(s for s in in_place if s not in computed)]:
            nodes = computed.get(size, [])
            shape = nodes[0].shape if nodes else in_place[size][0][0].operands[1].shape
            items = [*nodes, *in_place.get(size, [])]
            kernels = [self.fuse(shape, items)]
            if len(kernels[0][0].buffers) + len(items) > MAX_BUFFERS:
                kernels = [self.fuse(shape, group) for group in self.bins(shape, items)]
            for kernel, stores, targets in kernels:
                name = f"fused_{len(self.launches)}"
                self.launch(kernel, kernel.render(targets, name, stores), name, targets, size=size)

    def fuse(self, shape: tuple[int, ...], items: list) -> tuple[Kernel, list[str], list[Node]]:
        """One kernel storing each item: a node, or an (update, target) pair."""
        kernel = self.new_kernel(shape)
        targets = [item if isinstance(item, Node) else item[1] for item in items]
        return kernel, fused_stores(kernel, items), targets

    def bins(self, shape: tuple[int, ...], items: list) -> list[list]:
        """Items in groups whose kernels bind at most MAX_BUFFERS buffers."""
        groups: list[list] = []
        bound: set[Node] = set()
        for item in items:
            self.shrink(item if isinstance(item, Node) else item[0].operands[1])
            needed = set(self.fuse(shape, [item])[0].buffers)
            if not groups or len(bound | needed) + len(groups[-1]) + 1 > MAX_BUFFERS:
                groups.append([])
                bound = set()
            groups[-1].append(item)
            bound |= needed
        return groups
