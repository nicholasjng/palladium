"""What a compiled program is: buffer slots and the steps that fill them."""

from __future__ import annotations

import dataclasses
import heapq
import math
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from palladium._runtime import MATH_MODES

SAFE, RELAXED, FAST = (MATH_MODES[mode] for mode in ("safe", "relaxed", "fast"))


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
    math_mode: int = FAST  # a runtime math mode: SAFE, RELAXED or FAST
    offsets: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class HostCall:
    """A function run on the host between launches, once the GPU has computed
    its inputs: Python callbacks, and ops without a kernel.

    `function` takes the input arrays and returns the result arrays; the
    first `inputs` of `slots` are its inputs, the others its results.
    `types` are the dtype and shape of each array: a slot may be bound at an
    offset into a longer buffer, or share storage of another shape.
    """

    function: Callable[..., Sequence[np.ndarray]]
    slots: tuple[int, ...]
    inputs: int
    types: tuple[tuple[np.dtype, tuple[int, ...]], ...]
    name: str = "host"
    offsets: tuple[int, ...] = ()  # the byte offset of each buffer (default 0)


@dataclasses.dataclass(frozen=True)
class Branch:
    """Skip `skip` steps (back, when negative) if the boolean scalar in
    `slot` is false; always, when `slot` is None. The host reads the value,
    so a conditional branch waits for the GPU."""

    slot: int | None
    skip: int


Step = Launch | HostCall | Branch


@dataclasses.dataclass(frozen=True)
class Program:
    """Buffer slots, the steps that fill them in order, and the output slots."""

    slots: list[Slot]
    steps: list[Step]
    outputs: list[int]
    aliases: tuple[tuple[int, int], ...] = ()  # parameter, output

    @property
    def launches(self) -> list[Launch]:
        """The kernel launches among the steps, in order."""
        return [step for step in self.steps if isinstance(step, Launch)]


def share_storage(program: Program) -> Program:
    """`program` with intermediate buffers sharing storage where their
    lifetimes are disjoint.

    A fresh slot that is not an output lives from the first step that binds
    it to the last. Inside a loop the host drives, the steps repeat, so a
    slot bound anywhere in the loop lives for all of it. Slots of equal byte
    size whose lifetimes do not overlap become one slot, which bounds the
    memory of an execution by its widest point instead of its length.
    """
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for k, step in enumerate(program.steps):
        slots = (
            ((step.slot,) if step.slot is not None else ())
            if isinstance(step, Branch)
            else step.slots
        )
        for s in slots:
            first.setdefault(s, k)
            last[s] = k
    loops = [
        (k + 1 + step.skip, k)
        for k, step in enumerate(program.steps)
        if isinstance(step, Branch) and step.skip < 0
    ]
    changed = True
    while changed:
        changed = False
        for start, end in loops:
            for s, begin in list(first.items()):
                if begin <= end and last[s] >= start and (begin > start or last[s] < end):
                    first[s], last[s] = min(begin, start), max(last[s], end)
                    changed = True
    outputs = set(program.outputs)
    shared = [
        s
        for s in sorted(first, key=first.__getitem__)
        if program.slots[s].kind == "fresh" and s not in outputs
    ]
    merged: dict[int, int] = {}
    active: list[tuple[int, int]] = []  # (last step, representative), a heap
    free: dict[int, list[int]] = {}  # byte size to representatives
    size = {s: math.prod(program.slots[s].shape) * program.slots[s].dtype.itemsize for s in shared}
    for s in shared:
        while active and active[0][0] < first[s]:
            _, done = heapq.heappop(active)
            free.setdefault(size[done], []).append(done)
        representative = free[size[s]].pop() if free.get(size[s]) else s
        merged[s] = representative
        heapq.heappush(active, (last[s], representative))
    if all(s == r for s, r in merged.items()):
        return program
    # Renumber: the slots that remain, in order.
    kept = [s for s in range(len(program.slots)) if merged.get(s, s) == s]
    number = {s: k for k, s in enumerate(kept)}
    renamed = {s: number[merged.get(s, s)] for s in range(len(program.slots))}

    def rename(step: Step) -> Step:
        if isinstance(step, Branch):
            slot = None if step.slot is None else renamed[step.slot]
            return dataclasses.replace(step, slot=slot)
        return dataclasses.replace(step, slots=tuple(renamed[s] for s in step.slots))

    return dataclasses.replace(
        program,
        slots=[program.slots[s] for s in kept],
        steps=[rename(step) for step in program.steps],
        outputs=[renamed[s] for s in program.outputs],
    )
