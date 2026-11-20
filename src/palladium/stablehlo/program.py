"""What a compiled program is: buffer slots and kernel launches."""

from __future__ import annotations

import dataclasses
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
class Program:
    """Buffer slots, the launches that fill them in order, and the output slots."""

    slots: list[Slot]
    launches: list[Launch]
    outputs: list[int]
    aliases: tuple[tuple[int, int], ...] = ()  # parameter, output
