"""MSL operands of a generated kernel: coordinates and the values read at them.

A `Term` carries its text together with what code generation asks of it: the
variables it uses, which decide the scope it may be computed in; its value,
when it is a constant index; and how tightly it binds, so it is parenthesized
only where needed. Index arithmetic folds constants as it is built.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

# How a term's outermost operator binds.
_OPERAND, _PRODUCT, _SUM = 0, 1, 2
_WRAP = 1 << 32  # indices are uint


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Term:
    """An MSL expression. Equal terms have equal text.

    `names` are the variables it uses. `value` is set for a constant index.
    `simple` terms are a name or a literal, which stand for themselves; the
    others are worth a variable when used more than once.
    """

    text: str
    names: frozenset[str] = frozenset()
    value: int | None = None
    simple: bool = True
    binding: int = _OPERAND

    @classmethod
    def name(cls, name: str) -> Term:
        return cls(name, frozenset((name,)))

    @classmethod
    def constant(cls, value: int) -> Term:
        value %= _WRAP
        return cls(f"{value}u", value=value)

    @classmethod
    def call(cls, text: str, *arguments: Term) -> Term:
        """`text`, an expression that is one operand, using `arguments`."""
        names = frozenset().union(*(a.names for a in arguments))
        return cls(text, names, simple=False)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Term) and self.text == other.text

    def __hash__(self) -> int:
        return hash(self.text)

    def __str__(self) -> str:
        """The term as one operand of any operator."""
        return self._bound(_OPERAND)

    def _bound(self, binding: int) -> str:
        return self.text if self.binding <= binding else f"({self.text})"

    def _combined(self, text: str, other: Term | None, binding: int) -> Term:
        names = self.names | other.names if other else self.names
        return Term(text, names, simple=False, binding=binding)

    def __add__(self, other: Term | int) -> Term:
        if isinstance(other, int):
            other = Term.constant(other)
        if self.value is not None and other.value is not None:
            return Term.constant(self.value + other.value)
        if self.value == 0:
            return other
        if other.value == 0:
            return self
        return self._combined(f"{self._bound(_SUM)} + {other._bound(_SUM)}", other, _SUM)

    def __sub__(self, other: Term | int) -> Term:
        if isinstance(other, int):
            other = Term.constant(other)
        if self.value is not None and other.value is not None:
            return Term.constant(self.value - other.value)
        if other.value == 0:
            return self
        return self._combined(f"{self._bound(_SUM)} - {other._bound(_PRODUCT)}", other, _SUM)

    def __rsub__(self, other: int) -> Term:
        return Term.constant(other) - self

    def __mul__(self, factor: int) -> Term:
        if self.value is not None or factor == 0:
            return Term.constant((self.value or 0) * factor)
        if factor == 1:
            return self
        return self._combined(f"{self._bound(_PRODUCT)} * {factor}u", None, _PRODUCT)

    def __floordiv__(self, divisor: int) -> Term:
        # A zero divisor comes from an empty shape, whose kernel runs no thread.
        if divisor == 0:
            return ZERO
        if self.value is not None:
            return Term.constant(self.value // divisor)
        if divisor == 1:
            return self
        return self._combined(f"{self._bound(_PRODUCT)} / {divisor}u", None, _PRODUCT)

    def __mod__(self, modulus: int) -> Term:
        if modulus in (0, 1):
            return ZERO
        if self.value is not None:
            return Term.constant(self.value % modulus)
        return self._combined(f"{self._bound(_PRODUCT)} % {modulus}u", None, _PRODUCT)


ZERO = Term.constant(0)

Coords = tuple[Term, ...]


def strides_of(shape: Sequence[int]) -> list[int]:
    return [math.prod(shape[k + 1 :]) for k in range(len(shape))]


def linear(coords: Sequence[Term], shape: Sequence[int]) -> Term:
    """The flat row-major index of `coords` in `shape`."""
    flat = ZERO
    for c, s, n in zip(coords, strides_of(shape), shape, strict=True):
        if n != 1:  # the only coordinate of a unit dimension is zero
            flat += c * s
    return flat


def delinear(flat: Term, shape: Sequence[int]) -> Coords:
    """Coordinates of a flat row-major index into `shape`.

    The leading non-unit dimension needs no modulus: the flat index is below
    the size.
    """
    coords = []
    leading = True
    for s, n in zip(strides_of(shape), shape, strict=True):
        if n == 1:
            coords.append(ZERO)
            continue
        coords.append(flat // s if leading else flat // s % n)
        leading = False
    return tuple(coords)
