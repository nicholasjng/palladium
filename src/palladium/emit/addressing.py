"""Address arithmetic for operand blocks and Ref views."""

from __future__ import annotations

import dataclasses
import math
from typing import cast

import jax.experimental.pallas as pl
from jax._src.state.indexing import NDIndexer
from jax.core import Atom
from jax.extend.core import Literal

from palladium.emit.core import PID, Cursor, CVal, EmitError, Environment, emit_jaxpr
from palladium.trace import BlockInfo, KernelSpec


@dataclasses.dataclass(frozen=True)
class CExpr:
    """Integer-expression tree for address arithmetic; sums and products
    stay structured until rendering so zero terms fold without parsing C.
    """

    op: str
    value: str | int | None = None
    args: tuple[CExpr, ...] = ()

    @classmethod
    def raw(cls, value: str | int) -> CExpr:
        if isinstance(value, str) and value.lstrip("-").isdigit():
            value = int(value)
        return cls("raw", value)

    @classmethod
    def add(cls, *terms: CExpr) -> CExpr:
        flattened = tuple(
            arg for term in terms for arg in (term.args if term.op == "add" else (term,))
        )
        kept = tuple(term for term in flattened if not (term.op == "raw" and term.value == 0))
        return cls("add", args=kept)

    @classmethod
    def mul(cls, left: CExpr, right: CExpr) -> CExpr:
        if (left.op == "raw" and left.value == 0) or (right.op == "raw" and right.value == 0):
            return cls.raw(0)
        return cls("mul", args=(left, right))

    def render(self) -> str:
        if self.op == "raw":
            return str(self.value)
        if self.op == "mul":
            return f"{self.args[0].render()} * {self.args[1].render()}"
        if self.op == "add":
            return " + ".join(term.render() for term in self.args) or "0"
        raise AssertionError(self.op)


@dataclasses.dataclass(frozen=True)
class BlockLayout:
    """Shared logical/physical facts for one Pallas operand block."""

    shape: tuple[int, ...]
    strides: tuple[int, ...]

    @classmethod
    def from_info(cls, info: BlockInfo) -> BlockLayout:
        return cls(full_block_shape(info), element_strides(info.array_shape))

    def offset(self, indices: list[str]) -> str:
        return CExpr.add(
            *(
                CExpr.mul(CExpr.raw(index), CExpr.raw(block * stride))
                for index, block, stride in zip(indices, self.shape, self.strides, strict=True)
            )
        ).render()

    def alignment(self) -> int:
        return math.gcd(
            0, *(block * stride for block, stride in zip(self.shape, self.strides, strict=True))
        )


def element_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.prod(shape[d + 1 :]) for d in range(len(shape)))


def flat_index(terms: list[tuple[str, int]]) -> str:
    """C expression for `sum(var * stride for var, stride in terms)`,
    omitting the `* 1` for a unit stride and any zero-stride term."""
    return CExpr.add(
        *(
            CExpr.raw(var) if stride == 1 else CExpr.mul(CExpr.raw(var), CExpr.raw(stride))
            for var, stride in terms
            if stride != 0
        )
    ).render()


def full_block_shape(info: BlockInfo) -> tuple[int, ...]:
    """block_shape with squeezed dims restored as 1, rank-matched to array."""
    return tuple(1 if dim is None else dim for dim in info.full_block_shape)


def constant_offset(info: BlockInfo) -> str | None:
    """Fold index maps with no inputs (gridless or constant) to an offset."""
    imj = info.index_map_jaxpr.jaxpr
    if imj.invars or imj.eqns:
        return None
    strides = element_strides(info.array_shape)
    full_block = full_block_shape(info)
    off = 0
    for o, b, s in zip(imj.outvars, full_block, strides, strict=True):
        # No invars and no eqns leaves only inline constants as outputs.
        assert isinstance(o, Literal), o
        off += int(o.val) * b * s
    return str(off)


def block_offset(env: Environment, cursor: Cursor, spec: KernelSpec, info: BlockInfo) -> str:
    """Element offset of this program instance's block, as a C expression.

    The index map is a jaxpr over grid indices (bound to _pid components),
    recursed through emit_jaxpr. Map outputs are block indices per array
    dim, converted to elements as

        offset = sum(idx[d] * full_block[d] * stride[d] for d in dims)

    Zero-literal terms are elided; falls back to "0" for an all-zero map.
    """
    pid_vals = [CVal(f"(int){PID[k]}", (), "int") for k in range(len(spec.grid))]
    out_vals = emit_jaxpr(env, cursor, info.index_map_jaxpr.jaxpr, pid_vals)
    return BlockLayout.from_info(info).offset([val.expr for val in out_vals])


def ref_view(env: Environment, ref: CVal, indexer: NDIndexer) -> CVal:
    """Compose logical Ref indexing with its underlying storage addressing.

    Slices keep dimensions and scalar indices squeeze them. Contiguous views
    retain pointer offsets; strided/guarded views map flattened logical indices
    to the original allocation, keeping its bounds predicate intact. Positive
    static slice strides are supported; arbitrary gathers are not.
    """
    strides = element_strides(indexer.shape)
    terms: list[CExpr] = []
    kept: list[tuple[int, int]] = []
    steps: dict[int, int] = {}
    align = ref.align
    for d, (idx, stride) in enumerate(zip(indexer.indices, strides, strict=True)):
        if isinstance(idx, pl.Slice):
            if idx.stride < 1:
                raise EmitError(f"ref access dim {d}: stride must be positive")
            steps[d] = idx.stride
            # pl.Slice.size is always static, and a dynamic start
            # is always a jaxpr atom (Var/Literal), never a live Array.
            kept.append((d, cast(int, idx.size)))
            start = idx.start
            expr = str(start) if isinstance(start, int) else env.val(cast(Atom, start)).expr
        else:
            # A non-Slice index here is always a jaxpr atom.
            expr = env.val(cast(Atom, idx)).expr
        if expr != "0":
            terms.append(CExpr.mul(CExpr.raw(expr), CExpr.raw(stride)))
            # A dynamic index contributes its stride as the provable
            # multiple; a literal index contributes its exact offset.
            lit = expr.lstrip("-").isdigit()
            align = math.gcd(align, int(expr) * stride if lit else stride)

    # Squeezed trailing dimensions can make even full slices strided:
    # x[:, 1] is a column, not a contiguous span starting at x[0, 1].
    kept_strides = element_strides(tuple(size for _, size in kept))
    noncontiguous = any(
        size > 1 and strides[d] * steps[d] != expected
        for (d, size), expected in zip(kept, kept_strides, strict=True)
    )

    offset = CExpr.add(*terms).render()
    if noncontiguous or ref.index_map is not None:
        coordinates = [f"(($i / {s}) % {size})" for (_, size), s in zip(kept, kept_strides)]
        flat = f"({offset}) + " + flat_index(
            [(c, strides[d] * steps[d]) for c, (d, _) in zip(coordinates, kept)]
        )
        address = flat if ref.index_map is None else ref.index_map.replace("$i", f"({flat})")
        valid = None if ref.valid is None else ref.valid.replace("$i", f"({flat})")
        return dataclasses.replace(
            ref, shape=tuple(size for _, size in kept), index_map=address, valid=valid, align=1
        )
    if not kept:
        return CVal(
            expr=f"{ref.expr}[{offset}]",
            shape=(),
            ctype=ref.ctype,
            space=ref.space,
            readonly=ref.readonly,
        )
    expr = f"({ref.expr} + {offset})" if offset != "0" else ref.expr
    return CVal(
        expr=expr,
        shape=tuple(size for _, size in kept),
        ctype=ref.ctype,
        space=ref.space,
        readonly=ref.readonly,
        align=align,
    )
