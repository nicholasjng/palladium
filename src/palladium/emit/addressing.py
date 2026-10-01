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
    """Keep address sums and products structured until they are rendered as MSL.

    Attributes
    ----------
    op : {"raw", "add", "mul"}
        Expression node kind; construct nodes through the class helpers.
    value : str, int, or None, optional
        Literal or opaque source for a raw node, otherwise None.
    args : tuple of CExpr, optional
        Child nodes for sums and products; empty for raw nodes.

    Notes
    -----
    This small tree folds zero and unit terms; it is not a general MSL expression parser.
    Raw fragments must already have the precedence needed by the surrounding address
    expression.
    """

    op: str
    value: str | int | None = None
    args: tuple[CExpr, ...] = ()

    @classmethod
    def raw(cls, value: str | int) -> CExpr:
        """Wrap a literal or source fragment as an address-expression leaf.

        Parameters
        ----------
        value : str or int
            Integer literal or opaque MSL expression.

        Returns
        -------
        CExpr
            Raw leaf; strings of decimal digits with optional leading minus are normalized
            to integers.
        """
        if isinstance(value, str) and value.lstrip("-").isdigit():
            value = int(value)
        return cls("raw", value)

    @classmethod
    def add(cls, *terms: CExpr) -> CExpr:
        """Combine address terms while flattening sums and removing literal zeros.

        Parameters
        ----------
        *terms : CExpr
            Terms to add, in source order.

        Returns
        -------
        CExpr
            Sum node; an empty sum renders as "0".
        """
        flattened = tuple(
            arg for term in terms for arg in (term.args if term.op == "add" else (term,))
        )
        kept = tuple(term for term in flattened if not (term.op == "raw" and term.value == 0))
        return cls("add", args=kept)

    @classmethod
    def mul(cls, left: CExpr, right: CExpr) -> CExpr:
        """Multiply address terms while folding literal zero and unit factors.

        Parameters
        ----------
        left : CExpr
            Left factor.
        right : CExpr
            Right factor.

        Returns
        -------
        CExpr
            Product or simplified operand; general constant multiplication is left to MSL.
        """
        if (left.op == "raw" and left.value == 0) or (right.op == "raw" and right.value == 0):
            return cls.raw(0)
        if left.op == "raw" and left.value == 1:
            return right
        if right.op == "raw" and right.value == 1:
            return left
        return cls("mul", args=(left, right))

    def render(self) -> str:
        """Render a structured address expression as MSL source.

        Returns
        -------
        str
            Raw leaf, product, or sum; an empty sum becomes "0".

        Raises
        ------
        AssertionError
            The node operation is unknown.
        """
        if self.op == "raw":
            return str(self.value)
        if self.op == "mul":
            return f"{self.args[0].render()} * {self.args[1].render()}"
        if self.op == "add":
            return " + ".join(term.render() for term in self.args) or "0"
        raise AssertionError(self.op)


@dataclasses.dataclass(frozen=True)
class BlockLayout:
    """Describe physical block extents and backing-array element strides.

    Attributes
    ----------
    shape : tuple of int
        Unsqueezed block extents, with squeezed axes represented by one.
    strides : tuple of int
        Row-major backing-array strides, measured in elements.

    Notes
    -----
    Block indices are measured in tiles. Shape and strides must have equal rank; use
    ``from_info`` to construct them from traced metadata.
    """

    shape: tuple[int, ...]
    strides: tuple[int, ...]

    @classmethod
    def from_info(cls, info: BlockInfo) -> BlockLayout:
        """Derive an unsqueezed physical layout from traced operand metadata.

        Parameters
        ----------
        info : BlockInfo
            Operand description containing backing and full block shapes.

        Returns
        -------
        BlockLayout
            Layout with squeezed axes restored as unit extents and row-major element
            strides.
        """
        return cls(full_block_shape(info), element_strides(info.array_shape))

    def offset(self, indices: list[str]) -> str:
        """Render the backing-array element offset for a list of block indices.

        Parameters
        ----------
        indices : list of str
            MSL block-index expressions, one per physical array axis.

        Returns
        -------
        str
            Sum of index times block extent times backing stride, with zero and unit terms
            folded.

        Raises
        ------
        ValueError
            Index, block, and stride ranks do not match.
        """
        return CExpr.add(
            *(
                CExpr.mul(CExpr.raw(index), CExpr.raw(block * stride))
                for index, block, stride in zip(indices, self.shape, self.strides, strict=True)
            )
        ).render()

    def alignment(self) -> int:
        """Find the element-offset multiple preserved by every possible block origin.

        Returns
        -------
        int
            Gcd of block-extent times backing-stride terms; zero for a rank-zero layout.

        Raises
        ------
        ValueError
            Block and stride ranks do not match.

        Notes
        -----
        This is relative to an aligned buffer base, not an absolute byte-alignment guarantee.
        """
        return math.gcd(
            0, *(block * stride for block, stride in zip(self.shape, self.strides, strict=True))
        )


def element_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute row-major strides measured in elements.

    Parameters
    ----------
    shape : tuple of int
        Logical array dimensions.

    Returns
    -------
    tuple of int
        Product of trailing extents for each axis; () for a scalar.
    """
    return tuple(math.prod(shape[d + 1 :]) for d in range(len(shape)))


def flat_index(terms: list[tuple[str, int]]) -> str:
    """Render a flattened element index from coordinate and stride pairs.

    Parameters
    ----------
    terms : list of tuple of (str, int)
        MSL coordinate expressions and their element strides.

    Returns
    -------
    str
        Sum of coordinate times stride, dropping zero strides and unit multipliers; "0"
        for no terms.
    """
    return CExpr.add(
        *(
            CExpr.raw(var) if stride == 1 else CExpr.mul(CExpr.raw(var), CExpr.raw(stride))
            for var, stride in terms
            if stride != 0
        )
    ).render()


def full_block_shape(info: BlockInfo) -> tuple[int, ...]:
    """Restore squeezed block axes as unit physical extents.

    Parameters
    ----------
    info : BlockInfo
        Traced operand whose full block shape retains original array axes.

    Returns
    -------
    tuple of int
        Rank-matched physical block shape; each None becomes one.
    """
    return tuple(1 if dim is None else dim for dim in info.full_block_shape)


def constant_offset(info: BlockInfo) -> str | None:
    """Fold a constant-only block index map into a literal element offset.

    Parameters
    ----------
    info : BlockInfo
        Operand layout and staged index map.

    Returns
    -------
    str or None
        Decimal element offset, including "0", or None when the map has inputs or
        equations.

    Notes
    -----
    Only literal outputs of an input-free, equation-free map are folded. A map that ignores
    its grid inputs still returns None and is emitted normally.
    """
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
    """Emit a block index map and render this program's backing-array offset.

    Parameters
    ----------
    env : Environment
        Bindings and def-use information for the current jaxpr.
    cursor : Cursor
        Destination for generated MSL statements and storage accounting.
    spec : KernelSpec
        Traced Pallas kernel, including operand layouts, scratch buffers, and grid.
    info : BlockInfo
        Operand whose map returns one block index per physical array axis.

    Returns
    -------
    str
        Element offset computed as the sum of block index times block extent times array
        stride.

    Notes
    -----
    Grid coordinates bind to _pid components. Index-map equations use ordinary emission
    rules; an all-zero map renders as "0" and unsupported primitives propagate emission
    errors.
    """
    pid_vals = [CVal(f"(int){PID[k]}", (), "int") for k in range(len(spec.grid))]
    out_vals = emit_jaxpr(env, cursor, info.index_map_jaxpr.jaxpr, pid_vals)
    return BlockLayout.from_info(info).offset([val.expr for val in out_vals])


def ref_view(env: Environment, ref: CVal, indexer: NDIndexer) -> CVal:
    """Compose logical Ref indexing with a lowered value's physical addressing.

    Parameters
    ----------
    env : Environment
        Bindings and def-use information for the current jaxpr.
    ref : CVal
        Ref storage, including any existing index map and bounds predicate.
    indexer : NDIndexer
        Scalar indices and static-size slices; slice starts may be dynamic.

    Returns
    -------
    CVal
        View retaining a pointer offset when contiguous, otherwise a flat-index map and
        any inherited validity predicate.

    Raises
    ------
    EmitError
        A slice stride is zero or negative.

    Notes
    -----
    Scalar indices squeeze axes and slices preserve them. Positive static strides are
    supported; arbitrary gathers and out-of-range user indices are not. Bounds predicates
    preserve edge-tile guards rather than adding checks for arbitrary indexing.
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
