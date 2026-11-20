"""Address arithmetic for operand blocks and Ref views."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from typing import cast

import jax.experimental.pallas as pl
from jax._src.state.indexing import NDIndexer
from jax.core import Atom

from palladium.emit.core import PID, Cursor, CVal, EmitError, Environment, emit_jaxpr
from palladium.trace import BlockInfo, KernelSpec


def element_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major strides of `shape`, in elements."""
    return tuple(math.prod(shape[d + 1 :]) for d in range(len(shape)))


def flat_index(terms: Iterable[tuple[str, int]]) -> str:
    """Render `sum(index * stride)` as MSL, without zero terms and unit factors.

    An index is a literal or an expression that binds at least as tightly as
    a product; no terms render as "0".
    """
    parts = []
    for index, stride in terms:
        if stride == 0 or (index.lstrip("-").isdigit() and int(index) == 0):
            continue
        if index == "1":
            parts.append(str(stride))
        else:
            parts.append(index if stride == 1 else f"{index} * {stride}")
    return " + ".join(parts) or "0"


def full_block_shape(info: BlockInfo) -> tuple[int, ...]:
    """The block's extent along every array axis; squeezed axes have extent one."""
    return tuple(1 if dim is None else dim for dim in info.full_block_shape)


def block_strides(info: BlockInfo) -> tuple[int, ...]:
    """Elements between the origins of neighbouring blocks, along every array axis."""
    return tuple(
        block * stride
        for block, stride in zip(
            full_block_shape(info), element_strides(info.array_shape), strict=True
        )
    )


def block_offset(env: Environment, cursor: Cursor, spec: KernelSpec, info: BlockInfo) -> str:
    """Emit `info`'s index map and return this program's element offset into the array.

    Grid coordinates bind to the `_pid` components and the map's equations go
    through the ordinary rules, so an unsupported primitive raises as it does
    in a kernel body. A map of constants renders a literal offset.
    """
    pids = [CVal(f"(int){PID[k]}", (), "int") for k in range(len(spec.grid))]
    origin = emit_jaxpr(env, cursor, info.index_map_jaxpr.jaxpr, pids)
    return flat_index(zip((o.expr for o in origin), block_strides(info), strict=True))


def ref_view(env: Environment, ref: CVal, indexer: NDIndexer) -> CVal:
    """Compose a Ref index with the Ref's own addressing.

    Scalar indices squeeze their axis; slices keep it, with a static size, a
    positive static stride and a start that may be dynamic. The view stays a
    pointer when its elements are contiguous and becomes a flat-index map
    otherwise, inheriting any bounds predicate. That predicate guards edge
    tiles; indices outside the block are not checked. Raises EmitError for a
    slice stride below one.
    """
    strides = element_strides(indexer.shape)
    terms: list[tuple[str, int]] = []
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
            terms.append((expr, stride))
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

    offset = flat_index(terms)
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
