"""Lowerings for one part of the MSL execution model."""

from __future__ import annotations

import dataclasses
import math

from jax.extend.core import JaxprEqn

from palladium.emit.core import (
    Cursor,
    CVal,
    EmitError,
    Environment,
    _element_strides,
    _flat_index,
    _transpose_is_dot_rhs_only,
    _unwrapped,
    declare,
    rule,
    shaped,
)
from palladium.emit.rules.elementwise import _rule_elementwise


@rule("reshape")
def _rule_reshape(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """Reshape aliases row-major storage where possible. An axis
    permutation copies directly into the reshaped destination, without
    allocating an intermediate transposed array.
    """
    src = env.val(eqn.invars[0])
    new_shape = eqn.params["new_sizes"]
    perm = eqn.params["dimensions"]
    if perm is not None and tuple(perm) != tuple(range(len(src.shape))):
        dst = declare(env, cursor, eqn.outvars[0])
        _emit_permuted_copy(cursor, src, dst, tuple(perm))
        return
    if bool(src.shape) != bool(new_shape):
        # Rank zero uses a scalar expression, while ranked values use
        # indexable storage. A shape-only alias cannot cross that boundary.
        dst = declare(env, cursor, eqn.outvars[0])
        scalar = dataclasses.replace(src, expr=src.at("0"), shape=())
        cursor.copy(dst, scalar, 1)
        return
    env.bind(eqn.outvars[0], dataclasses.replace(src, shape=eqn.params["new_sizes"]))


@rule("transpose")
def _rule_transpose(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`x.T` / `jnp.transpose(x, perm)` -> a materialized, permuted copy.

    A real copy, not a view like reshape: transposed storage is different
    bytes in the row-major flat-array model. `a.T` inside a dot product
    stages as a standalone `transpose` equation ahead of `dot_general`;
    the contraction itself stays (lhs dim 1, rhs dim 0).

    Exception: a rank-2 transpose consumed only as `dot_general` rhs
    never materializes. It binds a `transposed` CVal (untransposed
    storage plus a flag) and the dot reads element `(k, j)` at
    `[j * K + k]`, unit-stride in the contraction index for both
    operands. Gated on `Environment.consumers` so no generic
    flat-indexing rule can observe the transposed CVal.
    """
    src = env.val(eqn.invars[0])
    outvar = eqn.outvars[0]
    if _transpose_is_dot_rhs_only(env, eqn) and not src.transposed:
        env.bind(
            outvar,
            dataclasses.replace(src, shape=(src.shape[1], src.shape[0]), transposed=True),
        )
        return
    perm: tuple[int, ...] = eqn.params["permutation"]
    dst = declare(env, cursor, eqn.outvars[0])
    _emit_permuted_copy(cursor, src, dst, perm)


def _emit_permuted_copy(cursor: Cursor, src: CVal, dst: CVal, perm: tuple[int, ...]) -> None:
    """Copy the permuted source in row-major order into flat dst storage.

    The destination may reshape that order; its rank is independent of
    the iteration domain, which is the source shape after permutation.
    """
    perm_shape = tuple(src.shape[d] for d in perm)
    src_strides = _element_strides(src.shape)
    dst_strides = _element_strides(perm_shape)
    rank = len(src.shape)
    idx_vars = [cursor.fresh(f"_t{d}") for d in range(rank)]

    def emit_loops(d: int) -> None:
        if d == rank:
            src_idx = _flat_index([(idx_vars[dd], src_strides[perm[dd]]) for dd in range(rank)])
            dst_idx = _flat_index([(idx_vars[dd], dst_strides[dd]) for dd in range(rank)])
            cursor.emit(f"{dst.at(dst_idx)} = {src.at(src_idx)};")
            return
        with cursor.block(
            f"for (uint {idx_vars[d]} = 0; {idx_vars[d]} < {perm_shape[d]}; ++{idx_vars[d]})"
        ):
            emit_loops(d + 1)

    emit_loops(0)


@rule("select_n")
def _rule_select_n(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """Select among equal-shaped cases using a scalar or per-element index.

    Integer selection uses a balanced comparison tree, as JAX's lowering
    does. Out-of-range indices select the nearest endpoint case; JAX's
    public contract leaves those indices implementation-defined.
    """
    which, *cases = [env.val(v) for v in eqn.invars]
    if which.ctype not in ("bool", "int", "uint"):
        raise EmitError(f"select_n requires a bool or integer index, got {which.ctype}")
    if not cases or (which.ctype == "bool" and len(cases) > 2):
        raise EmitError("select_n needs at least one case and bool permits at most two")
    if len(cases) == 1:
        env.bind(eqn.outvars[0], cases[0])
        return
    if which.ctype == "bool":
        # Preserve the established boolean codegen, including its snapshots.
        _rule_elementwise(env, cursor, eqn)
        return

    dst = declare(env, cursor, eqn.outvars[0])

    def select(index: str, lo: int, hi: int) -> str:
        if hi - lo == 1:
            return cases[lo].at(index)
        mid = (lo + hi) // 2
        threshold = f"{mid}u" if which.ctype == "uint" else str(mid)
        left, right = select(index, lo, mid), select(index, mid, hi)
        return f"({which.at(index)} < {threshold} ? {left} : {right})"

    if dst.shape:
        with cursor.loop(dst.size) as index:
            cursor.emit(f"{dst.at(index)} = {_unwrapped(select(index, 0, len(cases)))};")
    else:
        cursor.emit(f"{dst.expr} = {_unwrapped(select('0', 0, len(cases)))};")


@rule("broadcast_in_dim")
def _rule_broadcast_in_dim(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`jnp.broadcast_to`/rank-matching before an elementwise op.

    Size-preserving broadcasts alias; the rest materialize a copy,
    replicating size-1 (or absent) input dims.
    """
    src = env.val(eqn.invars[0])
    aval = shaped(eqn.outvars[0].aval)
    out_shape = tuple(int(d) for d in aval.shape)
    if src.shape and src.size == math.prod(out_shape):
        env.bind(eqn.outvars[0], dataclasses.replace(src, shape=out_shape))
        return
    dst = declare(env, cursor, eqn.outvars[0])
    bcast_dims: tuple[int, ...] = eqn.params["broadcast_dimensions"]
    src_strides = _element_strides(src.shape)
    dst_strides = _element_strides(dst.shape)
    rank = len(dst.shape)
    idx_vars = [cursor.fresh(f"_b{d}") for d in range(rank)]

    def emit_loops(d: int) -> None:
        if d == rank:
            src_idx = _flat_index(
                [
                    (idx_vars[od], src_strides[sd])
                    for sd, od in enumerate(bcast_dims)
                    if src.shape[sd] != 1
                ]
            )
            dst_idx = _flat_index(list(zip(idx_vars, dst_strides)))
            cursor.emit(f"{dst.at(dst_idx)} = {src.at(src_idx)};")
            return
        with cursor.block(
            f"for (uint {idx_vars[d]} = 0; {idx_vars[d]} < {dst.shape[d]}; ++{idx_vars[d]})"
        ):
            emit_loops(d + 1)

    emit_loops(0)
