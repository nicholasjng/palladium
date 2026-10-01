"""Shape and layout primitives: reshape, squeeze, transpose, select_n,
broadcast_in_dim."""

from __future__ import annotations

import dataclasses
import math
import re

from jax.extend.core import JaxprEqn

from palladium.emit.addressing import element_strides, flat_index
from palladium.emit.core import Cursor, CVal, Environment, declare, rule, shaped
from palladium.emit.numeric import unwrapped
from palladium.emit.rules.elementwise import _rule_elementwise, reads_by_flat_index
from palladium.errors import EmitError


@rule("reshape")
def _rule_reshape(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """Alias row-major storage where possible; an axis permutation copies
    directly into the reshaped destination.
    """
    src = env.val(eqn.invars[0])
    new_shape = eqn.params["new_sizes"]
    perm = eqn.params["dimensions"]
    if perm is not None and tuple(perm) != tuple(range(len(src.shape))):
        dst = declare(env, cursor, eqn.outvars[0])
        _emit_permuted_copy(cursor, src, dst, tuple(perm))
        return
    if bool(src.shape) != bool(new_shape):
        # A shape-only alias cannot cross between a rank-0 scalar
        # expression and indexable ranked storage.
        dst = declare(env, cursor, eqn.outvars[0])
        scalar = dataclasses.replace(src, expr=src.at("0"), shape=())
        cursor.copy(dst, scalar, 1)
        return
    env.bind(eqn.outvars[0], dataclasses.replace(src, shape=eqn.params["new_sizes"]))


@rule("squeeze")
def _rule_squeeze(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """Remove size-one axes without moving row-major array storage."""
    src = env.val(eqn.invars[0])
    dimensions = tuple(eqn.params["dimensions"])
    if (
        len(set(dimensions)) != len(dimensions)
        or any(axis < 0 or axis >= len(src.shape) for axis in dimensions)
        or any(src.shape[axis] != 1 for axis in dimensions)
    ):
        raise EmitError("squeeze requires distinct axes of size one")
    out_shape = tuple(size for axis, size in enumerate(src.shape) if axis not in dimensions)
    expected_shape = tuple(int(d) for d in shaped(eqn.outvars[0].aval).shape)
    if out_shape != expected_shape:
        raise EmitError(f"squeeze shape mismatch: got {out_shape}, expected {expected_shape}")
    if src.transposed:
        raise EmitError("squeeze of a lazily transposed value is unsupported")
    if out_shape:
        env.bind(eqn.outvars[0], dataclasses.replace(src, shape=out_shape))
        return
    dst = declare(env, cursor, eqn.outvars[0])
    cursor.emit(f"{dst.expr} = {src.read('0')};")


@rule("transpose")
def _rule_transpose(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`jnp.transpose(x, perm)` -> a materialized, permuted copy.

    A rank-2 transpose consumed only as `dot_general` rhs instead binds
    a lazy `transposed` CVal over the untransposed storage, which the
    dot reads unit-stride in the contraction index.
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
    src_strides = element_strides(src.shape)
    dst_strides = element_strides(perm_shape)
    rank = len(src.shape)
    with cursor.loop_nest(tuple(perm_shape), "_t") as idx_vars:
        src_idx = flat_index([(idx_vars[dd], src_strides[perm[dd]]) for dd in range(rank)])
        dst_idx = flat_index([(idx_vars[dd], dst_strides[dd]) for dd in range(rank)])
        cursor.emit(f"{dst.at(dst_idx)} = {src.at(src_idx)};")


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
            cursor.emit(f"{dst.at(index)} = {unwrapped(select(index, 0, len(cases)))};")
    else:
        cursor.emit(f"{dst.expr} = {unwrapped(select('0', 0, len(cases)))};")


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
    if not src.shape and _scalar_reads_only(env, eqn.outvars[0], src):
        # Every element reads the same scalar: a lazy value whose
        # expression ignores the index.
        env.bind(eqn.outvars[0], CVal("", out_shape, src.ctype, lazy=src.expr))
        return
    dst = declare(env, cursor, eqn.outvars[0])
    bcast_dims: tuple[int, ...] = eqn.params["broadcast_dimensions"]
    src_strides = element_strides(src.shape)
    dst_strides = element_strides(dst.shape)
    with cursor.loop_nest(dst.shape, "_b") as idx_vars:
        src_idx = flat_index(
            [
                (idx_vars[od], src_strides[sd])
                for sd, od in enumerate(bcast_dims)
                if src.shape[sd] != 1
            ]
        )
        dst_idx = flat_index(list(zip(idx_vars, dst_strides)))
        cursor.emit(f"{dst.at(dst_idx)} = {src.at(src_idx)};")


_CHEAP_SCALAR = re.compile(r"[A-Za-z_]\w*|-?[\d.]+(e[-+]?\d+)?f?|bfloat\(-?[\d.]+(e[-+]?\d+)?f\)")


def _scalar_reads_only(env: Environment, var, src: CVal) -> bool:
    """Whether a broadcast of `src` can stay a scalar: it is a variable or a
    literal, and every consumer reads it by flat index."""
    return _CHEAP_SCALAR.fullmatch(src.expr) is not None and reads_by_flat_index(env, var)


def _transpose_is_dot_rhs_only(env: Environment, eqn: JaxprEqn) -> bool:
    """Whether this rank-2 `(1, 0)` transpose is consumed only as a
    dot_general rhs and never escapes, so it may lower to a lazy
    `transposed` CVal."""
    if tuple(eqn.params["permutation"]) != (1, 0):
        return False
    outvar = eqn.outvars[0]
    uses = env.consumer_eqns(outvar)
    return (
        bool(uses)
        and not env.escapes(outvar)
        and all(
            use.primitive.name == "dot_general"
            and use.invars[1] is outvar
            and use.invars[0] is not outvar
            for use in uses
        )
    )
