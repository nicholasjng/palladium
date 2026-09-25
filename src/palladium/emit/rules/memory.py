"""Lowerings for one part of the MSL execution model."""

from __future__ import annotations

import dataclasses
import math

from jax.extend.core import Jaxpr, JaxprEqn

from palladium.emit.core import (
    Cursor,
    EmitError,
    Environment,
    declare,
    emit_jaxpr,
    ref_view,
    rule,
    shaped,
)


@rule("get")
def _rule_get(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """Load from a Ref: `y = x_ref[...]` (full block) or an indexed access
    like `y = x_ref[i, :]` (non-Slice dims squeeze into the offset, Slice
    dims are kept in the loaded shape).

    An indexed load from a read-only (input) ref binds the pointer view
    instead of copying: nothing can write through a `const device` ref,
    so the view has snapshot semantics. Full-block loads copy, since
    those blocks are re-read many times -- except a block consumed only
    as scan xs, read once per element, which binds the ref directly.
    """
    indexer_args = eqn.params["tree"].unflatten(eqn.invars[1:])
    src = env.val(eqn.invars[0])
    out_aval = shaped(eqn.outvars[0].aval)
    from palladium.emit.rules.control import _consumed_only_as_scan_xs

    if (
        not indexer_args
        and src.readonly
        and src.index_map is None
        and src.space == "device"
        and src.shape
        and out_aval.shape
        and math.prod(src.shape) == math.prod(tuple(int(d) for d in out_aval.shape))
        and _consumed_only_as_scan_xs(env, eqn.outvars[0])
    ):
        env.bind(
            eqn.outvars[0],
            dataclasses.replace(src, shape=tuple(int(d) for d in out_aval.shape)),
        )
        return
    if indexer_args:
        (indexer,) = indexer_args
        view = ref_view(env, src, indexer)
        aval = shaped(eqn.outvars[0].aval)
        if (
            view.readonly
            and view.index_map is None
            and view.space == "device"
            and view.shape
            and math.prod(view.shape) == math.prod(tuple(int(d) for d in aval.shape))
        ):
            env.bind(
                eqn.outvars[0],
                dataclasses.replace(view, shape=tuple(int(d) for d in aval.shape)),
            )
            return
        src = view
    dst = declare(env, cursor, eqn.outvars[0])
    cursor.copy(dst, src, dst.size)


@rule("swap")
def _rule_swap(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """Store to a Ref: `o_ref[...] = y` (full block) or an indexed store
    like `o_ref[i, :] = y`.

    A used result snapshots the old contents before the write. Ordinary
    stores discard that result and need no snapshot.
    """
    ref, value = eqn.invars[0], eqn.invars[1]
    indexer_args = eqn.params["tree"].unflatten(eqn.invars[2:])
    dst_ref = env.val(ref)
    if indexer_args:
        (indexer,) = indexer_args
        dst_ref = ref_view(env, dst_ref, indexer)
    stored = env.val(value)
    outvar = eqn.outvars[0]
    used = bool(env.consumer_eqns(outvar)) or env.escapes(outvar)
    if used:
        old = declare(env, cursor, outvar)
        cursor.copy(old, dst_ref, old.size)
    if stored.expr != dst_ref.expr:
        cursor.copy(dst_ref, stored, dst_ref.size)
    if not used:
        env.bind(outvar, dst_ref)


@rule("jit")
def _inline_jit(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """Inline a jit-wrapped call by walking its body jaxpr."""
    inner: Jaxpr = eqn.params["jaxpr"]
    body = inner.jaxpr
    if inner.consts:
        raise EmitError("jit with consts is unsupported")

    invals = [env.val(invar) for invar in eqn.invars]
    outvals = emit_jaxpr(env, cursor, body, invals)

    for outvar, val in zip(eqn.outvars, outvals, strict=True):
        env.bind(outvar, val)


@rule("random_wrap")
def _rule_random_wrap(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`jax.random.wrap_key_data`: pure type-level wrap (uint32[2] array
    to a `key<fry>[]`-typed value), no MSL. Alias, like `program_id`."""
    env.bind(eqn.outvars[0], env.val(eqn.invars[0]))


@rule("random_unwrap")
def _rule_random_unwrap(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`jax.random.key_data`: the inverse of `random_wrap`, equally pure
    aliasing (a `key<fry>[]`-typed value is a uint32[2] array underneath
    the whole time; nothing to convert)."""
    env.bind(eqn.outvars[0], env.val(eqn.invars[0]))
