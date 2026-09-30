"""Ref loads and stores, jit inlining, and random key wrapping."""

from __future__ import annotations

import dataclasses
import math

from jax.extend.core import Jaxpr, JaxprEqn

from palladium.emit.addressing import ref_view
from palladium.emit.core import Cursor, Environment, declare, emit_jaxpr, rule, shaped
from palladium.emit.rules.control import _consumed_only_as_scan_xs
from palladium.errors import EmitError


@rule("get")
def _rule_get(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """Load from a Ref, full block or indexed (`x_ref[i, :]`; non-Slice
    dims squeeze, Slice dims are kept).

    An indexed load from a read-only ref binds the pointer view instead
    of copying, since nothing writes through a `const device` ref.
    Full-block loads copy, except a block consumed only as scan xs,
    which binds the ref directly.
    """
    indexer_args = eqn.params["tree"].unflatten(eqn.invars[1:])
    src = env.val(eqn.invars[0])
    out_shape = tuple(int(d) for d in shaped(eqn.outvars[0].aval).shape)
    if indexer_args:
        (indexer,) = indexer_args
        src = ref_view(env, src, indexer)
    viewable = (
        src.readonly
        and src.index_map is None
        and src.space == "device"
        and src.shape
        and math.prod(src.shape) == math.prod(out_shape)
        and (
            bool(indexer_args)
            or (bool(out_shape) and _consumed_only_as_scan_xs(env, eqn.outvars[0]))
        )
    )
    if viewable:
        env.bind(eqn.outvars[0], dataclasses.replace(src, shape=out_shape))
        return
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


@rule("random_wrap", "random_unwrap")
def _rule_random_wrap(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`jax.random.wrap_key_data` and `key_data`: type-level wraps of
    uint32[2] key data; pure aliases."""
    env.bind(eqn.outvars[0], env.val(eqn.invars[0]))
