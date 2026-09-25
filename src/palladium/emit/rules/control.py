"""Lowerings for one part of the MSL execution model."""

from __future__ import annotations

import dataclasses
import math

from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var

from palladium.emit.core import (
    _PID,
    _TID,
    _TPT,
    CTYPES,
    Cursor,
    CVal,
    EmitError,
    Environment,
    declare,
    emit_jaxpr,
    rule,
    shaped,
)


@rule("program_id")
def _rule_program_id(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`pl.program_id(axis)` -> a component of _pid, as a rank-0 int.

    Pure aliasing, no storage or code. The (int) cast keeps index arithmetic signed.
    """
    axis: int = eqn.params["axis"]
    env.bind(eqn.outvars[0], CVal(f"(int){_PID[axis]}", (), "int"))


@rule("palladium_barrier")
def _rule_barrier(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`palladium.barrier()` -> `threadgroup_barrier(mem_flags::mem_threadgroup)`.

    Emitted verbatim wherever the author placed it; palladium never
    infers barrier placement from a hazard analysis. Zero outputs, so
    nothing to bind -- the primitive carries a JAX effect purely to
    survive DCE on the way here.
    """
    cursor.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")


@rule("palladium_thread_index")
def _rule_thread_index(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """Linearize the actual threadgroup coordinates, x fastest, as int32.

    Pure aliasing, same as `program_id`. The (int) cast keeps index
    arithmetic signed.
    """
    env.bind(eqn.outvars[0], CVal(f"(int){_TID}", (), "int"))


@rule("palladium_threads_per_threadgroup")
def _rule_threads_per_threadgroup(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """Product of the actual threadgroup dimensions, as int32.

    Reports the *actual* size of this threadgroup, which for the final
    group of a non-uniform dispatch is smaller than the requested size.
    That is the whole reason cooperative loops bound themselves with this
    instead of a compile-time constant.
    """
    env.bind(eqn.outvars[0], CVal(f"(int){_TPT}", (), "int"))


@rule("scan")
def _rule_scan(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`lax.fori_loop` / `lax.scan` -> a C for-loop.

    Consts pass through unchanged; carries get fresh mutable loop
    variables bound on the outvars; scanned xs bind per-iteration
    strided views; stacked ys stream straight to a device ref
    (`_ys_stream_target`) or fill thread-local stacked storage, bounded
    by the per-thread stack.

    Copy-back runs in two phases: scan updates all carries
    simultaneously, so phase 1 snapshots reads that alias other carries
    into temps, then phase 2 overwrites the carries. ys slices are
    stored before copy-back, while the body outputs still hold this
    iteration's values.
    """
    length: int = eqn.params["length"]
    body: Jaxpr = eqn.params["jaxpr"]
    reverse: bool = eqn.params.get("reverse", False)
    shape = _scan_shape(env, eqn)
    num_carry = shape.num_carry

    const_vals = [env.val(v) for v in eqn.invars[: shape.num_consts]]
    xs_vals = [env.val(v) for v in eqn.invars[shape.first_xs :]]
    if any(xs.transposed for xs in xs_vals):
        raise EmitError("scanning a lazily transposed xs is unsupported")

    carries = []
    for invar, outvar in zip(
        eqn.invars[shape.num_consts : shape.first_xs],
        eqn.outvars[:num_carry],
        strict=True,
    ):
        dst = declare(env, cursor, outvar)
        cursor.copy(dst, env.val(invar), dst.size)
        carries.append(dst)

    ys_targets = []
    for outvar in eqn.outvars[num_carry:]:
        target = _ys_stream_target(env, outvar)
        if target is not None:
            env.bind(outvar, target)
        else:
            target = declare(env, cursor, outvar)
        ys_targets.append(target)

    # xs/ys keep their stacked positions under reverse, so the same
    # idx-based addressing serves both directions.
    with cursor.loop(length, "_s", reverse=reverse) as idx:
        x_slices = [
            _xs_slice(xs, bv, idx)
            for xs, bv in zip(xs_vals, body.invars[shape.first_xs :], strict=True)
        ]
        outs = emit_jaxpr(env, cursor, body, const_vals + carries + x_slices)
        for target, y in zip(ys_targets, outs[num_carry:], strict=True):
            cursor.copy(target.slot(idx, (y.size,)), y, y.size)
        _copy_back_carries(cursor, outs[:num_carry], carries)


@dataclasses.dataclass(frozen=True)
class ScanShape:
    """The operand split of one scan eqn: `eqn.invars` is consts,
    carries, xs in that order (xs from index `first_xs`), and
    `eqn.outvars` is carries then stacked ys."""

    num_consts: int
    num_carry: int
    num_xs: int

    @property
    def first_xs(self) -> int:
        return self.num_consts + self.num_carry


def _scan_shape(env: Environment, eqn: JaxprEqn) -> ScanShape:
    """The operand split for `eqn`, memoized per emission."""
    key = ("scan_shape", id(eqn))
    shape = env.rule_cache.get(key)
    if shape is None:
        shape = _split_scan_operands(eqn, eqn.params["jaxpr"], eqn.params["length"])
        env.rule_cache[key] = shape
    assert isinstance(shape, ScanShape)
    return shape


def _split_scan_operands(eqn: JaxprEqn, body: Jaxpr, length: int) -> ScanShape:
    """Structurally classify scan operands into (consts, carries, xs) and
    outputs into (carries, ys).

    Derived from avals, not params: jax 0.11's scan params carry no
    num_carry/num_consts. A carry keeps its aval across eqn and body,
    a const likewise, while xs/ys gain a leading `length` dim; carries
    lead the outvars and trail-align against the body's x slices, so the
    counts fall out of pairwise aval comparison.
    """

    def sig(atom: Var | Literal) -> tuple[tuple[int, ...], str]:
        # Consts may be Refs, whose aval is not a ShapedArray but still
        # exposes .shape/.dtype.
        aval = atom.aval
        shape = tuple(int(d) for d in getattr(aval, "shape", ()))
        return shape, str(getattr(aval, "dtype", ""))

    def stacked(outer: Var | Literal, inner: Var | Literal) -> bool:
        (o_shape, o_dtype), (i_shape, i_dtype) = sig(outer), sig(inner)
        return o_dtype == i_dtype and o_shape == (length, *i_shape)

    if len(body.outvars) != len(eqn.outvars) or len(body.invars) != len(eqn.invars):
        raise EmitError("unrecognized scan structure: eqn/body arity mismatch")

    num_carry = 0
    while num_carry < len(eqn.outvars) and sig(eqn.outvars[num_carry]) == sig(
        body.outvars[num_carry]
    ):
        num_carry += 1
    for ov, bv in zip(eqn.outvars[num_carry:], body.outvars[num_carry:], strict=True):
        if not stacked(ov, bv):
            raise EmitError(
                "unrecognized scan structure: outputs after the carries must all be stacked ys"
            )

    num_xs = 0
    while num_xs < len(eqn.invars) - num_carry and stacked(
        eqn.invars[len(eqn.invars) - 1 - num_xs],
        body.invars[len(body.invars) - 1 - num_xs],
    ):
        num_xs += 1
    num_consts = len(eqn.invars) - num_carry - num_xs
    for iv, bv in zip(
        eqn.invars[num_consts : num_consts + num_carry],
        body.invars[num_consts : num_consts + num_carry],
        strict=True,
    ):
        if sig(iv) != sig(bv):
            raise EmitError(
                "unrecognized scan structure: carry avals disagree between eqn and body"
            )
    return ScanShape(num_consts, num_carry, num_xs)


def _xs_slice(xs: CVal, body_invar: Var, idx: str) -> CVal:
    """This iteration's x: a strided view into the stacked xs value
    (an immutable SSA value, so no copy is needed)."""
    aval = shaped(body_invar.aval)
    shape = tuple(int(d) for d in aval.shape)
    if not shape:
        return CVal(expr=xs.at(idx), shape=(), ctype=xs.ctype)
    return xs.slot(idx, shape)


def _consumed_only_as_scan_xs(env: Environment, var: Var) -> bool:
    """Whether every consumer of `var` is a scan taking it as xs.

    xs elements are read once per scan, so a device view costs what the
    copy would; consts are re-read every iteration and keep the copy.
    """
    if env.escapes(var) or not env.consumer_eqns(var):
        return False
    for eqn in env.consumer_eqns(var):
        if eqn.primitive.name != "scan":
            return False
        try:
            shape = _scan_shape(env, eqn)
        except EmitError:
            # Malformed scan: fall back to the copy; the scan rule will
            # raise the real diagnostic when it gets there.
            return False
        if any(iv is var for iv in eqn.invars[: shape.first_xs]):
            return False
    return True


def _ys_stream_target(env: Environment, outvar: Var) -> CVal | None:
    """The device ref to stream a stacked ys into, or None for the
    thread-local fallback.

    Streaming writes the ref earlier than its swap; that is unobservable
    only while the ref feeds nothing but that one full-block swap and
    shares its buffer with no other ref (`env.no_stream_refs`). The swap
    then degenerates to a self-copy, which `_rule_swap` skips.
    """
    swap = env.sole_consumer(outvar)
    if (
        swap is None
        or swap.primitive.name != "swap"
        or len(swap.invars) != 2  # an indexed swap targets a sub-block
        or swap.invars[1] is not outvar
    ):
        return None
    ref_var = swap.invars[0]
    if (
        not isinstance(ref_var, Var)  # a ref is always a Var, never a Literal
        or env.sole_consumer(ref_var) is not swap
        or ref_var in env.no_stream_refs
    ):
        return None
    ref = env.bindings.get(ref_var)
    if (
        ref is None
        or ref.space != "device"
        or ref.readonly
        or ref.transposed
        or ref.index_map is not None
    ):
        return None
    aval = shaped(outvar.aval)
    if ref.size != math.prod(aval.shape) or ref.ctype != CTYPES[str(aval.dtype)]:
        return None
    return ref


def _copy_back_carries(cursor: Cursor, outs: list[CVal], carries: list[CVal]) -> None:
    """Write loop-body outputs back into their carry storage.

    All carries update simultaneously, so every source element is read
    into a scalar before any carry element is written: one fused loop
    per element count (grouping by size is safe because an output that
    aliases a carry has that carry's aval, so aliasing never crosses
    sizes). Self-forwards are skipped. Shared by the scan and while
    lowerings. Pure text emission, no bindings: takes a Cursor, not an
    Environment.

    Carry permutations (an output that IS another carry) additionally
    read and write through volatile pointers, working around a Metal
    compiler bug: with three or more thread-local array temporaries live
    in the loop, the optimizer forwards a permuted carry's read across
    the write it must precede, producing wrong results on M-series GPUs.
    Snapshot arrays, fused loops, and hoisted declarations all still
    miscompile; volatile on the hazard endpoints is the narrowest fix
    that survives. The permutation-free path stays non-volatile.
    """
    updates = [
        (out, carry) for out, carry in zip(outs, carries, strict=True) if out.expr != carry.expr
    ]
    carry_exprs = {c.expr for c in carries}

    def stage(group: list[tuple[CVal, CVal]], index: str) -> None:
        hazard = any(out.expr in carry_exprs for out, _ in group)

        def read(out: CVal, carry: CVal) -> str:
            # Only carry storage is read volatile; fresh SSA values and
            # literals are not part of the hazard (and a literal has no
            # address to cast).
            if not (hazard and out.expr in carry_exprs):
                return out.at(index)
            if carry.shape:
                return f"((volatile thread {carry.ctype}*){out.expr})[{index}]"
            return f"(*(volatile thread {carry.ctype}*)&{out.expr})"

        def write(carry: CVal, t: str) -> str:
            if not hazard:
                return f"{carry.at(index)} = {t};"
            if carry.shape:
                return f"((volatile thread {carry.ctype}*){carry.expr})[{index}] = {t};"
            return f"(*(volatile thread {carry.ctype}*)&{carry.expr}) = {t};"

        temps = []
        for out, carry in group:
            t = cursor.fresh("_cb")
            cursor.emit(f"{carry.ctype} {t} = {read(out, carry)};")
            temps.append(t)
        for (_, carry), t in zip(group, temps, strict=True):
            cursor.emit(write(carry, t))

    scalars = [(o, c) for o, c in updates if not c.shape]
    if scalars:
        stage(scalars, "0")
    by_size: dict[int, list[tuple[CVal, CVal]]] = {}
    for o, c in updates:
        if c.shape:
            by_size.setdefault(c.size, []).append((o, c))
    for size, group in by_size.items():
        with cursor.loop(size) as i:
            stage(group, i)


@rule("while")
def _rule_while(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`lax.while_loop` -> `while (true) { cond; if (!p) break; body; }`.

    Carries follow the scan discipline (declared once, two-phase
    copy-back). The trip count is data-dependent, so threads diverge
    freely; that is fine in the one-thread-per-instance model.
    """
    cond_nconsts: int = eqn.params["cond_nconsts"]
    body_nconsts: int = eqn.params["body_nconsts"]
    cond_jaxpr = eqn.params["cond_jaxpr"]
    body_jaxpr = eqn.params["body_jaxpr"]
    if cond_jaxpr.consts or body_jaxpr.consts:
        raise EmitError("while with jaxpr consts is unsupported")

    cond_consts = [env.val(v) for v in eqn.invars[:cond_nconsts]]
    body_consts = [env.val(v) for v in eqn.invars[cond_nconsts : cond_nconsts + body_nconsts]]
    carries = []
    for invar, outvar in zip(eqn.invars[cond_nconsts + body_nconsts :], eqn.outvars, strict=True):
        dst = declare(env, cursor, outvar)
        cursor.copy(dst, env.val(invar), dst.size)
        carries.append(dst)

    with cursor.block("while (true)"):
        (pred,) = emit_jaxpr(env, cursor, cond_jaxpr.jaxpr, cond_consts + carries)
        with cursor.block(f"if (!{pred.at('0')})"):
            cursor.emit("break;")
        outs = emit_jaxpr(env, cursor, body_jaxpr.jaxpr, body_consts + carries)
        _copy_back_carries(cursor, outs, carries)


@rule("cond")
def _rule_cond(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`lax.cond` / `lax.switch` -> an if / else-if / else chain.

    `branches[i]` is selected by the integer index operand; following
    lax.switch semantics the index clamps to the valid range, so the
    first branch takes `<= 0` and the last takes the trailing `else`.
    Each branch emits into its own block and copies its results into
    storage declared ahead of the chain. Branches may diverge across
    threads; that is fine in the one-thread-per-instance model.
    """
    branches = eqn.params["branches"]
    index = env.val(eqn.invars[0])
    ops = [env.val(v) for v in eqn.invars[1:]]
    outs = [declare(env, cursor, ov) for ov in eqn.outvars]

    n = len(branches)
    for i, branch in enumerate(branches):
        if branch.consts:
            raise EmitError("cond with jaxpr consts is unsupported")
        if n == 1:
            header = "if (true)"
        elif i == 0:
            header = f"if ({index.at('0')} <= 0)"
        elif i < n - 1:
            header = f"else if ({index.at('0')} == {i})"
        else:
            header = "else"
        with cursor.block(header):
            vals = emit_jaxpr(env, cursor, branch.jaxpr, list(ops))
            for out, val in zip(outs, vals, strict=True):
                cursor.copy(out, val, out.size)
