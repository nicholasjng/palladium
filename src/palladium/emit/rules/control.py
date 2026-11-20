"""Control flow primitives: program_id, scan, while, and cond."""

from __future__ import annotations

import dataclasses
import math

from jax.extend.core import Jaxpr, JaxprEqn, Var

from palladium.emit.core import (
    REGISTER_BYTES,
    Cursor,
    CVal,
    Environment,
    declare,
    emit_jaxpr,
    msl_type,
    rule,
    shaped,
)
from palladium.errors import EmitError


@rule("program_id")
def _rule_program_id(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`pl.program_id(axis)` -> a component of _pid, as a rank-0 int.

    Pure aliasing, no storage or code. The (int) cast keeps index arithmetic signed.
    """
    axis: int = eqn.params["axis"]
    env.bind(eqn.outvars[0], CVal(f"(int){env.pids[axis]}", (), "int"))


@rule("num_programs")
def _rule_num_programs(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`pl.num_programs(axis)` -> the grid's extent along the axis, a constant."""
    env.bind(eqn.outvars[0], CVal(str(env.grid[eqn.params["axis"]]), (), "int"))


@rule("scan")
def _rule_scan(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`lax.fori_loop` / `lax.scan` -> a C for-loop.

    Consts pass through unchanged; carries get fresh mutable loop
    variables bound on the outvars; scanned xs bind per-iteration
    strided views; stacked ys stream straight to a device ref
    (`store_target`) or fill thread-local stacked storage, bounded
    by the per-thread stack.

    Carries update simultaneously (see `copy_back_carries`). ys slices
    are stored before copy-back, while the body outputs still hold this
    iteration's values.
    """
    length: int = eqn.params["length"]
    body: Jaxpr = eqn.params["jaxpr"]
    reverse: bool = eqn.params.get("reverse", False)
    consts, inits, xs = scan_operands(eqn)
    num_carry = len(inits)

    const_vals = [env.val(v) for v in consts]
    xs_vals = [env.val(v) for v in xs]
    if any(xs.transposed for xs in xs_vals):
        raise EmitError("scanning a lazily transposed xs is unsupported")

    carries = _bind_carries(env, cursor, inits, eqn.outvars[:num_carry])

    ys_targets = []
    for outvar in eqn.outvars[num_carry:]:
        target = store_target(env, outvar)
        if target is not None:
            env.bind(outvar, target)
        else:
            target = declare(env, cursor, outvar)
        ys_targets.append(target)

    # xs/ys keep their stacked positions under reverse, so the same
    # idx-based addressing serves both directions.
    with cursor.loop(length, "_s", reverse=reverse) as idx:
        # This iteration's xs are views into the stacked values, which are
        # immutable, so no copy is needed.
        x_slices = [
            xs.slot(idx, shaped(bv.aval).shape)
            if shaped(bv.aval).shape
            else CVal(xs.at(idx), (), xs.ctype, storage=xs.storage)
            for xs, bv in zip(xs_vals, body.invars[len(consts) + num_carry :], strict=True)
        ]
        outs = emit_jaxpr(env, cursor, body, const_vals + carries + x_slices)
        for target, y in zip(ys_targets, outs[num_carry:], strict=True):
            cursor.copy(target.slot(idx, (y.size,)), y, y.size)
        copy_back_carries(cursor, outs[:num_carry], carries)


def _bind_carries(env: Environment, cursor: Cursor, inits, outvars) -> list[CVal]:
    """Mutable loop variables bound on `outvars`, started from `inits`."""
    carries = []
    for invar, outvar in zip(inits, outvars, strict=True):
        init = env.val(invar)
        if (
            isinstance(invar, Var)
            and env.sole_consumer(invar) is not None
            and init.expr == init.storage  # the whole variable, not a view of it
            and init.space == "thread"
            and not init.transposed
            and env.storage_users[init.storage] == 1
        ):
            # A thread-local variable that nothing but this loop reads, under
            # any name: the carry updates it in place.
            dst = env.bind(outvar, init)
        else:
            dst = declare(env, cursor, outvar)
            cursor.copy(dst, init, dst.size)
        carries.append(dst)
    return carries


def scan_operands(eqn: JaxprEqn) -> tuple[list, list, list]:
    """A scan's operands as (consts, initial carries, xs).

    The split is the scan's own: `ft_in` is the tree jax packs the three
    groups into, and reads them back from.
    """
    consts, carries, xs = eqn.params["ft_in"].update(eqn.invars).unpack()
    return list(consts), list(carries), list(xs)


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
        consts, inits, _ = scan_operands(eqn)
        if any(iv is var for iv in (*consts, *inits)):
            return False
    return True


def _nbytes(var: Var) -> int:
    """Return the logical byte size of a shaped variable; scalars count as one element."""
    aval = shaped(var.aval)
    return math.prod(aval.shape) * aval.dtype.itemsize


def store_or_declare(env: Environment, cursor: Cursor, var: Var, *, min_bytes: int = 0) -> CVal:
    """Bind `var` to storage for a result that is written element by element.

    That is the output Ref the result is stored to, when `store_target` allows
    it and the result has at least `min_bytes`, which skips the later copy;
    otherwise fresh thread-local storage.
    """
    # Every rule that stores through this writes element by element, so a
    # block addressed through an index map is as good a target as a pointer.
    target = store_target(env, var, mapped=True) if _nbytes(var) >= min_bytes else None
    if target is None:
        return declare(env, cursor, var)
    shape = tuple(int(d) for d in shaped(var.aval).shape)
    return env.bind(var, dataclasses.replace(target, shape=shape or (1,)))


def store_target(env: Environment, outvar: Var, *, mapped: bool = False) -> CVal | None:
    """The output Ref a result may be computed into directly, or None.

    The result must feed one full-block store of a Ref that has no other use
    and shares no buffer with another Ref, so that the early write cannot be
    observed; layout and type must match, and a Ref reaching past its array is
    never a target. A Ref addressed through an index map has no pointer and
    qualifies only with `mapped`, for callers that write one element at a time.
    A result leaving an inlined jit body keeps local storage while it fits in
    registers, where writing a Ref inside a loop that reads device memory is
    slower.
    """
    outer = env.jit_outputs.get(outvar)
    if outer is not None and env.consumers.get(outvar) == [None]:
        # See the docstring: local storage wins while it fits in registers.
        if _nbytes(outvar) < REGISTER_BYTES:
            return None
        return store_target(env, outer, mapped=mapped)
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
        or ref.valid is not None
        or (ref.index_map is not None and not mapped)
    ):
        return None
    aval = shaped(outvar.aval)
    if ref.size != math.prod(aval.shape) or ref.ctype != msl_type(aval.dtype):
        return None
    return ref


def copy_back_carries(cursor: Cursor, outs: list[CVal], carries: list[CVal]) -> None:
    """Write loop-body outputs back into their carry storage.

    All carries update simultaneously, so every source element is read
    into a scalar before any carry element is written, in one fused loop
    per element count. An output that aliases a carry has that carry's
    aval, so aliasing never crosses sizes. Self-forwards are skipped.

    Carry permutations (an output that is another carry) read and write
    through volatile pointers to work around a Metal compiler bug: with
    three or more thread-local array temporaries live in the loop, the
    optimizer forwards a permuted carry's read across the write it must
    precede, producing wrong results on M-series GPUs. Volatile on the
    hazard endpoints is the narrowest workaround found; the
    permutation-free path stays non-volatile.
    """
    updates = [
        (out, carry) for out, carry in zip(outs, carries, strict=True) if out.expr != carry.expr
    ]
    carry_exprs = {c.expr for c in carries}

    def stage(group: list[tuple[CVal, CVal]], index: str) -> None:
        hazard = any(out.expr in carry_exprs for out, _ in group)

        def read(out: CVal, carry: CVal) -> str:
            # Only carry storage is read volatile; fresh values and
            # literals are outside the hazard, and a literal has no address.
            if not (hazard and out.expr in carry_exprs):
                return out.at(index)
            if carry.shape and not carry.scalar_storage:
                return f"((volatile thread {carry.ctype}*){out.expr})[{index}]"
            return f"(*(volatile thread {carry.ctype}*)&{out.expr})"

        def write(carry: CVal, t: str) -> str:
            if not hazard:
                return f"{carry.at(index)} = {t};"
            if carry.shape and not carry.scalar_storage:
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
    copy-back). Threads may diverge on the data-dependent trip count.
    """
    cond_nconsts: int = eqn.params["cond_nconsts"]
    body_nconsts: int = eqn.params["body_nconsts"]
    cond_jaxpr = eqn.params["cond_jaxpr"]
    body_jaxpr = eqn.params["body_jaxpr"]
    if cond_jaxpr.consts or body_jaxpr.consts:
        raise EmitError("while with jaxpr consts is unsupported")

    cond_consts = [env.val(v) for v in eqn.invars[:cond_nconsts]]
    body_consts = [env.val(v) for v in eqn.invars[cond_nconsts : cond_nconsts + body_nconsts]]
    carries = _bind_carries(env, cursor, eqn.invars[cond_nconsts + body_nconsts :], eqn.outvars)

    with cursor.block("while (true)"):
        (pred,) = emit_jaxpr(env, cursor, cond_jaxpr.jaxpr, cond_consts + carries)
        with cursor.block(f"if (!{pred.at('0')})"):
            cursor.emit("break;")
        outs = emit_jaxpr(env, cursor, body_jaxpr.jaxpr, body_consts + carries)
        copy_back_carries(cursor, outs, carries)


@rule("cond")
def _rule_cond(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`lax.cond` / `lax.switch` -> an if / else-if / else chain.

    `branches[i]` is selected by the integer index operand; following
    lax.switch semantics the index clamps to the valid range, so the
    first branch takes `<= 0` and the last takes the trailing `else`.
    Each branch emits into its own block and copies its results into
    storage declared ahead of the chain. Threads may diverge across
    branches.
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
