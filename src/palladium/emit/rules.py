"""Lowering rules for the one-thread-per-instance execution model.

One Metal thread runs one Pallas program instance; every jaxpr variable
becomes a thread-local C array and rules emit plain element loops.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

from jax.extend.core import Jaxpr, JaxprEqn

from palladium.emit.core import (
    ELEMENTWISE,
    PRIMITIVE_INVARS,
    RULES,
    CVal,
    EmitError,
    EmitState,
    _element_strides,
    _flat_index,
    _ref_view,
    _shaped,
    _template_fields,
    _transpose_is_dot_rhs_only,
    _unwrapped,
    emit_jaxpr,
    rule,
)


@rule("get")
def _rule_get(state: EmitState, eqn: JaxprEqn) -> None:
    """Load from a Ref: `y = x_ref[...]` (full block) or an indexed access
    like `y = x_ref[i, :]` (non-Slice dims squeeze into the offset, Slice
    dims are kept in the loaded shape).

    An indexed load from a read-only (input) ref binds the pointer view
    instead of copying: nothing can write through a `const device` ref,
    so the view has snapshot semantics. Full-block (un-indexed) loads
    still copy; those are small per-thread blocks that are re-read many
    times, where a thread-local copy pays off.
    """
    indexer_args = eqn.params["tree"].unflatten(eqn.invars[1:])
    src = state.val(eqn.invars[0])
    if indexer_args:
        (indexer,) = indexer_args
        view = _ref_view(state, src, indexer)
        aval = _shaped(eqn.outvars[0].aval)
        if (
            view.readonly
            and view.space == "device"
            and view.shape
            and math.prod(view.shape) == math.prod(tuple(int(d) for d in aval.shape))
        ):
            state.bind(
                eqn.outvars[0],
                dataclasses.replace(view, shape=tuple(int(d) for d in aval.shape)),
            )
            return
        src = view
    dst = state.declare(eqn.outvars[0])
    state.copy(dst, src, dst.size)


@rule("swap")
def _rule_swap(state: EmitState, eqn: JaxprEqn) -> None:
    """Store to a Ref: `o_ref[...] = y` (full block) or an indexed store
    like `o_ref[i, :] = y`.

    Deviation: the outvar binds to the (possibly indexed) ref view, not a
    pre-store snapshot, so reads of a swap result alias device memory.
    """
    ref, value = eqn.invars[0], eqn.invars[1]
    indexer_args = eqn.params["tree"].unflatten(eqn.invars[2:])
    dst_ref = state.val(ref)
    if indexer_args:
        (indexer,) = indexer_args
        dst_ref = _ref_view(state, dst_ref, indexer)
    stored = state.val(value)
    state.copy(dst_ref, stored, dst_ref.size)
    state.bind(eqn.outvars[0], dst_ref)


@rule("jit")
def _inline_jit(state: EmitState, eqn: JaxprEqn) -> None:
    """Inline a jit-wrapped call by walking its body jaxpr."""
    inner: Jaxpr = eqn.params["jaxpr"]
    body = inner.jaxpr
    if inner.consts:
        raise EmitError("jit with consts is unsupported")

    invals = [state.val(invar) for invar in eqn.invars]
    outvals = emit_jaxpr(state, body, invals)

    for outvar, val in zip(eqn.outvars, outvals, strict=True):
        state.bind(outvar, val)


@rule("random_wrap")
def _rule_random_wrap(state: EmitState, eqn: JaxprEqn) -> None:
    """`jax.random.wrap_key_data`: pure type-level wrap (uint32[2] array
    to a `key<fry>[]`-typed value), no MSL. Alias, like `program_id`."""
    state.bind(eqn.outvars[0], state.val(eqn.invars[0]))


@rule("random_unwrap")
def _rule_random_unwrap(state: EmitState, eqn: JaxprEqn) -> None:
    """`jax.random.key_data`: the inverse of `random_wrap`, equally pure
    aliasing (a `key<fry>[]`-typed value is a uint32[2] array underneath
    the whole time; nothing to convert)."""
    state.bind(eqn.outvars[0], state.val(eqn.invars[0]))


# Threefry-2x32-20, the algorithm `jax._src.random.threefry2x32` runs by default.
# Verified against `jax.random.bits(key, shape, 'uint32')` bit-for-bit:
# rotations, key schedule, counter construction (flat row-major index,
# hi word 0 for any shape under 2**32 elements).
_THREEFRY_ROT0 = (13, 15, 26, 6)
_THREEFRY_ROT1 = (17, 29, 16, 24)


def _emit_threefry2x32(
    state: EmitState, k1: str, k2: str, x1: str, x2: str
) -> tuple[str, str]:
    """Emit uint x0, x1 = threefry2x32(k1, k2, x1, x2); return their names."""
    ks2 = state.fresh("_ks2")
    state.emit(f"uint {ks2} = {k1} ^ {k2} ^ 0x1BD11BDAu;")
    ks = (k1, k2, ks2)

    x0 = state.fresh("_tx0")
    x1n = state.fresh("_tx1")
    state.emit(f"uint {x0} = {x1} + {ks[0]};")
    state.emit(f"uint {x1n} = {x2} + {ks[1]};")

    def apply_round(rot: int) -> None:
        state.emit(f"{x0} = {x0} + {x1n};")
        state.emit(f"{x1n} = ({x1n} << {rot}u) | ({x1n} >> {32 - rot}u);")
        state.emit(f"{x1n} = {x0} ^ {x1n};")

    # 5 groups of 4 rounds, alternating rotation sets; a, b index into ks
    # for the post-group key-schedule addition, n is that round's counter.
    schedule = (
        (_THREEFRY_ROT0, 1, 2, 1),
        (_THREEFRY_ROT1, 2, 0, 2),
        (_THREEFRY_ROT0, 0, 1, 3),
        (_THREEFRY_ROT1, 1, 2, 4),
        (_THREEFRY_ROT0, 2, 0, 5),
    )
    for rots, a, b, n in schedule:
        for r in rots:
            apply_round(r)
        state.emit(f"{x0} = {x0} + {ks[a]};")
        state.emit(f"{x1n} = {x1n} + {ks[b]} + {n}u;")

    return x0, x1n


@rule("random_bits")
def _rule_random_bits(state: EmitState, eqn: JaxprEqn) -> None:
    """`jax.random.bits`: Threefry-2x32-20 counter-based bits, one hash
    per output element. Counter is the element's flat row-major index
    (hi word 0, lo word the index); matches jax's own construction for
    any shape under 2**32 elements, which is every real kernel output.
    """
    if eqn.params["bit_width"] != 32:
        raise NotImplementedError("random_bits: only bit_width=32 is implemented")

    key = state.val(eqn.invars[0])
    dst = state.declare(eqn.outvars[0])
    k1, k2 = key.at("0"), key.at("1")

    idx = state.fresh("_i")
    with state.block(f"for (uint {idx} = 0; {idx} < {dst.size}; ++{idx})"):
        b0, b1 = _emit_threefry2x32(state, k1, k2, "0u", idx)
        state.emit(f"{dst.at(idx)} = {b0} ^ {b1};")


@rule("random_fold_in")
def _rule_random_fold_in(state: EmitState, eqn: JaxprEqn) -> None:
    """`jax.random.fold_in`: a fresh key from `(key, data)`, the same
    Threefry-2x32-20 hash seeded with (0, data) in place of a counter.
    Matches `jax.random.fold_in`'s `key_data` output.
    """
    key = state.val(eqn.invars[0])
    data = state.val(eqn.invars[1])
    k1, k2 = key.at("0"), key.at("1")
    b0, b1 = _emit_threefry2x32(state, k1, k2, "0u", data.expr)

    name = state.fresh()
    state.emit(f"uint {name}[2] = {{{b0}, {b1}}};")
    state.bind(eqn.outvars[0], CVal(expr=name, shape=(2,), ctype="uint"))


@rule("reshape")
def _rule_reshape(state: EmitState, eqn: JaxprEqn) -> None:
    """Reshape: row-major reinterpretation of the same flat storage, no
    data movement, since `CVal.at()` already indexes flatly regardless of
    rank.

    Raises
    ------
    NotImplementedError
        If `dimensions` (an axis permutation applied before reshaping)
        is set; that needs a real copy, not just a shape reinterpretation.
    """
    if eqn.params["dimensions"] is not None:
        raise NotImplementedError(
            "reshape with a dimensions permutation is unimplemented"
        )
    src = state.val(eqn.invars[0])
    state.bind(eqn.outvars[0], dataclasses.replace(src, shape=eqn.params["new_sizes"]))


@rule("transpose")
def _rule_transpose(state: EmitState, eqn: JaxprEqn) -> None:
    """`x.T` / `jnp.transpose(x, perm)` -> a materialized, permuted copy.

    A real copy, not a view like reshape: transposed storage is
    different bytes in the row-major flat-array model. `a.T` inside a
    dot product stages as a standalone `transpose` equation ahead of
    `dot_general`; the contraction itself always stays (lhs dim 1,
    rhs dim 0).

    Exception: a rank-2 transpose consumed only as `dot_general` rhs
    never materializes. It binds a `transposed` CVal (untransposed
    storage plus a flag) and the dot reads element `(k, j)` at
    `[j * K + k]`, unit-stride in the contraction index for both
    operands. Gated on `EmitState.consumers` so no generic flat-indexing
    rule can observe the transposed CVal.
    """
    src = state.val(eqn.invars[0])
    outvar = eqn.outvars[0]
    if _transpose_is_dot_rhs_only(state.consumers, eqn) and not src.transposed:
        state.bind(
            outvar,
            dataclasses.replace(
                src, shape=(src.shape[1], src.shape[0]), transposed=True
            ),
        )
        return
    perm: tuple[int, ...] = eqn.params["permutation"]
    dst = state.declare(eqn.outvars[0])
    src_strides = _element_strides(src.shape)
    dst_strides = _element_strides(dst.shape)
    rank = len(src.shape)
    idx_vars = [state.fresh(f"_t{d}") for d in range(rank)]

    def emit_loops(d: int) -> None:
        if d == rank:
            src_idx = _flat_index(
                [(idx_vars[dd], src_strides[perm[dd]]) for dd in range(rank)]
            )
            dst_idx = _flat_index(
                [(idx_vars[dd], dst_strides[dd]) for dd in range(rank)]
            )
            state.emit(f"{dst.at(dst_idx)} = {src.at(src_idx)};")
            return
        with state.block(
            f"for (uint {idx_vars[d]} = 0; {idx_vars[d]} < {dst.shape[d]}; ++{idx_vars[d]})"
        ):
            emit_loops(d + 1)

    emit_loops(0)


def _rule_elementwise(state: EmitState, eqn: JaxprEqn) -> None:
    """One rule for every pure elementwise primitive.

    Three templates are derived rather than looked up: integer_pow expands
    to repeated multiplication (reciprocal for negative y); convert_element_type
    casts to the output's ctype; bitcast_convert_type uses as_type<T>.

    Operands broadcast against `dst`'s shape numpy-style: right-aligned,
    size-1 dims replicate. The arity check requires every template field
    to be filled and every operand consumed.
    """

    dst = state.declare(eqn.outvars[0])
    ops = [state.val(v) for v in eqn.invars]
    opname = eqn.primitive.name

    if opname == "not":
        template = "(!{a})" if ops[0].ctype == "bool" else "(~{a})"
    elif opname == "integer_pow":
        exp: int = eqn.params["y"]
        body = " * ".join(["{a}"] * abs(exp))
        template = f"({body})" if exp > 0 else f"(1.0f / ({body}))"
    elif opname == "convert_element_type":
        template = f"(({dst.ctype}){{a}})"
    elif opname == "bitcast_convert_type":
        # Reinterprets bits without a numeric conversion (unlike
        # convert_element_type's cast); Metal's equivalent is as_type<T>.
        template = f"as_type<{dst.ctype}>({{a}})"
    elif opname == "select_n":
        if (pred_type := ops[0].ctype) != "bool":
            raise EmitError(
                f"select_n requires predicate of type bool, got {pred_type}"
            )
        template = ELEMENTWISE[opname]
    else:
        template = ELEMENTWISE[opname]

    fields = _template_fields(template)
    expected = set(PRIMITIVE_INVARS[: len(ops)])
    if fields != expected:
        raise EmitError(f"{opname} requires {len(fields)} operands, got {len(ops)}")

    rank = len(dst.shape)
    dst_strides = _element_strides(dst.shape)

    def op_index(op: CVal, idx_vars: list[str]) -> str:
        # numpy right-alignment: an operand's dim d lines up with dst's
        # dim d + (rank - len(op.shape)); size-1 dims always read index 0.
        rank_diff = rank - len(op.shape)
        op_strides = _element_strides(op.shape)
        return _flat_index(
            [
                (idx_vars[d + rank_diff], op_strides[d])
                for d, size in enumerate(op.shape)
                if size != 1
            ]
        )

    def assign(idx_vars: list[str]) -> str:
        dst_idx = _flat_index(list(zip(idx_vars, dst_strides)))
        inputs = {
            name: op.at(op_index(op, idx_vars))
            for name, op in zip(PRIMITIVE_INVARS, ops)
        }
        return f"{dst.at(dst_idx)} = {_unwrapped(template.format(**inputs))};"

    idx_vars = [state.fresh("_i") for _ in range(rank)]

    def emit_loops(d: int) -> None:
        if d == rank:
            state.emit(assign(idx_vars))
            return
        with state.block(
            f"for (uint {idx_vars[d]} = 0; {idx_vars[d]} < {dst.shape[d]}; ++{idx_vars[d]})"
        ):
            emit_loops(d + 1)

    emit_loops(0)


for _name in [
    *ELEMENTWISE,
    "integer_pow",
    "convert_element_type",
    "bitcast_convert_type",
    "not",
]:
    RULES[_name] = _rule_elementwise


@rule("broadcast_in_dim")
def _rule_broadcast_in_dim(state: EmitState, eqn: JaxprEqn) -> None:
    """`jnp.broadcast_to`/rank-matching before an elementwise op.

    Size-preserving broadcasts alias; the rest materialize a copy,
    replicating size-1 (or absent) input dims.
    """
    src = state.val(eqn.invars[0])
    aval = _shaped(eqn.outvars[0].aval)
    out_shape = tuple(int(d) for d in aval.shape)
    if src.shape and src.size == math.prod(out_shape):
        # Pure rank change (broadcast_dimensions are strictly increasing,
        # so row-major flat order is preserved): alias, no copy. Scalar
        # expressions (literals, rank-0 values) keep the copy path; they
        # have no storage to alias.
        state.bind(eqn.outvars[0], dataclasses.replace(src, shape=out_shape))
        return
    dst = state.declare(eqn.outvars[0])
    bcast_dims: tuple[int, ...] = eqn.params["broadcast_dimensions"]
    src_strides = _element_strides(src.shape)
    dst_strides = _element_strides(dst.shape)
    rank = len(dst.shape)
    idx_vars = [state.fresh(f"_b{d}") for d in range(rank)]

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
            state.emit(f"{dst.at(dst_idx)} = {src.at(src_idx)};")
            return
        with state.block(
            f"for (uint {idx_vars[d]} = 0; {idx_vars[d]} < {dst.shape[d]}; ++{idx_vars[d]})"
        ):
            emit_loops(d + 1)

    emit_loops(0)


@rule("dot_general")
def _rule_dot_general(state: EmitState, eqn: JaxprEqn) -> None:
    """`a @ b` for rank-2, non-batched operands.

    Scalar triple-nested loop for the plain matmul contraction (lhs dim
    1 with rhs dim 0); the vectorized paths below take over when operand
    layout and alignment allow. Batch dims, higher rank, and other
    contraction axes are unimplemented.

    The scalar path keeps `i, j` outer with `k` innermost on purpose: an
    `i, k, j` reorder turns the single per-`(i, j)` write into `k`
    read-modify-writes of `dst` per `j` and is slower in practice. Keep
    the reduction in a scalar register.
    """
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params[
        "dimension_numbers"
    ]
    if lhs_batch or rhs_batch:
        raise EmitError("dot_general: batch dims are unimplemented")
    if tuple(lhs_contract) != (1,) or tuple(rhs_contract) != (0,):
        raise EmitError(
            "dot_general: only the standard matmul contraction (lhs dim 1 "
            "with rhs dim 0) is implemented"
        )

    lhs = state.val(eqn.invars[0])
    rhs = state.val(eqn.invars[1])
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise EmitError("dot_general: only rank-2 operands are implemented")
    m, k = lhs.shape
    k2, n = rhs.shape
    if k != k2:
        raise EmitError(f"dot_general: inner dims disagree ({k} vs {k2})")

    dst = state.declare(eqn.outvars[0])

    if (
        dst.ctype == "float"
        and rhs.transposed
        and k % 4 == 0
        and lhs.align % 4 == 0
        and rhs.align % 4 == 0
    ):
        _emit_dot_general_rowdot_vectorized(state, lhs, rhs, dst, m, k, n)
        return

    if (
        m == 1
        and n % 4 == 0
        and n <= _M1_VECTORIZE_MAX_N
        and dst.ctype == "float"
        and not rhs.transposed
        and rhs.align % 4 == 0
    ):
        _emit_dot_general_m1_vectorized(state, lhs, rhs, dst, k, n)
        return

    i, j, kk, acc = (
        state.fresh("_mi"),
        state.fresh("_ni"),
        state.fresh("_ki"),
        state.fresh("_acc"),
    )
    with (
        state.block(f"for (uint {i} = 0; {i} < {m}; ++{i})"),
        state.block(f"for (uint {j} = 0; {j} < {n}; ++{j})"),
    ):
        state.emit(f"{dst.ctype} {acc} = 0;")
        with state.block(f"for (uint {kk} = 0; {kk} < {k}; ++{kk})"):
            lhs_idx = f"{i} * {k} + {kk}"
            rhs_idx = f"{j} * {k} + {kk}" if rhs.transposed else f"{kk} * {n} + {j}"
            state.emit(f"{acc} += {lhs.at(lhs_idx)} * {rhs.at(rhs_idx)};")
        state.emit(f"{dst.at(f'{i} * {n} + {j}')} = {acc};")


# Cutoff for the m == 1 float4 path. Above it the extra float4
# accumulator registers compete with the kernel's live carries for the
# register budget and the vectorization loses; the value is the largest
# width confirmed to win in a fused kernel.
_M1_VECTORIZE_MAX_N = 32


def _emit_dot_general_m1_vectorized(
    state: EmitState, lhs: CVal, rhs: CVal, dst: CVal, k: int, n: int
) -> None:
    """`(1, k) @ (k, n) -> (1, n)`, `n % 4 == 0`, `n <= _M1_VECTORIZE_MAX_N`.

    `k` outer / `j` inner, vectorized over `j` in float4 lanes, with the
    `n/4`-wide accumulator held in registers for the whole `k` sweep.
    Valid only while the accumulator stays register-resident, hence the
    cutoff.

    `dst` is always an emitter-declared thread-local array; `rhs` may be
    thread-local or a device view, so its float4 cast is qualified by
    `rhs.space` and gated on 4-element alignment at the call site. A
    mis-qualified or misaligned pointer cast is invalid MSL.
    """
    n4 = n // 4
    rspace = rhs.space
    kk, j4, acc, ak, brow = (
        state.fresh("_ki"),
        state.fresh("_j4"),
        state.fresh("_acc4"),
        state.fresh("_ak"),
        state.fresh("_brow"),
    )
    state.emit(f"thread float4 {acc}[{n4}];")
    with state.block(f"for (uint {j4} = 0; {j4} < {n4}u; ++{j4})"):
        state.emit(f"{acc}[{j4}] = float4(0.0f);")
    with state.block(f"for (uint {kk} = 0; {kk} < {k}u; ++{kk})"):
        state.emit(f"float {ak} = {lhs.at(kk)};")
        state.emit(
            f"{rspace} const float4* {brow} = "
            f"({rspace} const float4*)({rhs.expr}) + {kk} * {n4}u;"
        )
        with state.block(f"for (uint {j4} = 0; {j4} < {n4}u; ++{j4})"):
            state.emit(f"{acc}[{j4}] += {ak} * {brow}[{j4}];")
    with state.block(f"for (uint {j4} = 0; {j4} < {n4}u; ++{j4})"):
        state.emit(f"((thread float4*)&{dst.expr}[0])[{j4}] = {acc}[{j4}];")


def _emit_dot_general_rowdot_vectorized(
    state: EmitState, lhs: CVal, rhs: CVal, dst: CVal, m: int, k: int, n: int
) -> None:
    """`(m, k) @ (k, n)` with a lazy-transposed rhs.

    Element `(kk, j)` of the rhs lives at `[j * k + kk]` of the
    untransposed storage, so lhs row `i` and rhs column `j` are both
    unit-stride in the contraction index: each output element is a dot
    of two contiguous rows, emitted as float4 multiply-accumulates over
    `k/4` lanes with one horizontal sum. `k % 4 == 0` checked at the
    call site.

    Either operand may live in `thread` or `device` space; casts are
    qualified accordingly and both operands' alignment is checked at
    the call site.
    """
    k4 = k // 4
    i, j, kk, acc, arow, brow = (
        state.fresh("_mi"),
        state.fresh("_ni"),
        state.fresh("_k4"),
        state.fresh("_acc4"),
        state.fresh("_arow"),
        state.fresh("_brow"),
    )
    with state.block(f"for (uint {i} = 0; {i} < {m}; ++{i})"):
        state.emit(
            f"{lhs.space} const float4* {arow} = "
            f"({lhs.space} const float4*)({lhs.expr}) + {i} * {k4}u;"
        )
        with state.block(f"for (uint {j} = 0; {j} < {n}; ++{j})"):
            state.emit(
                f"{rhs.space} const float4* {brow} = "
                f"({rhs.space} const float4*)({rhs.expr}) + {j} * {k4}u;"
            )
            state.emit(f"float4 {acc} = float4(0.0f);")
            with state.block(f"for (uint {kk} = 0; {kk} < {k4}u; ++{kk})"):
                state.emit(f"{acc} += {arow}[{kk}] * {brow}[{kk}];")
            state.emit(
                f"{dst.at(f'{i} * {n} + {j}')} = {acc}.x + {acc}.y + {acc}.z + {acc}.w;"
            )


def _emit_reduce(
    state: EmitState, eqn: JaxprEqn, init: str, combine: Callable[[str, str], str]
) -> None:
    """Shared shape for `reduce_sum`/`reduce_max`: nested loops, one per
    kept dim, each accumulating over the reduced dims via `combine`
    starting from `init`. `axes` may be any subset of dims, not just
    "reduce to a scalar"."""
    axes = set(eqn.params["axes"])
    src = state.val(eqn.invars[0])
    rank = len(src.shape)
    kept_dims = [d for d in range(rank) if d not in axes]
    reduced_dims = [d for d in range(rank) if d in axes]

    dst = state.declare(eqn.outvars[0])
    src_strides = _element_strides(src.shape)
    dst_strides = _element_strides(dst.shape)
    idx_vars: dict[int, str] = {}

    def emit_loops(dims: list[int], body: Callable[[], None]) -> None:
        if not dims:
            body()
            return
        d, rest = dims[0], dims[1:]
        idx_vars[d] = state.fresh(f"_d{d}")
        with state.block(
            f"for (uint {idx_vars[d]} = 0; {idx_vars[d]} < {src.shape[d]}; ++{idx_vars[d]})"
        ):
            emit_loops(rest, body)

    def accumulate() -> None:
        acc = state.fresh("_acc")
        state.emit(f"{dst.ctype} {acc} = {init};")

        def inner_body() -> None:
            src_idx = _flat_index([(idx_vars[d], src_strides[d]) for d in range(rank)])
            state.emit(f"{acc} = {combine(acc, src.at(src_idx))};")

        emit_loops(reduced_dims, inner_body)
        dst_idx = _flat_index(
            [(idx_vars[d], dst_strides[i]) for i, d in enumerate(kept_dims)]
        )
        state.emit(f"{dst.at(dst_idx)} = {acc};")

    emit_loops(kept_dims, accumulate)


@rule("reduce_sum")
def _rule_reduce_sum(state: EmitState, eqn: JaxprEqn) -> None:
    """`jnp.sum(x, axis=...)` -> nested loops, one per kept dim, each
    accumulating over the reduced dims.

    Verified against a real jaxpr before implementing: `jnp.sum` and
    `jnp.mean` both stage as `reduce_sum[axes=...]` (mean adds a plain
    `div` after, already emittable via ELEMENTWISE).
    """
    _emit_reduce(state, eqn, "0", lambda acc, x: f"{acc} + {x}")


@rule("reduce_max")
def _rule_reduce_max(state: EmitState, eqn: JaxprEqn) -> None:
    """`jnp.max(x, axis=...)` -> the same nested-loop shape as
    `reduce_sum`, `fmax`-combining from `-INFINITY` instead of summing
    from 0. Verified against a real jaxpr before implementing:
    `jnp.max` stages as `reduce_max[axes=...]`, exactly parallel to
    `reduce_sum`. Needed for softmax numerical stability (subtract the
    row max before `exp`), the actual motivating case.
    """
    _emit_reduce(state, eqn, "-INFINITY", lambda acc, x: f"fmax({acc}, {x})")


@rule("program_id")
def _rule_program_id(state: EmitState, eqn: JaxprEqn) -> None:
    """`pl.program_id(axis)` -> a component of _pid, as a rank-0 int.

    Pure aliasing, no storage or code. The (int) cast keeps index
    arithmetic signed.
    """
    axis: int = eqn.params["axis"]
    state.bind(eqn.outvars[0], CVal(f"(int){state.pid_exprs[axis]}", (), "int"))


@rule("scan")
def _rule_scan(state: EmitState, eqn: JaxprEqn) -> None:
    """`lax.fori_loop` / pure-carry `lax.scan` -> a C for-loop.

    fori_loop stages as scan with no xs/ys, usually with consts ordered
    before carries in both eqn.invars and body.invars. Consts pass through
    unchanged; carries get fresh mutable loop variables bound on the
    outvars.

    Copy-back runs in two phases: scan updates all carries simultaneously,
    so phase 1 snapshots reads that alias other carries into temps
    (skipping self-forward no-ops), then phase 2 overwrites the carries.
    """
    num_carry = len(eqn.outvars)
    num_consts = len(eqn.invars) - num_carry
    length: int = eqn.params["length"]
    body: Jaxpr = eqn.params["jaxpr"]

    if len(body.outvars) != num_carry:
        raise EmitError("lax.scan with stacked ys not supported, use lax.fori_loop")

    reverse: bool = eqn.params.get("reverse", False)
    if reverse:
        raise EmitError("no reverse-mode scan support, use lax.fori_loop")

    const_vals = [state.val(v) for v in eqn.invars[:num_consts]]

    carries = []
    for invar, outvar in zip(eqn.invars[num_consts:], eqn.outvars, strict=True):
        dst = state.declare(outvar)
        state.copy(dst, state.val(invar), dst.size)
        carries.append(dst)

    idx = state.fresh("_s")
    with state.block(f"for (uint {idx} = 0; {idx} < {length}; ++{idx})"):
        outs = emit_jaxpr(state, body, const_vals + carries)
        _copy_back_carries(state, outs, carries)


def _copy_back_carries(state: EmitState, outs: list[CVal], carries: list[CVal]) -> None:
    """Write loop-body outputs back into their carry storage.

    Two phases, because all carries update simultaneously: phase 1
    snapshots outputs that alias a *different* carry into temps (skipping
    self-forward no-ops), phase 2 overwrites the carries. Shared by the
    scan and while lowerings.
    """
    carry_exprs = {c.expr for c in carries}
    sources = []
    for out, carry in zip(outs, carries, strict=True):
        if out.expr == carry.expr:
            # Self-forward: the value is already in place.
            sources.append(None)
        elif out.expr in carry_exprs:
            # Aliases a different carry: snapshot before any overwrite.
            tmp = CVal(state.fresh(), out.shape, out.ctype)
            state.emit(
                f"{tmp.ctype} {tmp.expr}[{tmp.size}];"
                if tmp.shape
                else f"{tmp.ctype} {tmp.expr};"
            )
            state.copy(tmp, out, out.size)
            sources.append(tmp)
        else:
            # Fresh SSA name: no carry write can clobber it.
            sources.append(out)

    for src, carry in zip(sources, carries, strict=True):
        if src is not None:
            state.copy(carry, src, carry.size)


@rule("while")
def _rule_while(state: EmitState, eqn: JaxprEqn) -> None:
    """`lax.while_loop` -> `while (true) { cond; if (!p) break; body; }`.

    Carries follow the scan discipline (declared once, two-phase
    copy-back). The trip count is data-dependent, so threads diverge
    freely; that is fine in the one-thread-per-instance model and is why
    the cooperative model does not register this rule.
    """
    cond_nconsts: int = eqn.params["cond_nconsts"]
    body_nconsts: int = eqn.params["body_nconsts"]
    cond_jaxpr = eqn.params["cond_jaxpr"]
    body_jaxpr = eqn.params["body_jaxpr"]
    if cond_jaxpr.consts or body_jaxpr.consts:
        raise EmitError("while with jaxpr consts is unsupported")

    cond_consts = [state.val(v) for v in eqn.invars[:cond_nconsts]]
    body_consts = [
        state.val(v) for v in eqn.invars[cond_nconsts : cond_nconsts + body_nconsts]
    ]
    carries = []
    for invar, outvar in zip(
        eqn.invars[cond_nconsts + body_nconsts :], eqn.outvars, strict=True
    ):
        dst = state.declare(outvar)
        state.copy(dst, state.val(invar), dst.size)
        carries.append(dst)

    with state.block("while (true)"):
        (pred,) = emit_jaxpr(state, cond_jaxpr.jaxpr, cond_consts + carries)
        with state.block(f"if (!{pred.at('0')})"):
            state.emit("break;")
        outs = emit_jaxpr(state, body_jaxpr.jaxpr, body_consts + carries)
        _copy_back_carries(state, outs, carries)


@rule("cond")
def _rule_cond(state: EmitState, eqn: JaxprEqn) -> None:
    """`lax.cond` / `lax.switch` -> an if / else-if / else chain.

    `branches[i]` is selected by the integer index operand; following
    lax.switch semantics the index clamps to the valid range, so the
    first branch takes `<= 0` and the last takes the trailing `else`.
    Each branch emits into its own block and copies its results into
    storage declared ahead of the chain. Branches may diverge across
    threads; that is fine in the one-thread-per-instance model and is
    why the cooperative model does not register this rule.
    """
    branches = eqn.params["branches"]
    index = state.val(eqn.invars[0])
    ops = [state.val(v) for v in eqn.invars[1:]]
    outs = [state.declare(ov) for ov in eqn.outvars]

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
        with state.block(header):
            vals = emit_jaxpr(state, branch.jaxpr, list(ops))
            for out, val in zip(outs, vals, strict=True):
                state.copy(out, val, out.size)
