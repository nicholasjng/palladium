"""Lowerings for one part of the MSL execution model."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from jax.extend.core import JaxprEqn

from palladium.emit.core import (
    Cursor,
    CVal,
    EmitError,
    Environment,
    _element_strides,
    _flat_index,
    declare,
    rule,
)
from palladium.emit.numeric import extremum, extremum_identity


@rule("dot_general")
def _rule_dot_general(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`a @ b` for non-batched operands up to rank 2.

    Scalar triple-nested loop for the plain matmul contraction (lhs dim
    1 with rhs dim 0); the vectorized paths below take over when operand
    layout, dtype, and alignment allow. Rank-1 operands canonicalize to
    `(1, k)` lhs / `(k, 1)` rhs over the same flat storage, covering
    matvec, vecmat, and vecvec. Batch dims, higher rank, and other
    contraction axes are unimplemented.

    `preferred_element_type` is honored through the output aval (JAX
    computes the output dtype from it): accumulation runs in the output
    ctype, and mixed-precision products are cast to it before the
    multiply. The vectorized paths require f32 end to end.

    The scalar path keeps `i, j` outer with `k` innermost: an `i, k, j`
    reorder turns the single per-`(i, j)` write into `k`
    read-modify-writes of `dst` per `j`, and measures slower.
    """
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params["dimension_numbers"]
    if lhs_batch or rhs_batch:
        raise EmitError("dot_general: batch dims are unimplemented")

    lhs = env.val(eqn.invars[0])
    rhs = env.val(eqn.invars[1])
    if len(lhs.shape) == 1 and tuple(lhs_contract) == (0,):
        lhs = dataclasses.replace(lhs, shape=(1, lhs.shape[0]))
        lhs_contract = (1,)
    if len(rhs.shape) == 1 and tuple(rhs_contract) == (0,):
        rhs = dataclasses.replace(rhs, shape=(rhs.shape[0], 1))
    if tuple(lhs_contract) != (1,) or tuple(rhs_contract) != (0,):
        raise EmitError(
            "dot_general: only the standard matmul contraction (lhs dim 1 "
            "with rhs dim 0) is implemented"
        )
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise EmitError("dot_general: only rank-1/rank-2 operands are implemented")
    m, k = lhs.shape
    k2, n = rhs.shape
    if k != k2:
        raise EmitError(f"dot_general: inner dims disagree ({k} vs {k2})")

    dst = declare(env, cursor, eqn.outvars[0])
    all_f32 = dst.ctype == lhs.ctype == rhs.ctype == "float"

    if all_f32 and rhs.transposed and k % 4 == 0 and lhs.align % 4 == 0 and rhs.align % 4 == 0:
        _emit_dot_general_rowdot_vectorized(cursor, lhs, rhs, dst, m, k, n)
        return

    if (
        m == 1
        and n % 4 == 0
        and n <= _M1_VECTORIZE_MAX_N
        and all_f32
        and not rhs.transposed
        and rhs.align % 4 == 0
    ):
        _emit_dot_general_m1_vectorized(cursor, lhs, rhs, dst, k, n)
        return

    acc = cursor.fresh("_acc")
    with cursor.loop(m, "_mi") as i, cursor.loop(n, "_ni") as j:
        cursor.emit(f"{dst.ctype} {acc} = 0;")
        with cursor.loop(k, "_ki") as kk:
            lhs_idx = f"{i} * {k} + {kk}"
            rhs_idx = f"{j} * {k} + {kk}" if rhs.transposed else f"{kk} * {n} + {j}"
            a_elem = lhs.at(lhs_idx)
            if lhs.ctype != dst.ctype:
                # Mixed precision (preferred_element_type wider than the
                # operands): promote before the multiply so the product
                # accumulates in the output ctype.
                a_elem = f"(({dst.ctype}){a_elem})"
            cursor.emit(f"{acc} += {a_elem} * {rhs.at(rhs_idx)};")
        cursor.emit(f"{dst.at(f'{i} * {n} + {j}')} = {acc};")


# Cutoff for the m == 1 float4 path. Above it the extra float4
# accumulator registers compete with the kernel's live carries for the
# register budget and the vectorization loses; the value is the largest
# width confirmed to win in a fused kernel.
_M1_VECTORIZE_MAX_N = 32


def _emit_dot_general_m1_vectorized(
    cursor: Cursor, lhs: CVal, rhs: CVal, dst: CVal, k: int, n: int
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

    Pure text emission, no bindings: takes a Cursor, not an Environment.
    """
    n4 = n // 4
    rspace = rhs.space
    acc, ak, brow = (
        cursor.fresh("_acc4"),
        cursor.fresh("_ak"),
        cursor.fresh("_brow"),
    )
    cursor.emit(f"thread float4 {acc}[{n4}];")
    with cursor.loop(f"{n4}u", "_j4") as j4:
        cursor.emit(f"{acc}[{j4}] = float4(0.0f);")
    with cursor.loop(f"{k}u", "_ki") as kk:
        cursor.emit(f"float {ak} = {lhs.at(kk)};")
        cursor.emit(
            f"{rspace} const float4* {brow} = ({rspace} const float4*)({rhs.expr}) + {kk} * {n4}u;"
        )
        with cursor.loop(f"{n4}u", "_j4") as j4:
            cursor.emit(f"{acc}[{j4}] += {ak} * {brow}[{j4}];")
    with cursor.loop(f"{n4}u", "_j4") as j4:
        cursor.emit(f"((thread float4*)&{dst.expr}[0])[{j4}] = {acc}[{j4}];")


def _emit_dot_general_rowdot_vectorized(
    cursor: Cursor, lhs: CVal, rhs: CVal, dst: CVal, m: int, k: int, n: int
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

    Pure text emission, no bindings: takes a Cursor, not an Environment.
    """
    k4 = k // 4
    acc, arow, brow = (
        cursor.fresh("_acc4"),
        cursor.fresh("_arow"),
        cursor.fresh("_brow"),
    )
    with cursor.loop(m, "_mi") as i:
        cursor.emit(
            f"{lhs.space} const float4* {arow} = "
            f"({lhs.space} const float4*)({lhs.expr}) + {i} * {k4}u;"
        )
        with cursor.loop(n, "_ni") as j:
            cursor.emit(
                f"{rhs.space} const float4* {brow} = "
                f"({rhs.space} const float4*)({rhs.expr}) + {j} * {k4}u;"
            )
            cursor.emit(f"float4 {acc} = float4(0.0f);")
            with cursor.loop(f"{k4}u", "_k4") as kk:
                cursor.emit(f"{acc} += {arow}[{kk}] * {brow}[{kk}];")
            cursor.emit(f"{dst.at(f'{i} * {n} + {j}')} = {acc}.x + {acc}.y + {acc}.z + {acc}.w;")


def _emit_reduce(
    env: Environment,
    cursor: Cursor,
    eqn: JaxprEqn,
    init: str,
    combine: Callable[[str, str], str],
) -> None:
    """Shared shape for `reduce_sum`/`reduce_max`: nested loops, one per
    kept dim, each accumulating over the reduced dims via `combine`
    starting from `init`. `axes` may be any subset of dims, not just
    "reduce to a scalar"."""
    axes = set(eqn.params["axes"])
    src = env.val(eqn.invars[0])
    rank = len(src.shape)
    kept_dims = [d for d in range(rank) if d not in axes]
    reduced_dims = [d for d in range(rank) if d in axes]

    dst = declare(env, cursor, eqn.outvars[0])
    src_strides = _element_strides(src.shape)
    dst_strides = _element_strides(dst.shape)
    idx_vars: dict[int, str] = {}

    def emit_loops(dims: list[int], body: Callable[[], None]) -> None:
        if not dims:
            body()
            return
        d, rest = dims[0], dims[1:]
        idx_vars[d] = cursor.fresh(f"_d{d}")
        with cursor.block(
            f"for (uint {idx_vars[d]} = 0; {idx_vars[d]} < {src.shape[d]}; ++{idx_vars[d]})"
        ):
            emit_loops(rest, body)

    def accumulate() -> None:
        acc = cursor.fresh("_acc")
        cursor.emit(f"{dst.ctype} {acc} = {init};")

        def inner_body() -> None:
            src_idx = _flat_index([(idx_vars[d], src_strides[d]) for d in range(rank)])
            cursor.emit(f"{acc} = {combine(acc, src.at(src_idx))};")

        emit_loops(reduced_dims, inner_body)
        dst_idx = _flat_index([(idx_vars[d], dst_strides[i]) for i, d in enumerate(kept_dims)])
        cursor.emit(f"{dst.at(dst_idx)} = {acc};")

    emit_loops(kept_dims, accumulate)


@rule("reduce_sum")
def _rule_reduce_sum(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`jnp.sum(x, axis=...)` -> nested loops, one per kept dim, each
    accumulating over the reduced dims.

    Verified against a real jaxpr before implementing: `jnp.sum` and
    `jnp.mean` both stage as `reduce_sum[axes=...]` (mean adds a plain
    `div` after, already emittable via ELEMENTWISE).
    """
    _emit_reduce(env, cursor, eqn, "0", lambda acc, x: f"{acc} + {x}")


@rule("reduce_max", "reduce_min")
def _rule_reduce_extremum(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """Extrema with dtype-specific comparison and accumulator identities."""
    op = eqn.primitive.name.removeprefix("reduce_")
    ctype = env.val(eqn.invars[0]).ctype
    _emit_reduce(
        env,
        cursor,
        eqn,
        extremum_identity(op, ctype),
        lambda acc, x: extremum(op, ctype, acc, x),
    )
