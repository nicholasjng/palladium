"""Specialized lowering for GEMM-shaped kernels.

A kernel whose jaxpr is exactly load, load, (lazy transpose,) matmul,
store gets a dedicated multi-SIMD-group lowering: one threadgroup of
several SIMD-groups (`_select_geometry`), each owning one program
instance (8 output rows), all sharing one staged rhs slab in
threadgroup memory. Staging cuts device rhs traffic by the group factor (every instance reads the
same rhs), which is what the general cooperative dot lacks: there, each
instance streams the full rhs with reuse only through the cache, and
throughput plateaus well under the CPU's GEMM. Result fragments store
straight to the device output (`simdgroup_store` takes a row stride), so
no threadgroup result tile is needed and the slab gets the whole budget.

The pattern gate is deliberately narrow: kernels that do anything beyond
the bare matmul (attention's softmax chain, epilogues) keep the general
cooperative lowering unchanged.
"""

from __future__ import annotations

import dataclasses

from jax.extend.core import Var

from palladium.emit.core import (
    SIMDGROUP_WIDTH,
    Atom,
    CVal,
    EmitState,
    _shaped,
    _transpose_is_dot_rhs_only,
)
from palladium.trace import KernelSpec

__all__ = ["emit_gemm", "gemm_desc", "gemm_groups"]

# Output columns per fragment pass: 8 accumulator fragments at 8 rows,
# the same register budget as the general MMA dot.
_TILE_N = 64

# Output rows per instance; the gate requires exactly this.
_ROWS = 8


def _select_geometry(grid0: int, k: int) -> tuple[int, int] | None:
    """(simdgroups per threadgroup, k-slab depth), or None if no viable
    configuration exists for this shape.

    Both knobs are occupancy knobs, swept on an M1 Pro at 4096x512x512
    (paired-interleaved medians): (G=16, slab 128) 835 GF/s,
    (16, 64) 759, (8, 64) 740-810, (4, 64) 379, (8, 128) 531,
    (4, 128) 327, against XLA-CPU's ~467. Big slabs pay off only with
    enough SIMD-groups per threadgroup to hide the staging barriers;
    small G starves the core outright. G also needs enough threadgroups
    left over to spread across cores, hence the grid0 // g >= 8 floor.
    """
    for g in (16, 8, 4):
        if grid0 % g == 0 and grid0 // g >= 8:
            if g == 16 and k % 128 == 0:
                return g, 128
            if k % 64 == 0:
                return g, 64
            if k % 32 == 0:
                return g, 32
            return None
    return None


@dataclasses.dataclass(frozen=True)
class GemmDesc:
    """One matched GEMM kernel: operand indices, shapes, and geometry."""

    lhs_idx: int  # spec.inputs index of the lhs ref
    rhs_idx: int  # spec.inputs index of the rhs ref
    transposed: bool  # rhs stored (n, k), contraction via transpose
    k: int
    n: int
    groups: int  # SIMD-groups (instances) per threadgroup
    kslab: int  # k-slab depth staged per barrier round


def _full_block_get(eqn) -> bool:
    return len(eqn.invars) == 1


def gemm_desc(spec: KernelSpec) -> GemmDesc | None:
    """Match the load/load/(transpose)/dot/store jaxpr, or None.

    Shape gates keep the emission simple and fast: 8 lhs rows per
    instance, k and n divisible by the slab/tile sizes, a rank-1 grid
    whose extent divides the group count, f32 throughout, and an rhs
    block covering its whole array (every instance reads the same rhs,
    which is what makes staging shareable).
    """
    jaxpr = spec.jaxpr
    if jaxpr is None or len(spec.grid) != 1 or len(jaxpr.eqns) not in (4, 5):
        return None
    if len(spec.inputs) != 2 or len(spec.outputs) != 1:
        return None
    names = [e.primitive.name for e in jaxpr.eqns]
    transposed = names == ["get", "get", "transpose", "dot_general", "swap"]
    if not transposed and names != ["get", "get", "dot_general", "swap"]:
        return None
    get_a, get_b = jaxpr.eqns[0], jaxpr.eqns[1]
    dot, swap = jaxpr.eqns[-2], jaxpr.eqns[-1]
    if not (_full_block_get(get_a) and _full_block_get(get_b)):
        return None
    in_refs = list(jaxpr.invars[:2])
    if get_a.invars[0] not in in_refs or get_b.invars[0] not in in_refs:
        return None
    if get_a.invars[0] is get_b.invars[0]:
        return None

    # Map the dot operands back to the two gets (rhs possibly through
    # the transpose).
    rhs_var: Atom = dot.invars[1]
    if transposed:
        t = jaxpr.eqns[2]
        if tuple(t.params["permutation"]) != (1, 0) or rhs_var is not t.outvars[0]:
            return None
        if not _transpose_is_dot_rhs_only({t.outvars[0]: [dot]}, t):
            return None
        rhs_var = t.invars[0]
    lhs_var = dot.invars[0]
    var_to_get = {get_a.outvars[0]: get_a, get_b.outvars[0]: get_b}
    if lhs_var not in var_to_get or rhs_var not in var_to_get:
        return None
    # Refs are always Vars; the index() calls need that established.
    lhs_ref = var_to_get[lhs_var].invars[0]
    rhs_ref = var_to_get[rhs_var].invars[0]
    assert isinstance(lhs_ref, Var) and isinstance(rhs_ref, Var)
    lhs_idx = in_refs.index(lhs_ref)
    rhs_idx = in_refs.index(rhs_ref)

    (lc, rc), (lb, rb) = dot.params["dimension_numbers"]
    if lb or rb or tuple(lc) != (1,) or tuple(rc) != (0,):
        return None
    # The store must be the full output block of the dot result.
    if len(swap.invars) != 2 or swap.invars[0] is not jaxpr.invars[2]:
        return None
    if swap.invars[1] is not dot.outvars[0]:
        return None

    lhs_shape = tuple(int(d) for d in _shaped(lhs_var.aval).shape)
    out_shape = tuple(int(d) for d in _shaped(dot.outvars[0].aval).shape)
    if len(lhs_shape) != 2 or len(out_shape) != 2:
        return None
    rows, k = lhs_shape
    n = out_shape[1]
    if rows != _ROWS or n % _TILE_N != 0:
        return None
    geometry = _select_geometry(int(spec.grid[0]), k)
    if geometry is None:
        return None
    groups, kslab = geometry
    for info, block_rows in (
        (spec.inputs[lhs_idx], rows),
        (spec.outputs[0], rows),
    ):
        if info.dtype.name != "float32" or info.block_shape[0] != block_rows:
            return None
    rhs_info = spec.inputs[rhs_idx]
    if rhs_info.dtype.name != "float32":
        return None
    if rhs_info.block_shape != rhs_info.array_shape:
        return None
    return GemmDesc(
        lhs_idx=lhs_idx,
        rhs_idx=rhs_idx,
        transposed=transposed,
        k=k,
        n=n,
        groups=groups,
        kslab=kslab,
    )


def gemm_groups(spec: KernelSpec) -> int:
    """SIMD-groups per threadgroup for this kernel's launch geometry;
    1 unless the specialized GEMM lowering applies. `emit_msl`,
    `dispatch.bind`, and the FFI path must all agree on this value."""
    desc = gemm_desc(spec)
    return desc.groups if desc is not None else 1


def emit_gemm(state: EmitState, ref_vals: list[CVal], desc: GemmDesc) -> None:
    """Emit the multi-SIMD-group GEMM body.

    Loop nest: column tiles (jt) outer, staged k-slabs (ks) inner, so
    accumulator fragments persist across the whole contraction while the
    slab holds one (tile, slab) block at a time. All G*32 threads
    cooperate on each staging copy; barriers bracket it (full
    threadgroup_barrier, same reasoning as the cooperative MMA dot).
    Every SIMD-group then runs its own fragment loads: lhs 8x8 tiles
    from its instance's device rows, rhs tiles from the shared slab
    (with the hardware transpose load for (n, k) storage), and stores
    its accumulators straight to its device output rows.
    """
    a = ref_vals[desc.lhs_idx]
    b = ref_vals[desc.rhs_idx]
    o = ref_vals[2]
    k, n = desc.k, desc.n
    kslab = desc.kslab
    g_threads = desc.groups * SIMDGROUP_WIDTH
    slab = state.fresh("_slab")
    state.prologue.append(f"threadgroup float {slab}[{_TILE_N * kslab}];")
    state.tg_bytes += 4 * _TILE_N * kslab
    tid = state.fresh("_tid")
    state.emit(f"uint {tid} = (uint)_sg * {SIMDGROUP_WIDTH}u + (uint)_lane;")

    jt = state.fresh("_jt")
    with state.block(f"for (uint {jt} = 0; {jt} < {n // _TILE_N}u; ++{jt})"):
        cfrags = [state.fresh(f"_c{ci}") for ci in range(_TILE_N // 8)]
        for c in cfrags:
            state.emit(f"simdgroup_float8x8 {c} = simdgroup_float8x8(0.0f);")
        ks = state.fresh("_ks")
        with state.block(f"for (uint {ks} = 0; {ks} < {k}u; {ks} += {kslab}u)"):
            # Full barrier: the previous slab's readers must finish
            # before it is overwritten, and the fresh slab must be
            # visible to every SIMD-group's cooperative loads after.
            state.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
            i = state.fresh("_i")
            r = state.fresh("_r")
            cc = state.fresh("_cc")
            with state.block(
                f"for (uint {i} = {tid}; {i} < {_TILE_N * kslab}u; {i} += {g_threads}u)"
            ):
                if desc.transposed:
                    # rhs stored (n, k): slab row = output column.
                    state.emit(f"uint {r} = {i} / {kslab}u;")
                    state.emit(f"uint {cc} = {i} % {kslab}u;")
                    state.emit(
                        f"{slab}[{i}] = {b.expr}[({jt} * {_TILE_N}u + {r}) "
                        f"* {k}u + {ks} + {cc}];"
                    )
                else:
                    # rhs stored (k, n): slab row = contraction index.
                    state.emit(f"uint {r} = {i} / {_TILE_N}u;")
                    state.emit(f"uint {cc} = {i} % {_TILE_N}u;")
                    state.emit(
                        f"{slab}[{i}] = {b.expr}[({ks} + {r}) * {n}u "
                        f"+ {jt} * {_TILE_N}u + {cc}];"
                    )
            state.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
            af = state.fresh("_af")
            bf = state.fresh("_bf")
            kt = state.fresh("_kt")
            with state.block(f"for (uint {kt} = 0; {kt} < {kslab // 8}u; ++{kt})"):
                state.emit(f"simdgroup_float8x8 {af};")
                state.emit(f"simdgroup_load({af}, {a.expr} + {ks} + {kt} * 8u, {k}u);")
                state.emit(f"simdgroup_float8x8 {bf};")
                for ci, c in enumerate(cfrags):
                    if desc.transposed:
                        state.emit(
                            f"simdgroup_load({bf}, {slab} + {ci * 8 * kslab}u "
                            f"+ {kt} * 8u, {kslab}u, ulong2(0, 0), true);"
                        )
                    else:
                        state.emit(
                            f"simdgroup_load({bf}, {slab} + {kt} * {8 * _TILE_N}u "
                            f"+ {ci * 8}u, {_TILE_N}u);"
                        )
                    state.emit(f"simdgroup_multiply_accumulate({c}, {af}, {bf}, {c});")
        for ci, c in enumerate(cfrags):
            state.emit(
                f"simdgroup_store({c}, {o.expr} + {jt} * {_TILE_N}u + {ci * 8}u, {n}u);"
            )
