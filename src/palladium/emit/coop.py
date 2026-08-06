"""Lowering rules for the simdgroup-cooperative execution model.

One 32-thread SIMD-group runs one Pallas program instance covering R
query rows; values are distributed over lanes per the layout taxonomy
below. Entered per kernel by `is_simdgroup_cooperative`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

from jax.extend.core import Jaxpr, JaxprEqn, Var

from palladium.emit.core import (
    CTYPES,
    ELEMENTWISE,
    PRIMITIVE_INVARS,
    SIMDGROUP_WIDTH,
    Atom,
    CVal,
    EmitError,
    EmitState,
    RuleFn,
    _emit_with_rules,
    _ref_view,
    _shaped,
    _template_fields,
    _transpose_is_dot_rhs_only,
    _unwrapped,
)
from palladium.emit.rules import (
    _check_pure_carry_scan,
    _rule_program_id,
    _rule_reshape,
)
from palladium.trace import KernelSpec

# ===========================================================================
# Simdgroup-cooperative execution model.
#
# The default model maps one Metal thread to one Pallas program instance.
# Kernels dominated by small per-instance matmuls are latency-bound that
# way: too few SIMD-groups in flight to hide device-memory latency. This
# model maps one 32-thread SIMD-group to one instance covering R query
# rows (R = EmitState.coop_rows, inferred from the kernel's dot_general
# lhs shapes; R = 1 is the single-row case and everything degenerates to
# it exactly). Lanes own columns for all R rows, so each rhs element
# loaded from device serves R rows instead of one.
#
# Cross-lane traffic uses SIMD-group functions (simd_broadcast,
# simd_sum, simd_max) and, for qualifying dots, the SIMD-group matrix
# units with a threadgroup result tile. Control flow is uniform throughout; lane validity is handled
# with identity-masked values, never divergent branches around a
# SIMD-group operation.
#
# Value layouts (CVal.layout), for a kernel at R rows:
#   "rep"    - size-1 values and scalars: every lane computes the
#              identical value redundantly. Uniform because every rep
#              input is itself uniform (loop counters, literals).
#   "rowvec" - shape (R,) or (R, 1), R > 1: one value per row, R elements
#              per lane, every lane holds all rows. Row-uniform.
#   "shard"  - shape (R, W), W > 1: lane t holds columns
#              [t*ppw, (t+1)*ppw) of every row, ppw = ceil(W/32), stored
#              row-major as R*ppw thread elements. Lanes past W hold
#              garbage that is never observed: reductions mask it with
#              the combine identity, stores guard it, device reads clamp
#              the index in-bounds. (At R = 1 this is a flat shard.)
#   Device views (space == "device") need no layout: any lane reads any
#              element through the pointer.
#
# Entered per kernel by is_simdgroup_cooperative(): only when the jaxpr
# contains an m==R dot_general (one consistent R) fed by a device-view
# rhs, every staged primitive has a cooperative lowering, and every
# intermediate value fits the taxonomy. All other kernels keep the
# one-thread-per-instance model.
#
# Performance constraints encoded below (violating any of these was
# slower when tried):
# - Dot accumulators are individually named registers (python-side
#   unroll): dynamically indexed thread arrays can spill to stack.
# - The dot lhs is re-read from device each loop iteration on purpose:
#   per-lane register copies of it cost more in occupancy than the
#   redundant, lane-uniform loads.
# - The dot rhs is read directly from device, not staged through
#   threadgroup memory: the per-threadgroup allocation costs more in
#   occupancy than the load coalescing wins at these shapes.
# ===========================================================================

_COOP_LANES = SIMDGROUP_WIDTH

# Lower qualifying cooperative dots on the SIMD-group matrix units
# (simdgroup_float8x8). Module-level so an interleaved A/B can toggle it
# per build.
_COOP_MMA_DOT = True

# Accumulator fragments live at once in the MMA dot. Wide outputs are
# column-tiled so this bound, not the output width, sets register
# pressure; 8 measured best on the attention shapes (16 measured worse).
_MMA_FRAGMENT_CAP = 8

# Threadgroup bytes one MMA dot's result tile may claim (the full output
# width is staged). Two such dots plus elementwise staging tiles still
# fit the 32KB hardware budget checked at emission.
_MMA_TG_TILE_LIMIT = 16384


def _coop_pl(size: int) -> int:
    """Elements per lane for a flat-sharded value: ceil(size / 32)."""
    return -(-size // _COOP_LANES)


def _coop_geom(rows: int, shape: tuple[int, ...]) -> tuple[str, int, int]:
    """Classify a logical shape under the R-row layout taxonomy.

    Returns (kind, W, ppw): kind is "rep" | "rowvec" | "shard"; W is the
    per-row width and ppw the columns per lane (both 1 unless "shard").

    Raises
    ------
    EmitError
        If the shape fits no layout at this R. The detection walker runs
        the same classification first, so reaching that from emission
        means walker and rules disagree.
    """
    size = math.prod(shape)
    if size <= 1:
        return ("rep", 1, 1)
    if rows == 1:
        return ("shard", size, _coop_pl(size))
    if size == rows and shape[0] == rows:
        return ("rowvec", 1, 1)
    if len(shape) == 2 and shape[0] == rows:
        return ("shard", shape[1], _coop_pl(shape[1]))
    raise EmitError(
        f"cooperative layout: shape {shape} does not fit the R={rows} row "
        "taxonomy (scalar, (R,), (R,1), or (R,W))"
    )


def _coop_kind(val: CVal) -> str:
    """How a value is accessed in the cooperative model: "device" (any
    lane indexes the pointer; also threadgroup-memory views), "shard"/
    "rowvec" (per-lane storage), "rep" (size-1/scalar, identical on
    every lane)."""
    if val.space in ("device", "threadgroup") and val.shape:
        return "device"
    if val.layout in ("shard", "rowvec"):
        return val.layout
    return "rep"


class _CoopUnsupported(Exception):
    """Raised by the detection walker: this jaxpr can't lower cooperatively."""


_COOP_ELEMENTWISE = set(ELEMENTWISE) | {
    "integer_pow",
    "convert_element_type",
    "bitcast_convert_type",
    "not",
}


def is_simdgroup_cooperative(spec: KernelSpec) -> bool:
    """Whether `emit_msl` will (by default) lower this kernel with the
    simdgroup-cooperative execution model.

    True iff the kernel jaxpr contains at least one `m == R` standard
    `dot_general` whose rhs is a (possibly lazily-transposed) view of an
    input ref, and every staged primitive has a cooperative lowering.
    `dispatch.bind` uses the same predicate to pick the launch geometry
    (one 32-thread threadgroup per program instance), so the decision
    must be a pure function of the spec.
    """
    return coop_probe(spec)[0] is not None


def coop_rows(spec: KernelSpec) -> int | None:
    """Rows per program instance (R) if this kernel lowers cooperatively,
    else None. `is_simdgroup_cooperative(spec)` is `coop_rows(spec) is not
    None`; `emit_msl` uses the value itself."""
    return coop_probe(spec)[0]


def coop_probe(spec: KernelSpec) -> tuple[int | None, str | None]:
    """(rows, reason) for the cooperative-model decision.

    `rows` is R when the kernel lowers cooperatively (reason None);
    otherwise rows is None and `reason` names the first thing that kept
    the kernel on the classic model. The classic model is a correct but
    much slower lowering for dot-heavy kernels, so the reason is worth
    surfacing; `explain()` and PALLADIUM_EXPLAIN do.
    """
    if spec.jaxpr is None:
        # Hand-built specs (tests, raw-MSL bind users) may carry no
        # jaxpr; they can only mean the classic model.
        return None, "spec carries no jaxpr"
    if not spec.grid or len(spec.grid) > 3:
        return None, f"grid {spec.grid} is empty or rank > 3"
    from palladium.emit.gemm import _ROWS, gemm_desc

    if gemm_desc(spec) is not None:
        # The specialized GEMM lowering needs no per-instance result
        # tile or accumulator array, so the walker's width caps below
        # do not apply to it.
        return _ROWS, None
    n_in = len(spec.inputs)
    ro_refs = set(spec.jaxpr.invars[:n_in])
    try:
        rows = _coop_rows_checked(spec, ro_refs)
    except _CoopUnsupported as e:
        return None, str(e)
    if rows is None:
        return None, "no qualifying dot_general in the kernel"
    return rows, None


def _coop_rows_checked(spec: KernelSpec, ro_refs: set[Var]) -> int | None:
    """Infer R from the kernel's dot_general lhs shapes, then run the
    detection walk at that R. Returns R, or None when there is no dot to
    justify the cooperative model; raises _CoopUnsupported when there is
    one but something else can't lower."""
    ms: set[int] = set()
    _coop_collect_dot_rows(spec.jaxpr, ms)
    if not ms:
        return None
    if -1 in ms:
        raise _CoopUnsupported("dot_general lhs is not rank-2")
    if len(ms) != 1:
        raise _CoopUnsupported(f"dot_generals disagree on m: {sorted(ms)}")
    rows = ms.pop()
    if rows < 1 or rows > _COOP_LANES or _COOP_LANES % rows != 0:
        raise _CoopUnsupported(f"m={rows} does not divide the SIMD-group evenly")
    if not _coop_walk(spec.jaxpr, ro_refs, set(), set(), rows):
        return None
    return rows


def _coop_collect_dot_rows(jaxpr: Jaxpr, out: set[int]) -> None:
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            lshape = _shaped(eqn.invars[0].aval).shape
            out.add(int(lshape[0]) if len(lshape) == 2 else -1)
        elif eqn.primitive.name == "scan":
            _coop_collect_dot_rows(eqn.params["jaxpr"], out)


def _coop_walk(
    jaxpr: Jaxpr,
    ro_refs: set[Var],
    device_vars: set[Var],
    transposed_vars: set[Var],
    rows: int,
) -> bool:
    """Detection pass mirroring the COOP_RULES' own preconditions; raises
    _CoopUnsupported on the first primitive the cooperative emitter can't
    lower at R=rows, returns whether a qualifying dot_general was found."""
    found_dot = False
    consumers: dict[Var, list[JaxprEqn | None]] = {}
    for eqn in jaxpr.eqns:
        for iv in eqn.invars:
            if isinstance(iv, Var):
                consumers.setdefault(iv, []).append(eqn)
    for ov in jaxpr.outvars:
        if isinstance(ov, Var):
            consumers.setdefault(ov, []).append(None)

    def size_of(atom: Atom) -> int:
        return math.prod(tuple(int(d) for d in _shaped(atom.aval).shape))

    def shape_of(atom: Atom) -> tuple[int, ...]:
        return tuple(int(d) for d in _shaped(atom.aval).shape)

    def classify(atom: Atom) -> tuple[str, int, int]:
        try:
            return _coop_geom(rows, shape_of(atom))
        except EmitError as e:
            raise _CoopUnsupported(str(e)) from None

    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name == "get":
            if eqn.invars[0] not in ro_refs:
                raise _CoopUnsupported("get from a non-input ref")
            device_vars.add(eqn.outvars[0])
        elif name == "swap":
            if eqn.invars[0] in ro_refs:
                raise _CoopUnsupported("swap on an input ref")
            if eqn.outvars[0] in consumers:
                raise _CoopUnsupported("swap result is read")
        elif name == "transpose":
            if eqn.invars[0] not in device_vars:
                raise _CoopUnsupported("transpose of a non-device value")
            if not _transpose_is_dot_rhs_only(consumers, eqn):
                raise _CoopUnsupported("transpose not fusable into dot_general")
            device_vars.add(eqn.outvars[0])
            transposed_vars.add(eqn.outvars[0])
        elif name == "reshape":
            if eqn.params["dimensions"] is not None:
                raise _CoopUnsupported("reshape with permutation")
            if eqn.invars[0] in device_vars:
                device_vars.add(eqn.outvars[0])
        elif name == "dot_general":
            (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
            if lb or rb or tuple(lc) != (1,) or tuple(rc) != (0,):
                raise _CoopUnsupported("non-standard dot_general contraction")
            lshape = shape_of(eqn.invars[0])
            if len(lshape) != 2 or lshape[0] != rows:
                raise _CoopUnsupported(f"dot_general with m != R ({lshape})")
            if eqn.invars[1] not in device_vars:
                raise _CoopUnsupported("dot_general rhs is not a device view")
            if eqn.invars[0] is eqn.invars[1]:
                raise _CoopUnsupported("dot_general lhs aliases rhs")
            if eqn.invars[0] not in device_vars:
                classify(eqn.invars[0])  # sharded lhs must fit the taxonomy
            _, _, ppn = classify(eqn.outvars[0])
            out_shape = shape_of(eqn.outvars[0])
            n_dim = int(out_shape[1]) if len(out_shape) == 2 else 1

            def _f32(atom: Atom) -> bool:
                return str(_shaped(atom.aval).dtype) == "float32"

            # Mirrors the rule's MMA eligibility: the MMA path column-tiles
            # wide outputs at fixed register pressure, so its width is
            # bounded by the threadgroup result tile, not the accumulator
            # count. Every other path python-unrolls rows*ppn accumulator
            # registers, so it keeps the hard cap.
            mma_ok = (
                _COOP_MMA_DOT
                and eqn.invars[0] in device_vars
                and eqn.invars[0] not in transposed_vars
                and _f32(eqn.invars[0])
                and _f32(eqn.invars[1])
                and _f32(eqn.outvars[0])
                and rows % 8 == 0
                and int(lshape[1]) % 8 == 0
                and n_dim % 8 == 0
                and rows * n_dim * 4 <= _MMA_TG_TILE_LIMIT
            )
            if not mma_ok and rows * ppn > 64:
                raise _CoopUnsupported("dot_general accumulator set too large")
            found_dot = True
        elif name in ("reduce_sum", "reduce_max"):
            out_size = size_of(eqn.outvars[0])
            axes = tuple(eqn.params["axes"])
            in_shape = shape_of(eqn.invars[0])
            classify(eqn.invars[0])
            classify(eqn.outvars[0])
            if out_size == 1:
                pass  # full reduction, any layout source
            elif (
                rows > 1
                and out_size == rows
                and len(in_shape) == 2
                and in_shape[0] == rows
                and axes == (1,)
            ):
                pass  # per-row reduction of an (R, W) shard
            else:
                raise _CoopUnsupported("unsupported reduction shape/axes")
        elif name == "broadcast_in_dim":
            src_size = size_of(eqn.invars[0])
            src_kind = classify(eqn.invars[0])[0] if src_size > 1 else "rep"
            dst_kind = classify(eqn.outvars[0])[0]
            if src_size == 1 or src_size == size_of(eqn.outvars[0]):
                pass  # scalar fill or pure rank change (alias)
            elif src_kind == "rowvec" and dst_kind == "shard":
                # (R,1) -> (R,W) fill needs the broadcast dims to keep the
                # row axis leading.
                if tuple(eqn.params["broadcast_dimensions"])[:1] != (0,):
                    raise _CoopUnsupported("row broadcast with non-leading rows")
            else:
                raise _CoopUnsupported("unsupported broadcast source layout")
        elif name == "scan":
            num_carry = len(eqn.outvars)
            num_consts = len(eqn.invars) - num_carry
            body: Jaxpr = eqn.params["jaxpr"]
            try:
                _check_pure_carry_scan(eqn, body)
            except EmitError as e:
                raise _CoopUnsupported(str(e)) from None
            if eqn.params.get("reverse", False):
                raise _CoopUnsupported("reverse-mode scan")
            inner_ro = {
                bv
                for bv, ov in zip(body.invars[:num_consts], eqn.invars[:num_consts])
                if isinstance(ov, Var) and ov in ro_refs
            }
            inner_dev = {
                bv
                for bv, ov in zip(body.invars[:num_consts], eqn.invars[:num_consts])
                if isinstance(ov, Var) and ov in device_vars
            }
            inner_trans = {
                bv
                for bv, ov in zip(body.invars[:num_consts], eqn.invars[:num_consts])
                if isinstance(ov, Var) and ov in transposed_vars
            }
            for invar in eqn.invars[num_consts:]:
                if isinstance(invar, Var):
                    classify(invar)
            found_dot |= _coop_walk(body, inner_ro, inner_dev, inner_trans, rows)
        elif name == "program_id":
            pass
        elif name in _COOP_ELEMENTWISE:
            out_kind, out_w, _ = classify(eqn.outvars[0])
            for iv in eqn.invars:
                if size_of(iv) == 1:
                    continue
                kind, w, _ = classify(iv)
                # A shard operand must match the output's width exactly
                # (same column ownership); rowvec broadcasts along W.
                if kind == "shard" and (out_kind != "shard" or w != out_w):
                    raise _CoopUnsupported("elementwise shard width mismatch")
                if kind == "rowvec" and out_kind == "rep":
                    raise _CoopUnsupported("elementwise folds rows into a scalar")
        else:
            raise _CoopUnsupported(f"no cooperative rule for '{name}'")
    return found_dot


COOP_RULES: dict[str, RuleFn] = {}


def coop_rule(*names: str) -> Callable[[RuleFn], RuleFn]:
    """Register a cooperative lowering rule, mirroring `rule`."""

    def register(fn: RuleFn) -> RuleFn:
        for n in names:
            COOP_RULES[n] = fn
        return fn

    return register


def emit_jaxpr_coop(state: EmitState, jaxpr: Jaxpr, in_vals: list[CVal]) -> list[CVal]:
    """Walk a jaxpr with the cooperative rules (COOP_RULES).

    Every primitive here was vetted by `_coop_walk` before emission, so a
    missing rule is an internal inconsistency, not user error.
    """
    return _emit_with_rules(
        state,
        jaxpr,
        in_vals,
        COOP_RULES,
        lambda name: EmitError(
            f"no cooperative MSL rule for '{name}'; "
            "is_simdgroup_cooperative and COOP_RULES disagree"
        ),
    )


def _coop_declare(state: EmitState, var: Var) -> CVal:
    """Per-lane storage for `var` under the R-row taxonomy: a 1-element
    array for size-1 values ("rep"), an R-element array for per-row
    scalars ("rowvec"), an R*ppw-element row-major array for (R, W)
    values ("shard"). The CVal keeps the *logical* shape; only COOP_RULES
    may index it (they know the layout)."""
    aval = _shaped(var.aval)
    ctype = CTYPES[str(aval.dtype)]
    shape = tuple(int(d) for d in aval.shape)
    kind, _, ppw = _coop_geom(state.coop_rows, shape)
    name = state.fresh()
    if not shape:
        state.emit(f"{ctype} {name};")
        cval = CVal(expr=name, shape=(), ctype=ctype)
    elif kind == "rep":
        state.emit(f"{ctype} {name}[1];")
        cval = CVal(expr=name, shape=shape, ctype=ctype)
    elif kind == "rowvec":
        state.emit(f"{ctype} {name}[{state.coop_rows}];")
        cval = CVal(expr=name, shape=shape, ctype=ctype, layout="rowvec")
    else:
        state.emit(f"{ctype} {name}[{state.coop_rows * ppw}];")
        cval = CVal(expr=name, shape=shape, ctype=ctype, layout="shard")
    state.env[var] = cval
    return cval


def _coop_global_index(state: EmitState, pl: int, e: str) -> str:
    """Emit and return `g = lane * per_lane + e` as a fresh uint: the
    global column this lane's local element `e` maps to."""
    g = state.fresh("_g")
    state.emit(f"uint {g} = (uint)_lane * {pl}u + {e};")
    return g


def _coop_storage_len(state: EmitState, val: CVal) -> int:
    """Per-lane element count of declared coop storage."""
    kind, _, ppw = _coop_geom(state.coop_rows, val.shape)
    if kind == "rep":
        return 1
    if kind == "rowvec":
        return state.coop_rows
    return state.coop_rows * ppw


def _coop_copy(state: EmitState, dst: CVal, src: CVal) -> None:
    """Copy any coop value into declared rep/rowvec/shard storage of equal
    logical shape (scan carries, snapshots)."""
    rows = state.coop_rows
    kind_src = _coop_kind(src)
    if dst.layout == "rowvec":
        r = state.fresh("_r")
        with state.block(f"for (uint {r} = 0; {r} < {rows}u; ++{r})"):
            if kind_src == "rowvec":
                state.emit(f"{dst.expr}[{r}] = {src.expr}[{r}];")
            elif kind_src == "rep":
                state.emit(f"{dst.expr}[{r}] = {src.at('0')};")
            else:  # device (R,1)/(R,) view
                state.emit(f"{dst.expr}[{r}] = {src.expr}[{r}];")
        return
    if dst.layout != "shard":
        state.emit(f"{dst.at('0')} = {src.at('0')};")
        return
    _, w, ppw = _coop_geom(rows, dst.shape)
    ragged = ppw * _COOP_LANES != w
    r = state.fresh("_r")
    e = state.fresh("_e")
    with (
        state.block(f"for (uint {r} = 0; {r} < {rows}u; ++{r})"),
        state.block(f"for (uint {e} = 0; {e} < {ppw}u; ++{e})"),
    ):
        li = f"{r} * {ppw}u + {e}" if rows > 1 else e
        if kind_src == "shard":
            state.emit(f"{dst.expr}[{li}] = {src.expr}[{li}];")
        elif kind_src == "rowvec":
            state.emit(f"{dst.expr}[{li}] = {src.expr}[{r}];")
        elif kind_src == "rep":
            state.emit(f"{dst.expr}[{li}] = {src.at('0')};")
        else:  # device view: clamp keeps out-of-range lanes in bounds
            g = _coop_global_index(state, ppw, e)
            gc = f"min({g}, {w - 1}u)" if ragged else g
            state.emit(f"{dst.expr}[{li}] = {src.expr}[{r} * {w}u + {gc}];")


@coop_rule("get")
def _coop_rule_get(state: EmitState, eqn: JaxprEqn) -> None:
    """Cooperative load: always a device pointer view, never a copy.

    Per-lane copies of a shared block would replicate data 32x, and
    every consumer knows how to index device storage directly. Register
    copies of the dot lhs in particular are slower than the redundant,
    lane-uniform device reads (occupancy loss or stack spill).
    """
    indexer_args = eqn.params["tree"].unflatten(eqn.invars[1:])
    src = state.val(eqn.invars[0])
    if indexer_args:
        (indexer,) = indexer_args
        src = _ref_view(state, src, indexer)
    if not (src.readonly and src.space == "device"):
        raise EmitError("cooperative get: only input (const) refs are supported")
    aval = _shaped(eqn.outvars[0].aval)
    shape = tuple(int(d) for d in aval.shape)
    if shape and math.prod(shape) != src.size:
        raise EmitError(
            f"cooperative get: view size {src.size} != output size {math.prod(shape)}"
        )
    state.bind(eqn.outvars[0], dataclasses.replace(src, shape=shape or src.shape))


@coop_rule("swap")
def _coop_rule_swap(state: EmitState, eqn: JaxprEqn) -> None:
    """Cooperative store: sharded values write each lane's own elements
    (guarded when 32*per_lane overshoots the size); size-1 values write
    from lane 0 only. Divergence is fine here; no SIMD-group function
    executes inside the guards."""
    ref, value = eqn.invars[0], eqn.invars[1]
    indexer_args = eqn.params["tree"].unflatten(eqn.invars[2:])
    dst_ref = state.val(ref)
    if indexer_args:
        (indexer,) = indexer_args
        dst_ref = _ref_view(state, dst_ref, indexer)
    stored = state.val(value)
    size = dst_ref.size
    rows = state.coop_rows
    kind = _coop_kind(stored)
    if kind in ("rep", "rowvec"):
        # Every lane holds the identical value(s); lane 0 writes them.
        with state.block("if (_lane == 0)"):
            if kind == "rep":
                state.copy(dst_ref, stored, size)
            else:
                r = state.fresh("_r")
                with state.block(f"for (uint {r} = 0; {r} < {rows}u; ++{r})"):
                    state.emit(f"{dst_ref.at(r)} = {stored.expr}[{r}];")
        state.bind(eqn.outvars[0], dst_ref)
        return
    _, w, ppw = _coop_geom(rows, stored.shape)
    ragged = ppw * _COOP_LANES != w
    r = state.fresh("_r")
    e = state.fresh("_e")
    with (
        state.block(f"for (uint {r} = 0; {r} < {rows}u; ++{r})"),
        state.block(f"for (uint {e} = 0; {e} < {ppw}u; ++{e})"),
    ):
        g = _coop_global_index(state, ppw, e)
        li = f"{r} * {ppw}u + {e}" if rows > 1 else e
        gi = f"{r} * {w}u + {g}" if rows > 1 else g
        src_elem = f"{stored.expr}[{li}]" if kind == "shard" else f"{stored.expr}[{gi}]"
        if ragged:
            with state.block(f"if ({g} < {w}u)"):
                state.emit(f"{dst_ref.at(gi)} = {src_elem};")
        else:
            state.emit(f"{dst_ref.at(gi)} = {src_elem};")
    state.bind(eqn.outvars[0], dst_ref)


def _coop_elementwise_tg_dst(state: EmitState, eqn: JaxprEqn) -> CVal | None:
    """Threadgroup placement for an elementwise output that feeds a
    SIMD-group matrix dot as lhs (fragments load from memory, not from
    lane-sharded registers). Returns a threadgroup-view CVal to write
    into, or None for the normal per-lane storage."""
    if not _COOP_MMA_DOT:
        return None
    rows = state.coop_rows
    outvar = eqn.outvars[0]
    shape = tuple(int(d) for d in _shaped(outvar.aval).shape)
    try:
        kind, w, _ = _coop_geom(rows, shape)
    except EmitError:
        return None
    if (
        kind != "shard"
        or CTYPES[str(_shaped(outvar.aval).dtype)] != "float"
        or rows % 8 != 0
        or w % 8 != 0
        or rows * w * 4 > 4096
    ):
        return None
    uses = state.consumers.get(outvar, [])
    feeds_dot_lhs = any(
        u is not None
        and u.primitive.name == "dot_general"
        and u.invars[0] is outvar
        and u.invars[1] is not outvar
        for u in uses
    )
    if not feeds_dot_lhs:
        return None
    tg = state.fresh("_tg")
    state.prologue.append(f"threadgroup float {tg}[{rows * w}];")
    state.tg_bytes += 4 * rows * w
    return CVal(expr=tg, shape=shape, ctype="float", space="threadgroup")


def _coop_rule_elementwise(state: EmitState, eqn: JaxprEqn) -> None:
    """Elementwise over coop layouts: rep outputs are a single uniform
    assignment; sharded outputs loop over the lane's own elements, with
    size-1 operands read at [0], sharded operands at the local index, and
    device operands at the (clamped, in-bounds) global index."""
    ops = [state.val(v) for v in eqn.invars]
    opname = eqn.primitive.name
    tg_dst = _coop_elementwise_tg_dst(state, eqn)
    dst = tg_dst if tg_dst is not None else _coop_declare(state, eqn.outvars[0])

    if opname == "not":
        template = "(!{a})" if ops[0].ctype == "bool" else "(~{a})"
    elif opname == "integer_pow":
        exp: int = eqn.params["y"]
        body = " * ".join(["{a}"] * abs(exp))
        template = f"({body})" if exp > 0 else f"(1.0f / ({body}))"
    elif opname == "convert_element_type":
        template = f"(({dst.ctype}){{a}})"
    elif opname == "bitcast_convert_type":
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

    rows = state.coop_rows

    def emit_stmt(target: str, elem: Callable[[CVal], str]) -> None:
        inputs = {
            name: elem(op) for name, op in zip(PRIMITIVE_INVARS, ops, strict=False)
        }
        state.emit(f"{target} = {_unwrapped(template.format(**inputs))};")

    is_tg = tg_dst is not None
    if not is_tg and dst.layout == "rowvec":
        r = state.fresh("_r")
        with state.block(f"for (uint {r} = 0; {r} < {rows}u; ++{r})"):

            def elem_rv(op: CVal) -> str:
                if op.size == 1:
                    return op.at("0")
                # rowvec storage or a device (R,1)/(R,) view: one per row.
                return f"{op.expr}[{r}]"

            emit_stmt(f"{dst.expr}[{r}]", elem_rv)
        return

    if not is_tg and dst.layout != "shard":
        emit_stmt(dst.at("0"), lambda op: op.at("0"))
        return

    _, w, ppw = _coop_geom(rows, dst.shape)
    ragged = ppw * _COOP_LANES != w
    needs_g = is_tg or any(
        _coop_kind(op) == "device" and _coop_geom(rows, op.shape)[0] == "shard"
        for op in ops
        if op.size > 1
    )
    if is_tg:
        # Threadgroup destination: barrier against the previous loop
        # iteration's readers before overwriting the tile. Must be the
        # full threadgroup_barrier; see _coop_emit_dot_mma.
        state.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
    r = state.fresh("_r")
    e = state.fresh("_e")
    with (
        state.block(f"for (uint {r} = 0; {r} < {rows}u; ++{r})"),
        state.block(f"for (uint {e} = 0; {e} < {ppw}u; ++{e})"),
    ):
        gc = ""
        g = ""
        if needs_g:
            g = _coop_global_index(state, ppw, e)
            gc = f"min({g}, {w - 1}u)" if ragged else g
        li = f"{r} * {ppw}u + {e}" if rows > 1 else e

        def elem_sh(op: CVal) -> str:
            if op.size == 1:
                return op.at("0")
            kind = _coop_kind(op)
            op_kind = _coop_geom(rows, op.shape)[0]
            if kind == "device":
                if op_kind == "shard":
                    return (
                        f"{op.expr}[{r} * {w}u + {gc}]"
                        if rows > 1
                        else (f"{op.expr}[{gc}]")
                    )
                return f"{op.expr}[{r}]"  # device (R,1)/(R,) view
            if kind == "rowvec":
                return f"{op.expr}[{r}]"
            return f"{op.expr}[{li}]"

        if is_tg:
            target = f"{dst.expr}[{r} * {w}u + {g}]"
            if ragged:
                with state.block(f"if ({g} < {w}u)"):
                    emit_stmt(target, elem_sh)
            else:
                emit_stmt(target, elem_sh)
        else:
            emit_stmt(f"{dst.expr}[{li}]", elem_sh)
    if is_tg:
        state.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        state.bind(eqn.outvars[0], dst)


for _name in _COOP_ELEMENTWISE:
    COOP_RULES[_name] = _coop_rule_elementwise


@coop_rule("broadcast_in_dim")
def _coop_rule_broadcast_in_dim(state: EmitState, eqn: JaxprEqn) -> None:
    """Layout-aware broadcast: a pure rank change aliases; a size-1
    source fills the destination; a rowvec source fills each row's
    columns from that row's value. Anything else is rejected by
    `_coop_walk` before emission."""
    src = state.val(eqn.invars[0])
    aval = _shaped(eqn.outvars[0].aval)
    out_shape = tuple(int(d) for d in aval.shape)
    if src.shape and src.size == math.prod(out_shape):
        # (R,) <-> (R,1) and friends: same storage, new logical shape.
        # Scalar-expression sources (literals, rank-0 values) fall through
        # to the fill path instead: they have no storage to alias, and
        # re-shaping their CVal would make `.at()` emit `literal[0]`.
        state.bind(eqn.outvars[0], dataclasses.replace(src, shape=out_shape))
        return
    dst = _coop_declare(state, eqn.outvars[0])
    rows = state.coop_rows
    kind_src = _coop_kind(src)
    if dst.layout == "rowvec":
        r = state.fresh("_r")
        with state.block(f"for (uint {r} = 0; {r} < {rows}u; ++{r})"):
            val = f"{src.expr}[{r}]" if kind_src == "rowvec" else src.at("0")
            state.emit(f"{dst.expr}[{r}] = {val};")
        return
    if dst.layout != "shard":
        state.emit(f"{dst.at('0')} = {src.at('0')};")
        return
    _, _, ppw = _coop_geom(rows, dst.shape)
    r = state.fresh("_r")
    e = state.fresh("_e")
    with (
        state.block(f"for (uint {r} = 0; {r} < {rows}u; ++{r})"),
        state.block(f"for (uint {e} = 0; {e} < {ppw}u; ++{e})"),
    ):
        li = f"{r} * {ppw}u + {e}" if rows > 1 else e
        val = f"{src.expr}[{r}]" if kind_src == "rowvec" else src.at("0")
        state.emit(f"{dst.expr}[{li}] = {val};")


@coop_rule("transpose")
def _coop_rule_transpose(state: EmitState, eqn: JaxprEqn) -> None:
    """Only the lazy dot_general-rhs fusion exists cooperatively; a
    materialized permuted copy of a sharded value would need cross-lane
    shuffles nothing here uses. `_coop_walk` guarantees the condition."""
    src = state.val(eqn.invars[0])
    if not (
        _transpose_is_dot_rhs_only(state.consumers, eqn)
        and not src.transposed
        and _coop_kind(src) == "device"
    ):
        raise EmitError(
            "cooperative transpose: only a rank-2 device view consumed "
            "solely as dot_general rhs"
        )
    state.bind(
        eqn.outvars[0],
        dataclasses.replace(src, shape=(src.shape[1], src.shape[0]), transposed=True),
    )


COOP_RULES["reshape"] = _rule_reshape
COOP_RULES["program_id"] = _rule_program_id


def _coop_emit_reduce(
    state: EmitState,
    eqn: JaxprEqn,
    init: str,
    combine: Callable[[str, str], str],
    simd_fn: str,
) -> None:
    """Reduction over the width axis: each lane combines its own
    (validity-masked) columns into a partial, then one `simd_sum`/
    `simd_max` merges the 32 partials, so every lane receives the result,
    keeping it uniform by construction. Invalid lanes/elements contribute
    `init`, the combine identity, so no divergent branch ever surrounds
    the SIMD-group function.

    At R > 1 the (R, W) -> (R,)/(R, 1) case loops rows, one SIMD-group
    reduction per row, producing a rowvec. A full reduction to size 1
    additionally folds the row loop into the per-lane partial. The
    `simd_sum` of a shard's partials is exact per row because every lane's
    partial covers disjoint columns; cross-lane order differs from the
    sequential oracle only within the documented FAST-math tolerance.
    """
    src = state.val(eqn.invars[0])
    dst = _coop_declare(state, eqn.outvars[0])
    rows = state.coop_rows
    size = src.size
    if size == 1 or size == dst.size:
        # Identity reduction (nothing along the reduced axes): copy through.
        _coop_copy(state, dst, src)
        return
    kind = _coop_kind(src)
    if kind == "rowvec":
        if dst.size != 1:
            raise EmitError("cooperative reduce: rowvec source to non-scalar")
        # All lanes hold all rows: a lane-local fold is already the answer.
        part = state.fresh("_part")
        state.emit(f"{dst.ctype} {part} = {init};")
        r = state.fresh("_r")
        with state.block(f"for (uint {r} = 0; {r} < {rows}u; ++{r})"):
            state.emit(f"{part} = {combine(part, f'{src.expr}[{r}]')};")
        state.emit(f"{dst.at('0')} = {part};")
        return

    _, w, ppw = _coop_geom(rows, src.shape)
    ragged = ppw * _COOP_LANES != w
    per_row = dst.layout == "rowvec"

    def fold_columns(part: str, r_expr: str) -> None:
        e = state.fresh("_e")
        with state.block(f"for (uint {e} = 0; {e} < {ppw}u; ++{e})"):
            if kind == "shard":
                li = f"{r_expr} * {ppw}u + {e}" if rows > 1 else e
                val = f"{src.expr}[{li}]"
                if ragged:
                    g = _coop_global_index(state, ppw, e)
                    state.emit(
                        f"{part} = ({g} < {w}u) ? {combine(part, val)} : {part};"
                    )
                else:
                    state.emit(f"{part} = {combine(part, val)};")
            else:  # device view
                g = _coop_global_index(state, ppw, e)
                gc = f"min({g}, {w - 1}u)" if ragged else g
                gi = f"{r_expr} * {w}u + {gc}" if rows > 1 else gc
                val = f"{src.expr}[{gi}]"
                if ragged:
                    state.emit(
                        f"{part} = ({g} < {w}u) ? {combine(part, val)} : {part};"
                    )
                else:
                    state.emit(f"{part} = {combine(part, val)};")

    if per_row:
        r = state.fresh("_r")
        with state.block(f"for (uint {r} = 0; {r} < {rows}u; ++{r})"):
            part = state.fresh("_part")
            state.emit(f"{dst.ctype} {part} = {init};")
            fold_columns(part, r)
            state.emit(f"{dst.expr}[{r}] = {simd_fn}({part});")
        return

    if dst.size != 1:
        raise EmitError("cooperative reduce: unsupported output layout")
    part = state.fresh("_part")
    state.emit(f"{dst.ctype} {part} = {init};")
    r = state.fresh("_r")
    with state.block(f"for (uint {r} = 0; {r} < {rows}u; ++{r})"):
        fold_columns(part, r)
    state.emit(f"{dst.at('0')} = {simd_fn}({part});")


@coop_rule("reduce_sum")
def _coop_rule_reduce_sum(state: EmitState, eqn: JaxprEqn) -> None:
    _coop_emit_reduce(state, eqn, "0", lambda a, b: f"({a} + {b})", "simd_sum")


@coop_rule("reduce_max")
def _coop_rule_reduce_max(state: EmitState, eqn: JaxprEqn) -> None:
    _coop_emit_reduce(
        state, eqn, "-INFINITY", lambda a, b: f"fmax({a}, {b})", "simd_max"
    )


def _coop_emit_dot_mma(
    state: EmitState, outvar: Var, lhs: CVal, rhs: CVal, rows: int, k: int, n: int
) -> None:
    """`(R, k) @ (k, n)` on the SIMD-group matrix units.

    Loads 8x8 tiles of both operands straight from device memory,
    accumulates with `simdgroup_multiply_accumulate`, and stores the
    result tiles to a threadgroup scratch tile covering the full output
    width. Wide outputs are computed in column tiles of at most
    _MMA_FRAGMENT_CAP accumulator fragments each, so register pressure
    is fixed by the cap, not the width; each column tile re-runs the k
    loop (the lhs fragments are re-loaded per tile, lane-uniform reads
    the cache serves). Fragment storage is opaque (lanes cannot index
    it), so the scratch tile is what downstream rules read: the bound
    CVal is a threadgroup-space view, which consumers treat like a
    device view. Barriers bracket the store phase so one loop
    iteration's stores cannot race the previous iteration's reads. The
    full threadgroup_barrier is required even for a single-SIMD-group
    threadgroup: simdgroup_barrier does not reliably order per-lane
    threadgroup stores against cooperative simdgroup_load reads. A
    transposed rhs uses the hardware transpose load on the untransposed
    (n, k) storage.
    """
    tg = state.fresh("_mma")
    state.prologue.append(f"threadgroup float {tg}[{rows * n}];")
    state.tg_bytes += 4 * rows * n
    rt_n, ct_n, kt_n = rows // 8, n // 8, k // 8
    cw = max(1, _MMA_FRAGMENT_CAP // rt_n)
    state.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
    for ct_lo in range(0, ct_n, cw):
        cts = range(ct_lo, min(ct_lo + cw, ct_n))
        accs: dict[tuple[int, int], str] = {}
        for rt in range(rt_n):
            for ct in cts:
                c = state.fresh(f"_c{rt}_{ct}")
                state.emit(f"simdgroup_float8x8 {c} = simdgroup_float8x8(0.0f);")
                accs[(rt, ct)] = c
        a_frags = [state.fresh(f"_ma{rt}") for rt in range(rt_n)]
        b_frags = {ct: state.fresh(f"_mb{ct}") for ct in cts}
        kt = state.fresh("_kt")
        with state.block(f"for (uint {kt} = 0; {kt} < {kt_n}u; ++{kt})"):
            for rt in range(rt_n):
                state.emit(f"simdgroup_float8x8 {a_frags[rt]};")
                state.emit(
                    f"simdgroup_load({a_frags[rt]}, "
                    f"{lhs.expr} + {rt * 8 * k}u + {kt} * 8u, {k}u);"
                )
            for ct in cts:
                state.emit(f"simdgroup_float8x8 {b_frags[ct]};")
                if rhs.transposed:
                    state.emit(
                        f"simdgroup_load({b_frags[ct]}, "
                        f"{rhs.expr} + {ct * 8 * k}u + {kt} * 8u, {k}u, "
                        f"ulong2(0, 0), true);"
                    )
                else:
                    state.emit(
                        f"simdgroup_load({b_frags[ct]}, "
                        f"{rhs.expr} + {kt} * {8 * n}u + {ct * 8}u, {n}u);"
                    )
            for rt in range(rt_n):
                for ct in cts:
                    c = accs[(rt, ct)]
                    state.emit(
                        f"simdgroup_multiply_accumulate("
                        f"{c}, {a_frags[rt]}, {b_frags[ct]}, {c});"
                    )
        for rt in range(rt_n):
            for ct in cts:
                state.emit(
                    f"simdgroup_store({accs[(rt, ct)]}, "
                    f"{tg} + {rt * 8 * n + ct * 8}u, {n}u);"
                )
    state.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
    state.bind(
        outvar,
        CVal(expr=tg, shape=(rows, n), ctype="float", space="threadgroup"),
    )


@coop_rule("dot_general")
def _coop_rule_dot_general(state: EmitState, eqn: JaxprEqn) -> None:
    """`(R, k) @ (k, n)` across the SIMD-group: the output is sharded
    over lanes (lane t owns output columns [t*ppn, (t+1)*ppn) of every
    row), the rhs is a device view (possibly lazily transposed), and the
    lhs is either

    - a device view (the QK^T case: q is a get-aliased block): k-outer,
      row/column-inner float4 loops so each rhs element loaded serves all
      R rows (the traffic saving that makes query blocking pay) while
      lhs loads at a fixed (r, kk) are lane-uniform (hardware broadcast);
    - sharded (p computed cooperatively, columns per lane): one
      `simd_broadcast` per (row, k)-step hands every lane the owner
      lane's element, uniformly.

    All row/column loops are python-unrolled: accumulators are
    individually-named registers, never dynamically-indexed thread arrays
    (dynamically indexed thread arrays can spill to stack).
    Out-of-range lanes compute a clamped (in-bounds, discarded) column so
    control flow stays uniform through the `simd_broadcast`.
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
    if len(rhs.shape) == 1:
        # Matvec: a rank-1 rhs acts as (k, 1) over the same flat storage,
        # same canonicalization as the classic rule. Rank-1 lhs never
        # reaches this rule (the detection walker keeps it classic).
        rhs = dataclasses.replace(rhs, shape=(rhs.shape[0], 1))
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise EmitError("dot_general: only rank-2 operands are implemented")
    m, k = lhs.shape
    k2, n = rhs.shape
    if k != k2:
        raise EmitError(f"dot_general: inner dims disagree ({k} vs {k2})")
    rows = state.coop_rows
    if m != rows:
        raise EmitError(f"cooperative dot_general: m={m} != R={rows}")
    if _coop_kind(rhs) != "device":
        raise EmitError("cooperative dot_general: rhs must be a device view")

    if (
        _COOP_MMA_DOT
        and _coop_kind(lhs) == "device"
        and lhs.space in ("device", "threadgroup")
        and rhs.space == "device"
        and not lhs.transposed
        and lhs.ctype == "float"
        and rhs.ctype == "float"
        and CTYPES[str(_shaped(eqn.outvars[0].aval).dtype)] == "float"
        and rows % 8 == 0
        and k % 8 == 0
        and n % 8 == 0
        and rows * n * 4 <= _MMA_TG_TILE_LIMIT
    ):
        _coop_emit_dot_mma(state, eqn.outvars[0], lhs, rhs, rows, k, n)
        return

    dst = _coop_declare(state, eqn.outvars[0])
    out_kind, _, ppn = _coop_geom(rows, dst.shape)
    sharded_out = out_kind == "shard"
    ragged = sharded_out and ppn * _COOP_LANES != n

    def rhs_at(kk: str, j: str) -> str:
        if rhs.transposed:
            return f"{rhs.expr}[{j} * {k}u + {kk}]"
        return f"{rhs.expr}[{kk} * {n}u + {j}]"

    def dst_index(r: int, e: int) -> str:
        if out_kind == "shard":
            return str(r * ppn + e) if rows > 1 else str(e)
        if out_kind == "rowvec":
            return str(r)
        return "0"

    # Lane-owned output columns, one C variable per local column index
    # (r/e loops are python-unrolled throughout this rule: accumulators
    # become individually-named registers, never dynamically-indexed
    # arrays, which can spill to stack).
    def emit_jcs() -> list[str]:
        jcs: list[str] = []
        for e in range(ppn):
            jc = state.fresh("_jc")
            if sharded_out:
                gexpr = f"((uint)_lane * {ppn}u + {e}u)"
                state.emit(
                    f"uint {jc} = {f'min({gexpr}, {n - 1}u)' if ragged else gexpr};"
                )
            else:
                state.emit(f"uint {jc} = 0u;")
            jcs.append(jc)
        return jcs

    lkind = _coop_kind(lhs)
    if lkind == "shard":
        jcs = emit_jcs()
        _, _, ppk = _coop_geom(rows, lhs.shape)
        accs: list[str] = []
        for r in range(rows):
            acc = state.fresh(f"_acc{r}")
            for e in range(ppn):
                acc_e = f"{acc}_{e}" if ppn > 1 else acc
                state.emit(f"{dst.ctype} {acc_e} = 0;")
                accs.append(acc_e)
        kk = state.fresh("_ki")
        with state.block(f"for (uint {kk} = 0; {kk} < {k}u; ++{kk})"):
            for r in range(rows):
                pk = state.fresh(f"_pk{r}")
                lhs_elem = (
                    f"{lhs.expr}[{r * ppk}u + {kk} % {ppk}u]"
                    if ppk > 1
                    else f"{lhs.expr}[{r}]"
                )
                owner = f"(ushort)({kk} / {ppk}u)" if ppk > 1 else f"(ushort)({kk})"
                state.emit(f"{lhs.ctype} {pk} = simd_broadcast({lhs_elem}, {owner});")
                for e in range(ppn):
                    state.emit(f"{accs[r * ppn + e]} += {pk} * {rhs_at(kk, jcs[e])};")
        for r in range(rows):
            for e in range(ppn):
                state.emit(f"{dst.expr}[{dst_index(r, e)}] = {accs[r * ppn + e]};")
        return

    # lhs is full-access: a device view, or a rep value (k == 1 edge case).
    lhs_is_array = lkind == "device" or lhs.size > 1

    def lhs_at(r: int, kk: str) -> str:
        if not lhs_is_array:
            return lhs.at("0")
        return f"{lhs.expr}[{r * k}u + {kk}]" if rows > 1 else f"{lhs.expr}[{kk}]"

    vectorized = (
        dst.ctype == lhs.ctype == rhs.ctype == "float"
        and k % 4 == 0
        and lkind == "device"
        and lhs.align % 4 == 0
        and rhs.align % 4 == 0
        and rhs.transposed
    )
    jcs = emit_jcs()
    if vectorized:
        # k-outer / row-and-column-inner: each rhs float4 is loaded once
        # per lane and reused across all R rows; the lhs loads at a fixed
        # (r, kk4) are the same address on every lane (uniform), which the
        # hardware serves as a broadcast rather than a gather.
        k4 = k // 4
        brows: list[str] = []
        for e in range(ppn):
            brow = state.fresh(f"_brow{e}")
            state.emit(
                f"device const float4* {brow} = "
                f"(device const float4*)({rhs.expr}) + {jcs[e]} * {k4}u;"
            )
            brows.append(brow)
        arow = state.fresh("_arow")
        state.emit(
            f"{lhs.space} const float4* {arow} = "
            f"({lhs.space} const float4*)({lhs.expr});"
        )
        accs = []
        for r in range(rows):
            for e in range(ppn):
                acc = state.fresh(f"_acc{r}_{e}")
                state.emit(f"float4 {acc} = float4(0.0f);")
                accs.append(acc)
        kk = state.fresh("_k4")
        with state.block(f"for (uint {kk} = 0; {kk} < {k4}u; ++{kk})"):
            bvals = []
            for e in range(ppn):
                b = state.fresh(f"_b{e}")
                state.emit(f"float4 {b} = {brows[e]}[{kk}];")
                bvals.append(b)
            for r in range(rows):
                a = state.fresh(f"_a{r}")
                state.emit(f"float4 {a} = {arow}[{r * k4}u + {kk}];")
                for e in range(ppn):
                    state.emit(f"{accs[r * ppn + e]} += {a} * {bvals[e]};")
        for r in range(rows):
            for e in range(ppn):
                acc = accs[r * ppn + e]
                state.emit(
                    f"{dst.expr}[{dst_index(r, e)}] = "
                    f"{acc}.x + {acc}.y + {acc}.z + {acc}.w;"
                )
        return

    accs = []
    for r in range(rows):
        for e in range(ppn):
            acc = state.fresh(f"_acc{r}_{e}")
            state.emit(f"{dst.ctype} {acc} = 0;")
            accs.append(acc)
    kk = state.fresh("_ki")
    with state.block(f"for (uint {kk} = 0; {kk} < {k}u; ++{kk})"):
        for r in range(rows):
            a = state.fresh(f"_a{r}")
            state.emit(f"{dst.ctype} {a} = {lhs_at(r, kk)};")
            for e in range(ppn):
                state.emit(f"{accs[r * ppn + e]} += {a} * {rhs_at(kk, jcs[e])};")
    for r in range(rows):
        for e in range(ppn):
            state.emit(f"{dst.expr}[{dst_index(r, e)}] = {accs[r * ppn + e]};")


@coop_rule("scan")
def _coop_rule_scan(state: EmitState, eqn: JaxprEqn) -> None:
    """`lax.fori_loop` cooperatively: identical structure to `_rule_scan`
    (uniform trip count, two-phase carry copy-back), with layout-aware
    carry storage and copies. The loop itself is uniform across lanes, so
    SIMD-group functions inside the body remain valid."""
    length: int = eqn.params["length"]
    body: Jaxpr = eqn.params["jaxpr"]
    num_carry = len(eqn.outvars)
    num_consts = len(eqn.invars) - num_carry

    _check_pure_carry_scan(eqn, body)
    if eqn.params.get("reverse", False):
        raise EmitError("no reverse-mode scan support, use lax.fori_loop")

    const_vals = [state.val(v) for v in eqn.invars[:num_consts]]

    carries = []
    for invar, outvar in zip(eqn.invars[num_consts:], eqn.outvars, strict=True):
        dst = _coop_declare(state, outvar)
        _coop_copy(state, dst, state.val(invar))
        carries.append(dst)

    carry_exprs = {c.expr for c in carries}
    idx = state.fresh("_s")
    with state.block(f"for (uint {idx} = 0; {idx} < {length}; ++{idx})"):
        outs = emit_jaxpr_coop(state, body, const_vals + carries)
        sources = []
        for out, carry in zip(outs, carries, strict=True):
            if out.expr == carry.expr:
                sources.append(None)
            elif out.expr in carry_exprs:
                tmp = CVal(
                    expr=state.fresh(),
                    shape=out.shape,
                    ctype=out.ctype,
                    layout=carry.layout,
                )
                count = _coop_storage_len(state, tmp)
                state.emit(
                    f"{tmp.ctype} {tmp.expr}[{count}];"
                    if tmp.shape
                    else f"{tmp.ctype} {tmp.expr};"
                )
                _coop_copy(state, tmp, out)
                sources.append(tmp)
            else:
                sources.append(out)

        for src, carry in zip(sources, carries, strict=True):
            if src is not None:
                _coop_copy(state, carry, src)
