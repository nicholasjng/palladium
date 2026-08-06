# Emitter: noted simplifications and deferred work

Running list of improvements identified during the 2026-08-14 audit and
the control-flow session, not yet done because each needs either a
demanding kernel or a measurement to justify it. Drive-bys already
applied are listed at the bottom for the record.

## Correctness-adjacent

- **`dot_general` ignores `preferred_element_type`.** Accumulation dtype
  comes from the output aval. Accidentally right for f16-in/f32-out on
  the scalar path (the accumulator is the output ctype), but untested,
  and every vectorized/MMA path additionally gates on f32. Honor the
  parameter explicitly before any mixed-precision kernel lands.
- **Rank-1 `dot_general` (matvec) is rejected**, because `jnp.dot(v, M)`
  stages a contraction on lhs dim 0. Canonicalize by treating rank-1 lhs
  as `(1, k)` (and rank-1 rhs as `(k, 1)`) at the top of both dot rules;
  small, mechanical, needs a test.

## Observability

- **Cooperative fallback is a silent ~5-10x cliff.** One ineligible dot
  (computed rhs, m not matching R, R not dividing 32) drops the whole
  kernel to the classic model with no signal beyond
  `BoundKernel.cooperative`. Make `coop_rows`/`_coop_walk` return or log
  the first rejection reason, and surface it via `debug_msl` or a
  `PALLADIUM_EXPLAIN=1` env hook.

## Cooperative-model gaps (each needs a motivating kernel)

- **`cond`/`while` cooperatively**: safe when the predicate is uniform
  (layout "rep"; every lane computes it identically), which the
  detection walker can check. A uniform `cond` would unblock the
  skip-rescale flash-attention optimization. Divergent predicates stay
  classic-only by construction.
- **`jit` inlining and the Threefry rules are classic-only.** The shared
  walker makes porting mechanical (`_inline_jit` needs the active rule
  table; Threefry values are per-instance uints, layout "rep"). Matters
  only when an RNG-plus-dot kernel appears.
- **Batched `dot_general`** (vmap-in-kernel, multi-head per instance):
  a real layout decision (batch dim across lanes? across simdgroups?),
  not a rule tweak. Design before implementing.
- **The G=1 unification** (scalar model as the 1-lane-group point of the
  cooperative taxonomy): degenerates exactly, provable via byte-identical
  golden snapshots, but melts the teachable classic rules into
  layout-parametrized code for ~300-400 saved lines. Rule of three: do it
  when a third execution pattern lands. The first real candidate is
  instances-as-lanes with cross-instance ensemble reductions
  (simd_sum of per-trajectory statistics), which neither current model
  expresses.

## Small polish

- **The MMA fragment cap is an inline literal**: done 2026-08-18, now
  `_MMA_FRAGMENT_CAP = 8`, and it bounds fragments per column tile
  rather than gating eligibility (wide outputs column-tile; see
  coop-colvec-and-transposed-lhs.md). Historical measurement that set
  it: raising to 16 let block_q=16's PV take the MMA path, 6.79ms ->
  3.98ms at seq 4096, and it still lost to block_q=8 at 0.87x.
- **`emit_jcs`/`dst_index`/`rhs_at` closures in the coop dot** are
  re-derived per call; harmless, but the dot rule is the file's longest
  function and would read better with a small emission-context object.
- **Alignment follow-ups now possible** (`CVal.align` is a gcd, not a
  bool): float2 paths for 2-but-not-4-aligned views, alignment
  assertions in the MMA loads, and 8/16-element checks if wider vector
  types ever matter.

## Metal compiler bug: carry permutations in loops (2026-08-18)

The Metal shader compiler miscompiles loop bodies that hold three or
more thread-local array temporaries next to a carry swap: the permuted
carry's read is forwarded across the write it must precede, so a
swap-plus-computed-carry scan returns wrong values from iteration 2 on
(both SAFE and FAST math, M1 Pro, macOS 25.6). Reproduced in
hand-written MSL, independent of palladium's codegen. Snapshot arrays,
fused copy-back loops, and hoisted declarations all still miscompile;
fully scalarizing the chain avoids it (no array temporaries left), and
volatile reads/writes at the two hazard endpoints avoid it at zero cost
to the permutation-free path. `_copy_back_carries` now emits the
volatile form for permuted carries; regression tests:
`test_carry_permutation_beside_computed_carry` and the formerly
strict-xfail `test_carry_read_and_alias_conflict`, which this fix
turned green (same root cause, found by the fuzzer months apart).
Watch item: the cooperative scan's snapshot copies (`_coop_copy` temps)
have the same structural shape; no failure demonstrated there yet, and
the scored kernels' oracle gates would catch one, but any future coop
carry-permutation kernel should get an oracle test first.

## Measured and rejected (do not re-try without new evidence)

- float4-vectorizing `state.copy`: 1.023x on a copy-bound synthetic; the
  Metal compiler already handles scalar copy loops (2026-08-14).
- Per-shape `block_q`, fragment cap 16, and the removed k-split /
  Case B vector paths: see the reward spec's v0.3.0 notes.

## Drive-bys applied (2026-08-17, FFI session)

- The FFI path now dispatches cooperative kernels with their required
  geometry: `palladium_ffi.cpp` gained `threadgroup_x/y/z` attrs
  (0 = runtime default, so classic kernels are unchanged), and `ffi.py`
  selects the model per cached entry instead of pinning classic.
  Measured end to end inside `jax.jit` with a `jnp` epilogue vs pure
  JAX-CPU: 2.39x at seq 1024, 1.98x at 4096. A first cross-run reading
  suggested the FFI path trailed the direct `MetalCallable` path ~2x at
  large seq; paired-interleaved re-measurement showed that was thermal
  contamination. Actual FFI overhead: ~0.25ms per call total (zero-copy
  page wrapping works; passthrough probes read 0.19ms at 16KB buffers,
  0.25ms at 1MB), and FFI attention at seq 4096 runs within 3% of
  direct-pinned (1.029x paired). The overhead only matters for kernels
  in the sub-millisecond range, where it is the known dispatch floor,
  not a buffer-path deficiency.

## Drive-bys applied (2026-08-14, control-flow session)

- `vec4_ok: bool` replaced by `align: int` (guaranteed element multiple,
  gcd-composed through views, 0 = exactly aligned). Scored kernels'
  emitted MSL verified byte-identical before/after.
- `cond` (with `lax.switch` clamping), `while`, `and`/`xor`/`not`, and
  `clamp` added to the classic model with oracle-backed tests
  (`tests/test_17_control_flow.py`). The adaptive-ODE workarounds
  (select-instead-of-branch, fixed-budget self-stalling loops, avoiding
  `&&`) are no longer forced by the emitter.
- Scan's two-phase carry copy-back factored into `_copy_back_carries`,
  shared with the `while` lowering.
