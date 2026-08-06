# `simdgroup_matrix` matmul: what it would take

Scoping note for ROADMAP stretch 12 (tiled/MMA `dot_general`). Not a plan
to implement yet — written up because the change touches more of the
emitter than a single rule, and that needs to be visible before anyone
starts.

## Context

`_rule_dot_general` (`src/palladium/emit.py`) is currently a naive
O(M·N·K) scalar loop, `i, j` outer with `k` innermost accumulating into a
register-resident scalar. A loop reorder (`i, k, j`, chasing unit-stride
`rhs`/`dst` access) was tried and measured ~4x *slower* at M=4096 on a
real generated kernel — reordering traded a strided-read cost for `k`
read-modify-writes of the `dst` array per `j`, more memory traffic than
it saved, worse as thread count grows. See that rule's docstring for the
full finding. The naive version is what's shipped.

Two options exist for something faster than the scalar loop:

- **MPSMatrixMultiplication** (Metal Performance Shaders): a pre-built,
  Apple-tuned matmul, called as a separate GPU dispatch. Rejected for
  this codebase's purposes: it can't be inlined into a fused kernel, so
  a `dot_general` inside e.g. the flash-attention example's online-softmax
  loop would have to become a second dispatch, losing the fusion that's
  the point of hand-writing the kernel at all.
- **`simdgroup_matrix`** (Metal 3 hardware matrix intrinsics): stays
  inside a single emitted kernel function, so fusion is preserved. This
  doc scopes that path.

## Why this isn't just a smarter `_rule_dot_general`

Every existing rule assumes one Metal thread is one independent Pallas
program instance: `thread_position_in_grid` maps straight to
`program_id`, and no rule ever reads another thread's data. Elementwise,
reduce, broadcast — all trivially parallel, zero inter-thread
coordination.

`simdgroup_float8x8` and `simdgroup_load`/`simdgroup_multiply_accumulate`/
`simdgroup_store` break that invariant at the root: they're cooperative
operations across a whole 32-thread SIMD-group. All 32 threads must call
them together, in uniform control flow — a tile of the matmul is computed
*jointly* by 32 threads, and none of them can simultaneously be running
an unrelated, independent program instance. A kernel that contains a
`dot_general` can't keep "one thread = one query row" (the flash-attention
example's model) for the threads participating in that op. This is an
execution-model fork, not a codegen-only change, and it's why it's scoped
separately from the ordinary rule-by-rule roadmap items.

## What has to change, piece by piece

**1. Address space and staging machinery `emit.py` doesn't have yet.**
`simdgroup_load`/`store` read/write a contiguous 2D region via
`(ptr, elements_per_row, matrix_origin, transpose)`. Pallas block refs
are already contiguous 2D blocks, so that part fits. But efficient
staging goes through `threadgroup` memory (shared, fast, needs
`threadgroup_barrier(mem_flags::mem_threadgroup)` between load, compute,
and store phases). Nothing today declares `threadgroup`-space storage or
emits a barrier — every array `EmitState.declare()` emits is implicitly
`thread`-local. This is a second address-space concept plus a
synchronization primitive, not just a new `RuleFn`.

**2. Tile-granularity shape handling.**
`simdgroup_float8x8` operates on 8×8 tiles. The current rule handles
arbitrary `(m, k, n)` exactly via scalar loops, no padding needed. A
tiled version has to decide what happens when `m`, `k`, or `n` isn't a
multiple of 8 — pad-and-mask the tile, or fall back to the scalar path
for the remainder. No other rule in the emitter needs shape handling
beyond "loop over the exact declared shape," so this is new territory.

**3. Launch configuration in `dispatch.py`.**
`bind()` currently launches with grid dims mapping directly to thread
count (one thread per grid cell). SIMD-group work wants grid dims to
index *tiles*, with `threads_per_threadgroup` fixed to a SIMD-group
multiple (32, or 32×N) so one SIMD-group owns one tile. That's a change
to how `KernelSpec.grid` becomes a dispatch call — separate from, and
upstream of, anything `emit.py` does.

**4. Which kernels opt in.**
Given (1)-(3), this can't be the default execution model — every
existing kernel (RK4/SDE integrators, elementwise-only kernels) is
correctly served by one-thread-per-instance and shouldn't pay tile/
threadgroup/barrier overhead it doesn't need. Realistically: a second
execution model, alongside the current one, entered only by kernels
whose jaxpr contains a `dot_general` above some size threshold worth
tiling for. The `KernelSpec` -> MSL pipeline needs a fork point where
that decision gets made, before any of (1)-(3) run.

## Suggested sequencing

1. Isolated hand-written MSL first (same methodology as the loop-reorder
   scoping pass): confirm `simdgroup_matrix` actually beats even the
   naive scalar loop at the sizes this project's kernels use (head_dim=64
   attention tiles, not the large square matmuls MPS/simdgroup tiling is
   usually justified by) before touching the emitter at all. The loop-
   reorder finding above is a specific warning that "textbook GPU matmul
   optimization" doesn't automatically transfer to this problem size and
   this backend. **Done, see below.**
2. If that holds up, design the threadgroup-memory/barrier primitives in
   `EmitState` as their own piece, independent of `dot_general` — they're
   reusable machinery, not `dot_general`-specific.
3. Design the opt-in fork (item 4 above) before writing the tiled rule
   itself, so the naive path stays untouched for every kernel that
   doesn't need this.

## Step 1 results: isolated `simdgroup_matrix` vs. naive scalar vs. JAX-CPU

Measured on an M1 Pro, hand-written MSL dispatched directly through
`metal_runtime` (bypassing the emitter, same methodology as the
loop-reorder scoping pass). Three kernels, all verified against NumPy
(`err=0.0` after fixing a `grid`-vs-`threadgroup` mixup in the tiled
kernel's first pass — `metal_runtime`'s `grid` argument is thread count,
not threadgroup count):

- **scalar**: one thread, the naive triple-nested loop, for an
  apples-to-apples floor.
- **simdgroup (1 simdgroup)**: single 32-thread SIMD-group, no grid
  parallelism, every 8×8 tile done in sequence.
- **simdgroup (tile grid)**: one threadgroup/SIMD-group per output tile,
  parallel across the `(M/8, N/8)` tile grid.

First pass also under-measured JAX: `bench_jax` called
`block_until_ready()` once after a loop of `n_iter` dispatches, not once
per call, which lets JAX pipeline dispatch of call `i+1` behind
execution of call `i` — an async-throughput number, not the
synchronous per-call latency `metal_runtime`'s `run()` measures (it
blocks until completion every call, per its own docstring). Comparing
pipelined-JAX against blocking-Metal flattered JAX at exactly the small
sizes where dispatch overhead dominates. Corrected by blocking after
every JAX call too, so both sides measure the same thing: one dispatch,
block, get the result.

**A second, larger measurement problem surfaced later, while debugging
why the standalone `simdgroup_matmul()` module (built after this first
pass) ran slower than these numbers implied it should.** Holding the
kernel, grid, and threadgroup config fixed and varying only how much
prior GPU work preceded the timed loop:

| warm-up calls before timing | dispatch time (512×512×512 tile-grid) |
|---|---|
| 1   | 0.862 ms |
| 20  | 0.773 ms |
| 100 | 0.748 ms |
| 300 | 0.523 ms |
| 600 | 0.837 ms |

Same code, same process, zero changes — a ~65% swing from GPU clock
ramp-up under sustained load, then apparent thermal throttling back down
past 300 calls (M1 Pro, fanless-adjacent chassis). The first-pass numbers
below were captured with only light warm-up per size, immediately after
several much more expensive scalar/single-simdgroup runs at smaller
sizes — meaning the `512` row landed in a favorably-ramped clock state
by accident, not by anything reproducible about the kernel itself.

**Corrected methodology:** a fixed *wall-clock* warm-up soak (1s of
back-to-back dispatches, not a fixed call count) before every timed
block, median of 7 such blocks per measurement. Same treatment for both
Metal and JAX so the comparison stays fair.

| M=K=N | scalar (1 thread) | simdgroup (1 simdgroup) | simdgroup (tile grid, soaked median) | `jax.jit` (CPU, soaked median) | tile-grid vs JAX |
|---|---|---|---|---|---|
| 64  | 5.51 ms   | 0.34 ms  | 0.197 ms | 0.030 ms | 0.15x |
| 128 | 49.96 ms  | 0.57 ms  | 0.227 ms | 0.078 ms | 0.34x |
| 256 | 385.90 ms | 3.04 ms  | 0.277 ms | 0.151 ms | 0.55x |
| 512 | 2773.75 ms| 25.28 ms | 0.524 ms | 0.508 ms | **0.97x** |

(Scalar and single-simdgroup columns are the unsoaked first-pass numbers,
kept for the naive-loop comparison; they're each one-off single-thread
or single-simdgroup runs orders of magnitude apart, where a 65% clock-
state swing doesn't change the qualitative conclusion. Only the tile-grid
vs. JAX comparison was close enough for the swing to matter, so only
that row got the soaked re-measurement.)

Empty-kernel dispatch floor: ~0.20ms (consistent across both passes).

**Reading it:**

- `simdgroup_matrix` beats the scalar loop by two to four orders of
  magnitude at every size tested — expected, since scalar is one thread
  with zero parallelism. That conclusion doesn't change.
- Against JAX-CPU, the earlier **"wins at 512" claim does not hold up.**
  With both sides measured fairly and both given time to reach a stable
  clock state, 512×512×512 is **parity** (0.97x), not a win — the
  original 1.61x came from measuring Metal mid-ramp in a lucky window,
  not from anything real about the kernel being faster than JAX there.
- The rest of the shape holds: tile-grid is still dispatch-floor-bound
  at 64/128 (0.20–0.23ms sits right on the ~0.20ms empty-kernel floor),
  and still closes the gap steadily as size grows (0.15x → 0.34x →
  0.55x → 0.97x). Just shifted down by roughly the same clock-state
  correction at every size — the *trend* was real, the *512 crossover*
  wasn't.
- The sizes this project's kernels actually use — `head_dim=64`,
  `block_kv` up to 128 — are still exactly the two rows dominated by
  `metal_runtime`'s per-call dispatch overhead rather than matmul work,
  for the same reason as before: that floor is a property of this
  benchmark's one-matmul-per-Python-call structure, not necessarily of a
  `dot_general` inlined inside an already-dispatched fused kernel. Step
  1 still can't distinguish those two cases — only inlining (steps 2-3)
  settles it.

**Verdict, corrected:** proceed to step 2, but on weaker footing than
first reported. `simdgroup_matrix` is not a dead end the way the naive
loop reorder was — the trend toward parity as size grows is real and
reproducible — but there is currently no size, tested here, where it
clearly beats JAX-CPU on a fair, clock-stable measurement. "Promising,
approaches parity at 512, unproven to actually win anywhere yet" is the
honest summary, not "wins at 512."

**Methodology note for future benchmarks in this project:** GPU
dispatch timing on this hardware is not stable call-to-call; a fixed
wall-clock warm-up soak (not a fixed call count) plus a median over
several post-soak trial blocks is needed before trusting any Metal-vs-
CPU comparison closer than roughly a 2x margin. Anything reported from a
single trial, or from a script that reuses GPU-clock state left over
from unrelated prior measurements, should be treated as provisional
until re-measured this way.

## Steps 2-4 outcome (2026-08-13): built, but not the way this doc scoped it

The execution-model fork landed in `emit.py` (see the "simdgroup-
cooperative execution model" section there) and took flash attention
(`examples/06`) from 0.06x to 0.88-0.92x of `jax.jit` CPU on a rested
M1 Pro (`docs/reward-spec-matmul-emitter.md` has the full calibration
history). Three of this doc's four "what has to change" items shipped,
with one deliberate substitution:

- **Item 1 (threadgroup memory + barriers): built differently.** The
  shapes real kernels stage ((1,16)/(1,64) values, scalars) never need
  staging; SIMD-group *functions* (`simd_broadcast`/`simd_sum`/
  `simd_max`/`simd_shuffle_xor`) cover all cross-lane traffic with
  uniform control flow and zero shared memory. Threadgroup-memory
  staging of the K block *was* implemented and measured — 0.43-0.79x
  interleaved, occupancy loss from the per-threadgroup allocation beat
  the coalescing win — and reverted (dead-ends note in `emit.py`).
- **Item 2 (tile-granularity shapes): moot** at these sizes — no 8x8
  tiles involved; the cooperative dot shards output columns over lanes
  with clamped-index masking, so any (1,k)@(k,n) works.
- **Item 3 (launch config): as scoped.** One 32-thread threadgroup per
  program instance (`dispatch.BoundKernel.cooperative`), instance =
  `threadgroup_position_in_grid.x`, lane = `thread_index_in_simdgroup`.
- **Item 4 (opt-in fork): as scoped.** `emit.is_simdgroup_cooperative`
  admits a kernel only when it stages an m==1 `dot_general` on a
  device-view rhs and every primitive has a cooperative lowering;
  everything else keeps the one-thread-per-instance model untouched.

`simdgroup_matrix` itself — this doc's original subject — is still only
in the standalone `palladium.simdgroup_matmul`: with m==1 per program
instance, 8x8 MMA tiles would waste 7/8 of every tile, and the
cooperative row-dot + lane-sharding model won instead. The intrinsics
become relevant again only if a kernel with m >= 8 per-instance matmuls
enters the benchmark set (ROADMAP stretch 12's own gating condition).

**Update (2026-08-14):** the m == 1 restriction noted above is gone — the
cooperative model now lowers `m == R` dot_generals (columns-per-lane
layout, `docs/query-blocking-scratch.md` has the motivating measurement,
reward spec v0.2.0 the resulting calibration: saturated at 1.42x over
jax.jit with block_q=8). Stretch 12's gating condition — an MMA-friendly
kernel in the benchmark set — is now genuinely met: the blocked kernel's
(8, 64) @ (64, 32) QK^T is exactly a simdgroup_matrix shape, so the MMA
intrinsics are the next untapped win if more speed is ever needed here.
