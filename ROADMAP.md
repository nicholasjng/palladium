# palladium roadmap: Pallas kernels on Apple GPU

Goal: show that hand-written kernels on Apple GPU beat Diffrax on
ODE-integration-shaped problems, authored in Pallas, executed through
`metal-runtime`. This repo owns everything upstream of "compile this MSL
text": tracing, codegen, dispatch glue, benchmarks. The runtime contract
(`Kernel`/`Buffer`/`run`/`Batch`) lives in the sibling `metal-runtime` repo.

Ecosystem survey, why there is no off-the-shelf path (researched
2026-08-09/10, verified against each project's current docs/source rather
than a saved note): `jax-metal` is Apple's official PJRT plugin and is
StableHLO-bound, explicitly experimental, with no Pallas/Mosaic lowering
for Metal. Triton has no Metal backend. Mosaic GPU (Pallas's CUDA backend)
is structurally NVIDIA: warpgroups, TMA, mbarrier, clusters are
Hopper/Blackwell hardware facts, not portable model choices. `jax-mps`
(github.com/tillahoffmann/jax-mps) is a real, working community PJRT
plugin that parses StableHLO and maps ops onto MLX, but its own docs
explicitly exclude "kernel-authoring suites that target CUDA/TPU": Pallas
doesn't decompose into generic StableHLO ops for a plugin to interpret,
so no StableHLO→MLX mapper gives you Pallas-on-Metal for free. That gap
is what this repo fills directly, by interpreting the Pallas kernel
jaxpr itself instead of waiting for an upstream backend.

Design docs referenced throughout this file, kept separate because they're
either scoping notes for work not yet started or too detailed to inline:
`docs/simdgroup-matmul-design.md` (the cooperative SIMD-group execution
model's design history and measurements) and
`docs/whole-jaxpr-lowering-plan.md` (the long-horizon plan for accepting
plain `jax.jit` jaxprs, not just Pallas kernels — see Production readiness
below).

## The pipeline

    Pallas kernel --trace--> KernelSpec --emit--> MSL text --bind--> callable
                  (trace.py)           (emit/)             (dispatch.py)

`metal_call(...)` composes the three behind a `pl.pallas_call`-shaped API,
dispatching eagerly through metal-runtime's Python bindings. `metal_call_jit`
(`ffi.py`) instead registers the pipeline as an `jax.ffi` target on
`platform="cpu"`, so the result composes inside `jax.jit` next to ordinary
`jnp` ops — this is the production entry point; `metal_call` is the
eager/debugging one. `.interpret` on either result is the CPU oracle
(`interpret=True`), which every test diffs against: the single biggest
de-risking trick in the project.

Emission model (`emit/core.py`, `emit/rules.py`): one Metal thread per
Pallas program instance (or one simdgroup, for cooperative/simdgroup-matmul
kernels via `emit/coop.py`); every jaxpr var becomes a thread-local array;
every rule emits plain element loops.

## Status: core pipeline done

Trace → emit → dispatch, both the eager (`metal_call`) and `jax.jit`-composable
(`metal_call_jit`, via a generic XLA FFI handler in `native/ffi/palladium_ffi.cpp`)
entry points, packaging (dylib ships inside the wheel via scikit-build-core,
verified against a clean venv install), and the primitive/feature set below
are all built and covered by regression tests. The exercise-by-exercise
build log (test-driven, one red test per emitter rule) that got here is in
git history, not reproduced here; `git log --oneline` and `tests/test_0*.py`
through `test_17_*.py` are the record of how each piece landed.

**Primitive coverage** (`emit/rules.py`): block load/store (`get`/`swap`),
elementwise ops, `program_id`/grid offsets, `fori_loop`/pure-carry `scan`,
`while_loop`, `cond`, indexed ref access (static and loop-var, unit-stride
only), in-kernel Threefry RNG (`random_bits`/`random_wrap`/`random_unwrap`/
`random_fold_in`, matching `jax.random` bit-for-bit), `dot_general` (rank-2,
no batch dims; scalar loop by default, or a cooperative simdgroup-lowered
path — see below), `reduce_sum`/`reduce_max` (any subset of axes),
`transpose` (general n-dim permutation), `reshape`, `broadcast_in_dim`.
Known, accepted restrictions: f32-first (Metal has no f64), contiguous
blocks only (partial leading dim), pure-carry scans, no strided ref slices.

**Cooperative (simdgroup) execution model** (`emit/coop.py`, design history
in `docs/simdgroup-matmul-design.md`): a second, opt-in lowering — one
32-thread SIMD-group per program instance instead of one thread — entered
only when `emit.is_simdgroup_cooperative` detects a kernel stages a
`dot_general` with a shape it can lower cooperatively (`m == R`, columns-
per-lane layout; `dot_general`'s naive scalar path remains the default
otherwise). Cross-lane traffic goes through SIMD-group functions
(`simd_broadcast`/`simd_sum`/`simd_max`/`simd_shuffle_xor`), not
threadgroup memory: a threadgroup-memory staging variant was built and
measured slower (0.43-0.79x, occupancy loss from per-threadgroup
allocation) and reverted. Raw `simdgroup_matrix` 8x8 MMA intrinsics exist
standalone (`palladium.simdgroup_matmul`) but aren't wired into the
emitter yet — not needed at the shapes measured so far (see below); the
next trigger is an `m >= 8`-per-instance matmul entering the benchmark set.

**Status, corrected 2026-08-18** (the flash-attention number below was
stale): the cooperative model took `examples/06_flash_attention.py` from
0.06x to a calibrated **1.42x over `jax.jit` CPU** (block_q=8, reward spec
v0.2.0). The naive scalar `dot_general` path is still what a kernel gets
by default outside the shapes the cooperative detector admits.

**Precision**: `MathMode` (FAST default, SAFE for compensated arithmetic)
travels end to end, including through the FFI path (`math_mode` is an FFI
`Attr`, verified with a sabotage test that the mode actually reaches
compilation). A hand-written df32 prelude (`examples/05_df32_precision.py`)
gets ~6x closer to float64 than plain float32 SAFE, at 5-8x wall-clock cost;
FAST and SAFE track each other within 6%, so reassociation alone isn't the
precision bottleneck — the float32 mantissa is.

**Benchmarks** (`examples/`): 100k-member Lotka-Volterra RK4 ensemble
(36x over `jit(vmap(diffeqsolve))` on CPU); adaptive-stepping lockstep-tax
measurement (BS3(2) + FSAL + PI controller, per-thread adaptivity removes
the tax at large N); 1M-path GBM Monte Carlo with in-kernel RNG, validated
against Black-Scholes; Gray-Scott reaction-diffusion stencil (halo reads via
indexed refs); single-head flash-attention-style kernel (online softmax,
streaming K/V), which started at ~15-20x slower than `jax.jit` CPU on the
naive scalar `dot_general` path and now runs 1.42x *faster* than it via
the cooperative simdgroup lowering described above.

Honest framing, stated wherever numbers are shown: on macOS the practical
Diffrax baseline is JAX's **CPU** backend (jax-metal cannot run Diffrax;
jax-mps coverage of its while_loop internals is untested).

### Scoping notes: what would actually improve dot_general performance

Investigated 2026-08-13 via hand-written MSL dispatched directly through
`metal_runtime` (bypassing the emitter, to isolate one variable at a time).
Relevant to Production readiness item 4 below.

- **`float4` vectorization of the K-reduction: a small win, not the 4x
  hoped for** (1.05x-1.2x). Metal's compiler likely already does some of
  this automatically for a short, bounded, compile-time-constant-trip-count
  loop; naive vectorization isn't leaving much on the table by itself.
- **The emitter's blanket copy-every-ref-into-a-local-array pattern: a net
  loss, not the caching win it looks like on paper.** Reading directly from
  `device` memory inside the reduction loop, no local copy, measured ~2x
  faster at K=N=64, still ~1.3x faster at K=N=8. Apple Silicon's unified
  memory means there's no separate slow VRAM tier the copy is protecting
  against; the copy is pure overhead. This is bigger than a `dot_general`
  finding: `declare()`'s copy pattern (`emit/core.py`) is what every rule
  goes through, so it's worth revisiting generally, not just here.

Neither result changes the standing conclusion from this pass: a codegen
tweak to the scalar loop isn't the fix. What actually shipped afterward
(`docs/simdgroup-matmul-design.md`) wasn't threadgroup-memory tiling,
though — a standalone hand-written-MSL trial found tiled `simdgroup_matrix`
only reached *parity* with `jax.jit` CPU at M=K=N=512 once GPU-clock
effects were measured out fairly, with no clear win at the smaller sizes
this project's kernels actually use (head_dim=64, block_kv up to 128).
The cooperative SIMD-group model (above) is what closed the gap instead:
lane-sharded dot products with `simd_sum`/`simd_shuffle_xor`, no
threadgroup memory, no 8x8 tiling. Raw `simdgroup_matrix` remains the
next lever, gated on an `m >= 8`-per-instance matmul actually entering
the benchmark set — the same gating condition the backend-endgame section
below already names.

## Known bugs

**`_rule_scan` carry corruption**, found 2026-08-11 by `test_fuzz_loops`,
not yet fixed. A carry read by a fresh computation and also aliased into a
different carry's output can come out wrong from the second loop iteration
onward — a Metal compiler code-generation issue, reproduced independently
in hand-written MSL with no palladium involved (not an emitter logic
error: the emitted C is provably correct by direct simulation). Full
diagnostic writeup, repro, and why a real fix needs full expression-AST
loop fusion (not a scan-specific patch): `msl-loop-bug.md`. Regression
test: `test_carry_read_and_alias_conflict` (`test_05_loops.py`, `xfail`).
Practical exposure is narrow: no kernel currently in this repo hits the
trigger shape. See Production readiness item 1 for the near-term mitigation.

## Backend endgame: adapt the Mosaic model, do not port it

Assessed 2026-08-04, against the Mosaic GPU reference docs, the Metal 4
tensor material (WWDC25/26), and the Rigel reverse-engineering paper.
Question: is a Mosaic-GPU-style backend for Apple GPUs worth building, or
is Pallas + Mosaic GPU too adapted to NVIDIA hardware?

The combination separates into three layers with three verdicts:

- **Pallas the DSL: reuse as-is.** Tracing, Refs, BlockSpecs, grids, the
  interpret oracle: backend-neutral, proven by already spanning Mosaic TPU,
  Triton, and Mosaic GPU, three models with nothing in common below the
  frontend. palladium already rides this layer.
- **Mosaic GPU's stance: adopt.** Hardware-truthful; explicit memory
  spaces and cooperative scopes; the user controls placement, the compiler
  does not guess. Metal 4 supplies the vocabulary: an
  `execution_simdgroups<N>` scope collectively owning a `cooperative_tensor`
  is the same *idea* as "one Pallas thread = one warpgroup owning an MMA
  fragment", grown independently on different soil.
- **Mosaic GPU's primitive set and pipelining doctrine: do not port.**
  Warpgroups, TMA, mbarrier, clusters, warp specialization are
  Hopper/Blackwell hardware facts, not model choices.

Concept mapping for a hypothetical `plmetal` dialect:

| Mosaic GPU | plmetal (Metal 4) |
|---|---|
| Pallas thread = warpgroup (128 lockstep) | Pallas thread = simdgroup (32), widened by `execution_simdgroups<N>` |
| Explicit GMEM / SMEM / registers | device / threadgroup / thread memory, plus `cooperative_tensor` register fragments |
| WGMMA / tcgen05 + TMEM | `simdgroup_matrix` (8x8); Metal 4 tensor ops over `tensor_handle` / `tensor_inline` |
| TMA async copies + mbarrier double-buffering (`emit_pipeline`) | No public DMA engine. Synchronous staging hidden by SMT occupancy, unified memory, dynamic caching (M3+). The pipelining doctrine is not ported because it is mostly not needed |
| Warp specialization (producer/consumer warps) | Omit. Apple documents no independent-progress guarantees to build it on |
| MLIR -> PTX -> ptxas -> SASS, dumpable per stage | MSL text -> closed AIR toolchain. Introspection: Metal System Trace + GPU counters only |

Two honesty checks before building further on this:

- **Where explicitness pays.** On NVIDIA, Mosaic-style control buys the
  margin automatic compilers leave on the table. On Apple, thread-scope MSL
  reaches the hardware ceiling for bandwidth-bound work: the machine does
  implicitly what Hopper needs spelled out. The dialect earns its
  complexity only where the tensor path matters: MMA-heavy kernels. No
  such kernel was in the benchmark set when this was assessed; the blocked
  attention kernel's `(8, 64) @ (64, 32)` QK^T (`docs/simdgroup-matmul-design.md`)
  is the first one that's actually MMA-shaped, though it's currently served
  by the cooperative SIMD-group model, not raw `simdgroup_matrix`, since
  that already cleared the performance bar (1.42x over `jax.jit` CPU).
- **Platform risk.** The Metal 4 tensor path is young (Rigel documents its
  quirks by reverse engineering, not by spec), macOS 26+, closed compiler,
  no stable public ISA.

## Explicitly out of scope for now

- Batched/strided `dot_general` beyond the 2D case.
- Any reduction/matmul tree strategy beyond one straight-line loop or the
  cooperative simdgroup path; optimize further once something is measured.
- A general PJRT plugin (`jax.jit` transparently targeting Metal). It
  needs a StableHLO bytecode parser in C++; `jax-mps` shows this is
  cheaper than it sounds if you delegate execution to MLX rather than
  writing an optimizing compiler, but it still doesn't buy Pallas support,
  since Pallas kernels aren't expressed as decomposable StableHLO ops for
  a plugin to map. The `jax.ffi` route (`metal_call_jit`) gets most of the
  composability this would buy, without a new PJRT client.

References: Mosaic GPU reference (docs.jax.dev/en/latest/pallas/gpu/),
Rigel (arxiv.org/abs/2606.12765), WWDC25 session 262, WWDC26 session 330,
llama.cpp Metal 4 tensor PR #16634: all checked against source as of
2026-08-10.

## Validation

Same doctrine as metal-runtime: verified by sabotage, not by passing.
Every emitter rule diffs GPU output against the `interpret=True` oracle
with f32-honest tolerances (FAST math: transcendentals are not bit-equal).
When a test goes green, break the rule deliberately (drop the carry
copy-back in scan, swap an offset sign) and confirm the suite catches it.

---

# Production readiness

Audited 2026-08-17: what stands between the current pipeline and being
safe to depend on for real JAX programs that stay on CPU with Pallas
calls dispatching to Apple GPU (`metal_call_jit`, the FFI path). The
architectural hard part — an FFI target that composes inside `jax.jit`
next to ordinary `jnp` ops, with per-shape MSL caching — is done
(`ffi.py`, `native/ffi/palladium_ffi.cpp`). What's left is closing
correctness, coverage, and operational gaps around it.

## Tier 1 — blocks calling this safe to depend on at all

1. **Guard the `_rule_scan` carry-corruption trigger shape.** The known
   bug above is a *silent wrong answer*, not a crash, with `MathMode`
   giving no signal either way. Until the real fix (full expression-AST
   loop fusion) lands, add a static check in `_rule_scan` that detects a
   carry both read by a fresh computation and aliased into a different
   carry's output, and raises `EmitError` instead of emitting wrong code.
   Cheap; turns a silent footgun into a loud one.
2. **Autodiff, near-term stopgap.** `ffi_call` has no JVP/transpose rule by
   default, so any kernel touched by `jax.grad` currently isn't
   differentiated. Wrap a forward `pallas_call` and a separately-traced
   backward `pallas_call` in `jax.custom_vjp`, the way real Pallas/Triton
   kernels (flash-attention included) are differentiated in practice. No
   emitter changes needed; ships fast and unblocks `jax.grad` for
   individually-authored kernels. Narrow by construction: it differentiates
   one hand-written kernel at a time, not an arbitrary training step — see
   "Road to production: whole-jaxpr lowering" below for the general path,
   which gets autodiff for free instead of per-kernel by hand.
3. **Audit the FFI boundary's error surface.** `FfiCallable._spec_and_msl`
   (`ffi.py`) traces and emits MSL at call time; confirm a trace-time
   failure (unsupported primitive, non-contiguous block, strided slice)
   during `jax.jit` compilation surfaces as an actionable JAX-native error,
   not a bare internal traceback from inside lowering.

## Tier 2 — coverage gaps that will hit real programs quickly

4. **`dot_general` perf: done for the shapes measured so far, narrower
   than it needs to be.** The cooperative SIMD-group model (above) closed
   this gap for the case it admits (`m == R`, no batch dims): 1.42x over
   `jax.jit` CPU on the attention kernel, corrected clean of the GPU-clock
   measurement noise that first suggested raw `simdgroup_matrix` tiling
   was the answer. What's still open: batch dims, higher rank, and
   contractions the cooperative detector doesn't admit all still fall back
   to the naive scalar loop with no warning that it'll be slow. General
   `dot_general` (batch dims, arbitrary contraction, f16/bf16 + f32
   accumulation) is scoped as T2.3 in the whole-jaxpr plan below rather
   than as a standalone item, since it needs the same lazy-transpose and
   per-cluster kernel-synthesis machinery that plan already scopes.
5. **Verify `jax.vmap` composes correctly over `metal_call_jit`.** Not
   yet checked: does batching a call to the FFI custom target work
   automatically, or does it need an explicit batching rule registered on
   it? "Production" usually implies vmap-over-batch is normal usage.
6. **Grow primitive coverage demand-driven, not speculatively** — same
   philosophy as the exercise arc. Conv, gather/scatter, and dtypes beyond
   f32 (f16/bf16 matter for real ML workloads on Apple Silicon and are
   currently unscoped) are the likely next asks; add them when a real
   kernel needs them and fails loud, not ahead of time.

## Tier 3 — operational maturity

7. **Bound the native `KernelCache`.** `native/ffi/palladium_ffi.cpp`
   compiles and holds one `MRLibrary`/`MRPipeline` per distinct
   `(msl_source, function_name)` for the process lifetime, with no
   eviction. Fine for a short-lived process or a fixed set of shapes;
   unbounded for a long-lived server tracing many distinct shapes (e.g.
   dynamic batch sizes). Add a cap or LRU before relying on this in a
   long-running process.
8. **User-facing usage doc.** `__init__.py`'s exported surface
   (`metal_call`, `metal_call_jit`, `debug_msl`, `bind`, `trace`,
   `emit_msl`, `rule`) is already small and stable; write a short "how do
   I call this from my own JAX program" doc separate from this roadmap's
   internal build narrative once Tier 1 lands.

Packaging is already production-shaped and not a gap: scikit-build-core
ships `libpalladium_ffi.dylib` inside the wheel, verified against a real
built wheel installed into a clean venv with no repo checkout present,
both eager and under `jax.jit`.

## Road to production, long-horizon: whole-jaxpr lowering

Full plan: `docs/whole-jaxpr-lowering-plan.md` (2026-08-17, scoping only,
nothing implemented yet). Summarized here because it's the real answer to
"autodiff on Apple GPU" for anything beyond a single hand-authored kernel:
instead of writing a `custom_vjp` backward pass per kernel (Tier 1 item 2
above), accept a plain `jax.jit` jaxpr — including one `jax.grad` has
already run over, since JAX supplies the frontend and autodiff and the
jaxpr palladium would consume is post-differentiation — and lower *that*
to Metal. No Pallas kernel authorship required; the tradeoff is that this
is materially more code than anything else in this plan.

Two tiers, split along one structural line: whether per-instance state
fits in registers.

- **Tier 1 (+2-3k lines): scan-shaped scientific programs.** ODE/SDE
  ensembles, fixed-point/line-search loops, MCMC samplers — a
  `scan`/`while` wrapping the whole computation with kilobyte-scale
  carries, batched over a leading axis. This is close to what the emitter
  already does; the "grid" becomes the batch dimension. Entry point
  (`metal_jit_map` or similar), a feasibility gate (reject over-budget
  per-instance state loudly, the 2048-element-wall analog), a demand-driven
  op-coverage tail, and an equivalence-vs-JAX-CPU gate (not a speedup gate
  — the speedup is already established by the existing hand-written
  examples).
- **Tier 2 (+6-10k lines): training steps, no conv.** MLP/small-transformer
  train steps, optimizer microbenches, embedding models — where carry is
  the parameters (megabyte-scale, can't live in registers), so the program
  is irreducibly a DAG of dispatches and the compiler must do fusion,
  buffer liveness, and per-op parallelization by hand. This is where
  `jax.grad`-produced backward passes and optimizer updates get lowered
  automatically, i.e. where autodiff actually becomes general rather than
  per-kernel. Staged: unfused buffer manager + executor first (so the
  fusion pass has a measured baseline to beat), then a deliberately narrow
  single-anchor fusion pass, general `dot_general` (batch dims, arbitrary
  contraction), restricted gather/scatter-add (embedding lookup and its
  gradient; scatter-add determinism decided explicitly, not discovered
  late), then a reward-spec benchmark pass reusing this project's existing
  soaked-median/interleaved-comparison discipline.

Performance bar from the plan's own analysis: a ~136us dispatch floor
(amortizing to ~33-83us pipelined via `mr.Batch`) means fused training
steps need to clear ~5-10ms to hide overhead — medium-and-up models
plausibly win 2-5x, tiny models lose to the CPU floor permanently, and
that's stated as a permanent limit in the plan, not a bug to fix later.
`float64` programs are rejected loudly (Metal has no doubles), consistent
with the rest of this project's f32-first stance.

This is explicitly *not* sequenced ahead of Production readiness Tiers 1-3
above — those are what makes the existing Pallas-authored-kernel path
(`metal_call_jit`) trustworthy today. Whole-jaxpr lowering is the next
horizon once that's solid, and Tier 1 of it (scan-shaped programs) is the
natural next step since it reuses almost all of today's emitter.

## Recommended sequencing

1 → 2 → 5 → 3 → 4, then 7, with 6 and 8 continuous alongside whichever
real kernel or consumer motivates them. Whole-jaxpr Tier 1 is the next
horizon after Tiers 1-3 are solid; whole-jaxpr Tier 2 (the general-autodiff
path) is gated on Tier 1 landing and a real training-shaped workload
motivating it. Everything in "Explicitly out of scope" above
(stretch-13-sized full expression-AST fusion beyond the scan guard,
upstream jax-mps registration) stays demand-driven.
# palladium roadmap: Pallas kernels on Apple GPU

Goal: show that hand-written kernels on Apple GPU beat Diffrax on
ODE-integration-shaped problems, authored in Pallas, executed through
`metal-runtime`. This repo owns everything upstream of "compile this MSL
text": tracing, codegen, dispatch glue, benchmarks. The runtime contract
(`Kernel`/`Buffer`/`run`/`Batch`) lives in the sibling `metal-runtime` repo.

Ecosystem survey (why there is no off-the-shelf path — jax-metal stale and
StableHLO-bound, Triton without a Metal backend, Mosaic GPU structurally
NVIDIA, jax-mps without custom_call): see metal-runtime
`notes/pallas-to-metal.md`, researched 2026-08-03. Unchanged since.

## What the current metal-runtime feature set changes in the plan

Re-checked 2026-08-04 against the runtime as built (not the README of the
public repo, which trails it). Four features change tactics; none change
the architecture:

- **`MathMode` + the measured df32 prelude.** The precision story is now
  concrete: kernels default FAST, compensated arithmetic requires SAFE
  (measured: FAST deletes the error terms of the two sums; `two_prod`'s
  fma-routed term survives). `emit_msl`/`bind` carry `math_mode` through,
  and "rerun the capstone under SAFE + df32 and plot the accuracy/speed
  trade" is now a planned experiment, not hand-waving about float64.
- **Function `constants`/`defines` on `Kernel`.** Trip counts and block
  sizes can be baked per-launch without regenerating MSL text — one source
  string, many specializations, and the source-keyed library cache keeps
  hitting. Adopt when benchmark sweeps (many N, many step counts) make
  regeneration noticeable; not before.
- **Indirect dispatch (`grid` accepts a Buffer).** GPU-decided launch
  sizes. This is the missing primitive for two-phase adaptive schemes
  (integrate; compact the not-yet-converged; relaunch just those). Feeds
  item 7.
- **`Batch` + `gpu_time`.** Benchmark timing without CPU round-trip noise,
  and multi-launch solves (PDE example) without per-launch overhead.

DLPack export on `Buffer` also landed; it becomes interesting at the
`jax.ffi` integration step (zero-copy handoff on unified memory) and is
noted there.

## The pipeline

    Pallas kernel --trace--> KernelSpec --emit--> MSL text --bind--> callable
                  (trace.py)           (emit.py)          (dispatch.py)

`metal_call(...)` composes the three behind a `pl.pallas_call`-shaped API;
`.interpret` on the result is the CPU oracle (`interpret=True`), which every
test diffs against — the single biggest de-risking trick in the project.

Emission model (documented in `emit.py`): one Metal thread per Pallas
program instance; every jaxpr var becomes a thread-local array; every rule
emits plain element loops. Known, accepted restrictions for the initial
build: f32-first, contiguous blocks only (partial leading dim), pure-carry
scans, no indexed ref access. Each has a designated exit (below).

## Core build (done)

Test-driven: each step landed against a pre-written test file, with
`trace.py`, `dispatch.py`, and the emit driver as the fixed scaffold and
the rules as the work. Status: **5/5 done** (2026-08-06); suite grew a
`test_carry_permutation` along the way (sequential copy-back clobbered
permuted scan carries — caught in review, fixed with two-phase staging).
First benchmark number: LV ensemble N=100k, 500 RK4 steps: 4 ms on M2,
36x over jit(vmap(diffeqsolve)) on CPU, max dev 7e-5 (see caveats below).

| # | What | Where | Test |
|---|------|-------|------|
| 1 | Block load/store (`get`/`swap`, `[...]` only) | `_rule_get` / `_rule_swap` | test_02 |
| 2 | Elementwise table + rule | `ELEMENTWISE`, `_rule_elementwise` | test_03 |
| 3 | Grids: `program_id`, BlockSpec offsets | `_rule_program_id`, `_block_offset` | test_04 |
| 4 | `fori_loop`/pure-carry `scan` -> C for-loop | `_rule_scan` | test_05 |
| 5 | Capstone: batched RK4 Lotka-Volterra (composition only) | — | test_06 |

Stretch goals, in rough order of payoff:

6. **Indexed ref access** (`x_ref[i, j]`, static and loop-var indices) +
   strided blocks. Unlocks stencils (example 4) and lifts the contiguity
   restriction. The `get`/`swap` rules grow an NDIndexer walk.
7. **Per-thread adaptive stepping** — the genuinely novel benchmark result
   (see example 2 for the measured lockstep tax). Embedded RK pair + PI
   controller per thread; needs `while`-shaped iteration (bounded-trip
   `for` with early exit is enough — no new jaxpr machinery if authored as
   fori_loop + select), plus optionally indirect dispatch for two-phase
   compaction.
8. **In-kernel counter-based RNG** (Philox) as an emitter intrinsic, so SDE
   kernels can be authored in Pallas instead of hand-written MSL
   (example 3 sets the bar).
9. **df32 kernels**: `math_mode=SAFE` + the metal-runtime df32 prelude
   prepended to emitted source; measure accuracy vs speed on the capstone.
10. **`jax.ffi` integration**: register dispatch as an FFI target on the
    CPU platform so `metal_call` results compose with jit-traced JAX code.
    Unified memory + DLPack should make the handoff cheap; measure, then
    document the alignment story (XLA gives 64-byte alignment,
    `newBufferWithBytesNoCopy` wants pages).
11. **Upstream-worthy backend**: `MetalCompilerParams` +
    `pallas_core.register_lowering_rule(...)`, patch the platform dict in
    `_pallas_call_lowering`, contribute custom_call execution to jax-mps.
    Only worth it if the showcase lands and there is community pull.
12. **plmetal scope tier**: promote "one Pallas thread = one Metal thread"
    to a scope ladder — simdgroup primitives, threadgroup staging +
    barriers, cooperative-tensor MMA as an intrinsic (the same entry path
    as Philox and df32). Explicitly gated on an MMA-heavy kernel entering
    the benchmark set; rationale and concept mapping in the backend
    endgame section below.
13. **Expression AST in the emitter**: fold single-use temporaries into
    their consumers and render with precedence-aware minimal parens.
    CVal.expr grows from string to a small tree; bundled deliberately,
    because minimal parens alone buy the machinery without the payoff
    (SSA materialization means expressions never nest today, so the
    assign-site outer-paren strip already covers the visible noise, and
    template parens stay as the atomicity contract on CVal.expr). Do it
    when emitted kernels get long enough that statement count, not paren
    depth, hurts readability; re-bless goldens and re-run the fuzzer as
    the gate.

## Benchmarks (examples/)

Honest framing, stated wherever numbers are shown: on macOS the practical
Diffrax baseline is JAX's **CPU** backend (jax-metal cannot run Diffrax;
jax-mps coverage of its while_loop internals is untested).

1. `01_ode_ensembles.py` — 100k-member Lotka-Volterra parameter sweep,
   RK4-in-kernel vs jit(vmap(diffeqsolve)). Activates with core item 5.
2. `02_adaptive_lockstep.py` — measures the vmap lockstep tax (2% stiff
   members poison the batch). Runs today; the fix is stretch 7.
3. `03_sde_montecarlo.py` — 1M-path GBM pricing, hand-written MSL with
   in-kernel RNG, validated against Black-Scholes. Runs today; sets the
   bar for stretch 8.
4. `04_reaction_diffusion.py` — Gray-Scott stencil, CPU baseline today;
   Metal version gated on stretch 6 (+ threadgroup memory tiling).

Later candidates: integral equations (Nystrom + in-kernel iterative solve),
repeated quadrature under parameter sweeps, and — significant because it is
the trigger condition for the backend endgame — an MMA-heavy workload:
batched implicit steps or neural ODEs with small dense Jacobian blocks,
where per-thread scalar loops stop being the right shape and the
simdgroup-matrix / cooperative-tensor path starts paying.

## Validation

Same doctrine as metal-runtime: verified by sabotage, not by passing.
Every emitter rule's tests diff GPU output against the `interpret=True` oracle
with f32-honest tolerances (FAST math: transcendentals are not bit-equal).
When a test goes green, break the rule deliberately (drop the carry
copy-back in scan, swap an offset sign) and confirm the suite catches it.

## Backend endgame: adapt the Mosaic model, do not port it

Assessed 2026-08-04, against the Mosaic GPU reference docs, the Metal 4
tensor material (WWDC25/26), and the Rigel reverse-engineering paper.
Question: is a Mosaic-GPU-style backend for Apple GPUs worth building, or
is Pallas + Mosaic GPU too adapted to NVIDIA hardware?

The combination separates into three layers with three verdicts:

- **Pallas the DSL — reuse as-is.** Tracing, Refs, BlockSpecs, grids, the
  interpret oracle: backend-neutral, proven by already spanning Mosaic TPU,
  Triton, and Mosaic GPU — three models with nothing in common below the
  frontend. Each backend is really its own primitive dialect (`plgpu.*`,
  `pltpu.*`) over shared machinery. palladium already rides this layer.
- **Mosaic GPU's stance — adopt.** Hardware-truthful; explicit memory
  spaces and cooperative scopes; the user controls placement, the compiler
  does not guess ("puts you more in control", per its own reference). The
  stance ports. Metal 4 even supplies the vocabulary: an
  `execution_simdgroups<N>` scope collectively owning a `cooperative_tensor`
  is the same *idea* as "one Pallas thread = one warpgroup owning an MMA
  fragment", grown independently on different soil.
- **Mosaic GPU's primitive set and pipelining doctrine — do not port.**
  Warpgroups, TMA, mbarrier, clusters, warp specialization are
  Hopper/Blackwell hardware facts, not model choices.

Concept mapping for a hypothetical `plmetal` dialect:

| Mosaic GPU | plmetal (Metal 4) |
|---|---|
| Pallas thread = warpgroup (128 lockstep) | Pallas thread = simdgroup (32), widened by `execution_simdgroups<N>` |
| Explicit GMEM / SMEM / registers | device / threadgroup / thread memory, plus `cooperative_tensor` register fragments |
| WGMMA / tcgen05 + TMEM | `simdgroup_matrix` (8x8); Metal 4 tensor ops over `tensor_handle` / `tensor_inline` |
| TMA async copies + mbarrier double-buffering (`emit_pipeline`) | No public DMA engine. Synchronous staging hidden by SMT occupancy, unified memory, dynamic caching (M3+). The pipelining doctrine is not ported because it is mostly not needed — this *simplifies* the model |
| Warp specialization (producer/consumer warps) | Omit. Apple documents no independent-progress guarantees to build it on |
| MLIR -> PTX -> ptxas -> SASS, dumpable per stage | MSL text -> closed AIR toolchain. Introspection: Metal System Trace + GPU counters only |

Two honesty checks before building any of it:

- **Where explicitness pays.** On NVIDIA, Mosaic-style control buys the
  margin automatic compilers leave on the table. On Apple, thread-scope MSL
  (what the current emitter produces) reaches the hardware ceiling for
  bandwidth-bound work — the machine does implicitly what Hopper needs
  spelled out. The dialect earns its complexity only where the tensor path
  matters: MMA-heavy kernels (neural ODEs, implicit methods with dense
  blocks). No such kernel in the benchmark set yet = no dialect yet.
- **Platform risk.** The Metal 4 tensor path is young (Rigel documents its
  quirks by reverse engineering, not by spec), macOS 26+, closed compiler,
  no stable public ISA.

The path there is evolutionary, and this repo is already on it: today's
emitter IS the thread-scope dialect. Scope becomes a concept (simdgroup
primitives, threadgroup staging + barriers), then cooperative-tensor MMA
enters as an intrinsic exactly the way Philox and df32 do (stretch 8/9).
Sequencing unchanged: jax-mps custom_call execution (stretch 10) comes
first — it is the production vehicle any backend needs, dialect or not —
and the upstream registration conversation (stretch 11) only after the
tensor-path kernels justify it.

References: Mosaic GPU reference (docs.jax.dev/en/latest/pallas/gpu/),
Rigel (arxiv.org/abs/2606.12765), WWDC25 session 262, WWDC26 session 330,
llama.cpp Metal 4 tensor PR #16634.
