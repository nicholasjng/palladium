# Whole-jaxpr lowering: Tier 1 and Tier 2 plan

**Scoping document, 2026-08-17.** Plan for accepting plain `jax.jit`
jaxprs — no Pallas kernel authorship — and running them on Metal faster
than JAX-on-CPU. Written as the stopping point of the second exploration
phase; nothing here is implemented.

## Context and the structural line

Everything palladium wins at today, it wins because a human put the whole
computation inside one dispatch: a Pallas kernel whose loop carries fit in
registers. Whole-jaxpr porting changes who does that work, and the cost
splits along one line:

- **Carry fits in a thread** (ODE ensembles, samplers, optimization inner
  loops): the jaxpr is a `scan`/`while` wrapping the whole computation
  with kilobyte-scale carries. Lowering degenerates to what the emitter
  already does; the "grid" is the batch dimension. This is Tier 1.
- **Carry is the parameters** (model training): megabyte-scale state
  cannot live in registers, so the program is irreducibly a DAG of
  dispatches over device buffers, and the compiler must do what the
  Pallas author did by hand: fusion, buffer liveness, per-op
  parallelization. This is Tier 2.

Line-count predictions against the current ~3.3k-line package: Tier 1
+2-3k, Tier 2 +6-10k on top. Calibration anchor: tinygrad's complete
lazy-graph/fusion/Metal/autodiff stack is ~10k lines; Tier 2 builds less
(JAX supplies the frontend and autodiff — `grad` runs before the jaxpr
reaches us) against a stronger baseline (XLA-CPU fuses better than eager
frameworks).

Performance bar, from this repo's measurements: the ~136us dispatch floor
amortizes to ~33-83us pipelined through `mr.Batch` (metal-runtime devlog),
so a ~20-kernel fused step carries ~1ms of overhead and needs step times
above ~5-10ms to hide it. GEMM phases favor the GPU decisively (matrix
units vs ~0.5 TFLOP/s NEON); bandwidth-bound elementwise phases are a
wash on unified memory, so fusion quality decides them. Expected outcome:
medium-and-up training steps win 2-5x; tiny models and small batches lose
to the CPU on the floor alone, permanently. `float64` programs are out;
Metal has no doubles (reject loudly, point at disabling `jax_enable_x64`).

---

## Tier 1: scan-shaped scientific programs (+2-3k lines)

Target workloads: ODE/SDE ensembles, fixed-point and line-search loops,
MCMC-style samplers — per-instance programs batched over a leading axis,
with per-instance state small enough for registers.

### T1.1 Entry point and API (~300 lines)

- `metal_jit_map(fn, batch: int)` (name TBD): traces `fn` *per-instance*
  with `jax.make_jaxpr`, no `pallas_call`. The user writes the
  per-instance function; the batch count supplies the grid — the same
  division of labor as a Pallas grid, minus refs and BlockSpecs.
- Synthesize the `KernelSpec` directly: inputs/outputs become device
  buffers with one-block-per-instance `BlockInfo`s. Needs a `trace.py`
  refactor allowing specs to be constructed without going through Pallas
  (the emitter is already Pallas-agnostic past `KernelSpec`).
- Batched-jaxpr inference (accepting an already-`vmap`ped function and
  recovering the mapped axis) is deliberately deferred to a T1.5: it is
  batch-dimension analysis across every primitive, and the explicit API
  covers the target workloads without it.

### T1.2 Feasibility gate (~150 lines)

- Static per-instance live-value estimate (bytes across the jaxpr, the
  2048-element-wall analog). Over budget -> reject with a message naming
  the largest values, not a silent slow path. This check is what keeps
  Tier 1 honest: anything that passes it inherits today's
  single-dispatch performance model wholesale.

### T1.3 Op coverage tail (~600-900 lines, demand-driven order)

- Elementwise table lines: `rsqrt`, `log1p`, `expm1`, `erf`, `logistic`,
  `floor`, `ceil`, `round`, `sign`, `rem`, `atan2`, `is_finite`,
  `square`, shifts. One line plus a Metal intrinsic each; add on first
  demand by a ported example.
- Rank-1 `dot_general` canonicalization (matvec as `(1, k)`), already
  noted in `docs/emitter-simplifications.md`.
- Value-level `slice`/`dynamic_slice`, `concatenate`, `iota`, `rev`,
  simple `pad`: flat-loop rules in the classic model, same shape
  discipline as `broadcast_in_dim`.
- `cumsum`/`cumprod` via serial per-instance loops; `argmax`/`argmin` as
  reduce variants carrying an index. `sort`/`top_k` deferred (Tier 3).
- Cooperative ports only when a Tier 1 program actually stages a dot
  (most will not): otherwise everything runs the classic model.

### T1.4 Validation and benchmark (~300 lines)

- Oracle is JAX itself: run `fn` on CPU, compare. No interpret plumbing
  needed — this is the one place whole-jaxpr is *simpler* than Pallas.
- Port example 01 or 03 (ODE/SDE ensemble) to the new entry point; gate:
  matches the hand-written Pallas version's timing within noise (it
  should be nearly the same emitted kernel). That equivalence, not a
  speedup, is the Tier 1 exit criterion; the speedup over JAX-CPU is
  already established by the existing examples.

---

## Tier 2: training steps, no conv (+6-10k lines)

Target workloads: MLP and small-transformer train steps (forward, loss,
backward, optimizer update), optimizer-only microbenches (Adam, L-BFGS),
embedding models. Explicitly out of scope: convolution and pooling
(Tier 3), general StableHLO `gather` semantics, `sort`/`top_k`, dynamic
shapes, multi-device, f64.

Build order is chosen so every stage is measurable on its own, and the
unfused baseline is measured *first* — it quantifies exactly what the
fusion pass is worth, in the same way the occupancy diagnostic justified
the cooperative model before it was built.

### T2.1 Buffer manager and executor (correct-but-slow first, ~1-1.5k)

- Liveness analysis over the jaxpr DAG; arena allocation with reuse;
  `donate_argnums` honored; outputs aliasable to donated inputs.
- Persistent device residency for parameters across steps: generalize
  `BoundKernel.pinned` into a device-array-like handle (params stay on
  device between calls, mirroring what `jnp` arrays give the CPU
  baseline; the reward-spec v0.2.0 symmetric-residency lesson, applied
  as API).
- Executor: one `mr.Batch` per step, one dispatch per equation (no
  fusion yet), scalar readbacks (loss) at the end. Cached plan per input
  shape/dtype signature.
- **Gate/measurement:** an MLP step runs correctly, and its unfused
  step time is recorded. This number is the denominator for everything
  after.

### T2.2 Fusion pass (~1.5-2.5k — the make-or-break subsystem)

- Cluster the DAG: elementwise/broadcast/lazy-transpose chains fuse into
  their consumers; reductions absorb elementwise prologues and
  epilogues; dots anchor clusters with fused bias/activation epilogues;
  `scan`/`while` bodies are clusters as-is (Tier 1 machinery per
  cluster).
- v1 legality rule, deliberately narrow: single anchor per cluster, all
  members sharing the anchor's iteration domain; multiple outputs
  allowed. Every widening of this rule must cite a measured win, or it
  will grow into a research project (this is the documented failure mode
  of every fusion pass ever written).
- Per-cluster kernel synthesis: iteration domain -> auto grid/BlockSpec
  (flat 1D for elementwise clusters with N elements/thread; row
  strategies for reductions; the existing cooperative/MMA model for dot
  clusters, with the measured defaults from the attention work: one
  SIMD-group per 8 output rows, fragment cap 8).
- The scalar-vs-cooperative decision becomes per-cluster; the existing
  detection walker generalizes (it already answers exactly this
  question for one kernel).
- **Gate:** fused step time vs T2.1 baseline, per workload; elementwise
  phases should approach bandwidth (compare against a memcpy roofline).

### T2.3 General `dot_general` (~500-1k)

- Batch dims: batch -> grid axis (large batch) or per-simdgroup loop
  (small batch); MMA path extended with a per-batch tile loop.
- Arbitrary contractions canonicalized to transpose + standard matmul,
  reusing the lazy-transpose machinery.
- `preferred_element_type` honored (accumulator dtype decoupled from
  output aval); f16/bf16 inputs admitted to the vectorized/MMA paths
  with f32 accumulation.
- **Gate:** batched GEMM microbench vs `jax.jit` CPU across sizes;
  record the crossover size — it defines "medium-and-up" concretely.

### T2.4 Gather / scatter-add (~500-1k)

- Restricted, common forms only: gather = embedding lookup (integer
  index vector over the leading axis); scatter-add = its gradient.
- Scatter-add strategy decision up front, because it interacts with the
  validation gates: device atomics on f32 are nondeterministic in
  summation order; alternatives are one-hot-matmul (small vocab) or
  sort-segment-reduce (deterministic, more code). Start with atomics +
  a documented tolerance for gradient checks; revisit if a gate needs
  bitwise stability.
- Everything outside these forms rejects with a message naming the
  unsupported dimension_numbers — no silent generality.

### T2.5 Remaining plumbing (~500-800)

- Threefry/`jit`-inline rules ported to the cooperative table (already
  noted in `docs/emitter-simplifications.md`; needed the moment dropout
  fuses next to a dot).
- Dtype promotion edge cases; `stop_gradient` (identity);
  `custom_jvp/vjp` residues (inlined by the time `grad` has run, but
  verify); `select_and_scatter`-free assert list so unsupported ops fail
  at trace time with the op name.

### T2.6 Benchmark and reward spec (~400 + a spec document)

- Reuse the reward-engineering methodology wholesale: gates first
  (parameter updates match a JAX-CPU reference step within tolerance
  after k steps; loss trajectory agreement; provenance; multi-seed),
  then one dense metric (step-time ratio vs `jax.jit` CPU, worst-case
  over the workload set, normalized to a measured ceiling).
- Workload set: MLP step (established baseline), transformer-block step
  (the attention kernel already exists and is 3.5x ahead), Adam-only
  microbench (bandwidth-bound; the honest "wash" case that keeps the
  metric from being flattered by GEMMs alone).
- All timing under the established discipline: soaked medians, paired
  interleaving for candidate comparisons, device time for
  kernel-vs-kernel, thermal spread recorded (`docs/reward-spec-*`).

---

## Risks, ranked

1. **Fusion-pass scope creep** — mitigated by the single-anchor legality
   rule and measured gates per widening.
2. **Scatter-add determinism vs correctness gates** — decided explicitly
   in T2.4, not discovered during calibration.
3. **Register estimation accuracy** (T1.2): a wrong feasibility gate
   either rejects good programs or ships spilling kernels; validate the
   estimator against the known 2048-element wall.
4. **JAX version coupling**: whole-jaxpr consumption touches many more
   primitive params than the Pallas subset; keep the structural-mirror
   discipline (`_Slice`-style Protocols) and add a jaxpr-inventory test
   that fails loudly on unknown params rather than mis-lowering.
5. **The dispatch floor is physics**: no plan item recovers small-model
   training; say so in the eventual reward spec's intent section rather
   than letting the metric imply otherwise.

## Sequencing summary

```
T1.1 entry point -> T1.2 feasibility gate -> T1.3 op tail (on demand)
  -> T1.4 example port + equivalence gate            [stop/ship point]
T2.1 buffers + unfused executor (measure baseline!)
  -> T2.2 fusion pass (measure vs baseline)
  -> T2.3 general dot (measure crossover)
  -> T2.4 gather/scatter (decide determinism)
  -> T2.5 plumbing -> T2.6 reward spec + calibration [stop/ship point]
```

Each arrow is a measurement gate in the project's established style: no
stage begins until the previous stage's number is recorded, and any
stage whose number disappoints gets the interleaved-A/B treatment before
being built upon.
