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
