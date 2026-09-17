# Compiler portfolio project ideas

Date: 2026-09-11. Brainstorm preserved from career/project planning.

Goal: demonstrate compiler engineering proficiency, especially MLIR, with
numerical computing as a natural starting point and room to explore other fields.
Choose one main project and make its semantics, legality, tests, and results
reviewable. A bounded upstream contribution emerging from the work would add
useful evidence.

## 1. Numerically sensitive arithmetic compiler

Build a small dialect for compensated arithmetic: `two_sum`, `two_prod`, and
double-float addition/multiplication. Lower to ordinary floating-point operations
while preserving requirements around FMA, contraction, and reassociation.

- Learn: TableGen operation/type definitions, verifiers, canonicalization,
  rewrite patterns, type conversion, arithmetic/math and LLVM lowering.
- First scope: straight-line arithmetic and one CPU loop-based integrator.
- Evidence: a profitable legal optimization and a tempting rejected one;
  cancellation, subnormal, overflow, and long-horizon validation.
- Avoid initially: automatic precision selection for arbitrary programs.
- Connection: Palladium's compensated df32 and long-horizon Kepler experiments.

## 2. Memory optimization for scan-heavy ODE kernels — selected direction

Eliminate intermediate trajectories and copies, place required outputs directly
in their destination, and fuse pure observations into stepping when the full
state trajectory is unnecessary.

- Learn: `scf`, `tensor`, `memref`, use-def and effects analysis, aliasing,
  bufferization, transformation legality.
- First scope: one fixed-step solver and one supported consumer pattern.
- Evidence: before/after IR, allocation/copy counts, peak storage, runtime, and
  negative cases in which the pass refuses to transform.
- Avoid initially: rewriting all of Palladium or building a general JAX compiler.
- Detailed stack and milestones: [ODE compiler project](ode-compiler-project.md).

## 3. Stencil compiler with explicit schedules

Compile reaction-diffusion or finite-difference kernels while separating their
mathematics from tiling, fusion, vectorization, and storage choices. Express
schedules with the Transform dialect and implement one transformation or
legality check.

- Learn: structured operations, `linalg`, `affine`/`scf`, `vector`, bufferization,
  and Transform dialect extensions.
- First scope: a CPU 2D stencil with boundary handling and multiple grid sizes.
- Evidence: explain gains and losses through cache behavior, redundant work,
  vectorization, and memory traffic.
- Avoid initially: autotuning before understanding a few manual schedules.
- Connection: Palladium's Gray–Scott reaction-diffusion workload.
- Reference: [Transform tutorial](https://mlir.llvm.org/docs/Tutorials/transform/).

## 4. Static checker for unsafe GPU synchronization

Detect barriers that only some threads in a workgroup may reach. Begin with
thread-dependent conditionals, then loops with nonuniform trip counts.

- Learn: GPU IR, regions/control flow, dataflow analysis, effects, diagnostics.
- First scope: a conservative classification of workgroup-uniform, potentially
  divergent, and unknown values, used to diagnose barrier placement.
- Evidence: diagnostics tracing divergence to its origin and tests separating
  proven-safe, unsafe, and unproven cases.
- Avoid initially: claiming complete GPU race detection.
- Connection: Palladium's explicit barriers and threadgroup memory.

## 5. Tiny relational query compiler

Compile explicit scan/filter/project/aggregate plans to native loops. Implement
predicate pushdown and pipeline fusion; investigate when materializing an
intermediate is preferable.

- Learn: custom dialects and regions, rewrite legality, layered lowering,
  runtime calls, LLVM generation.
- First scope: fixed-width non-null integer columns and a global aggregate.
- Evidence: high-level rewrites visibly changing generated loops and memory
  traffic; compilation and execution time measured separately.
- Avoid initially: SQL parsing, strings, joins, and full null semantics.
- Reference: [LingoDB](https://www.lingo-db.com/docs/) provides a real MLIR-based
  query-engine architecture to study or contribute to.

## 6. Differential fuzzer for an MLIR optimization pipeline

Generate valid small programs, compare execution before and after selected
passes, and reduce failures to readable reproducers.

- Learn: IR construction and verification, pass pipelines, execution, reduction.
- First scope: bounded loops and integer operations with carefully specified,
  defined behavior; floating-point programs later.
- Evidence: a real bug, minimized reproducer, root cause, and ideally a patch.
- Avoid initially: arbitrary MLIR generation without a reliable validity and
  execution oracle.
- Connection: Palladium's Hypothesis differential tests and sabotage testing.
- Reference: [MLIR testing guide](https://mlir.llvm.org/getting_started/TestingGuide/).

## Extensions worth retaining

- A pure-expression egglog optimizer within an ODE step, compared with ordinary
  canonicalization and CSE.
- Extraction costs accounting for shared expressions, live temporary storage,
  and recomputation rather than operation count alone.
- Explicit strict versus algebraic numerical modes, assessed against long-run
  dynamics as well as local error.
- Dynamic dialect/schema tooling with IRDL if compiler tooling itself becomes
  the main interest; keep this separate from the initial ODE project's scope.

For hiring evidence, prioritize a clear execution contract, readable IR,
nontrivial legality conditions, meaningful negative tests, reproducible
measurements, and a concise design document. A substantial native MLIR C++ pass
would complement a Python/xDSL prototype.
