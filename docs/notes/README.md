# Development notes

The project's lab notebook: design explorations and measurement
records, kept for provenance. Nothing here is user documentation (that
lives one level up in `docs/`), and nothing here is a stability
promise.

- [jax-mps-training-roadmap.md](jax-mps-training-roadmap.md): staged plan for
  differentiable parameter recovery and a continuous-normalizing-flow workload
  after the Palladium/jax-mps custom-call spike (2026-09-17).
- [ode-compiler-project.md](ode-compiler-project.md): proposed standalone
  Python/xDSL ODE trajectory compiler, storage optimizations, optional egglog,
  package structure, and milestones (2026-09-11).
- [compiler-project-ideas.md](compiler-project-ideas.md): compiler portfolio
  brainstorm, including numerical, GPU analysis, query, and fuzzing projects
  (2026-09-11).

- [emitter-feature-sketches.md](emitter-feature-sketches.md): 2026-09-06
  review follow-up; typed lowering, storage/views, masked blocks, primitive
  coverage, fusion, cooperative reductions, and validation gates.

- [emitter-simplifications.md](emitter-simplifications.md): running
  list of deferred work with the measurements gating each item;
  includes the Metal compiler carry-permutation bug record.
- [whole-jaxpr-lowering-plan.md](whole-jaxpr-lowering-plan.md): scoping
  plan for lowering plain jax.jit jaxprs (Tiers 1 and 2), shelved.
- [production-readiness-plan.md](production-readiness-plan.md): the
  plan this docs layout came from.
- [devlog.md](devlog.md): per-card narrative of the tutor arc — what
  landed and why, one subsection per card.

The cooperative SIMD-group/MMA GEMM lowering's design notes
(`query-blocking-scratch.md`, `simdgroup-matmul-design.md`,
`coop-colvec-and-transposed-lhs.md`) moved to the `mgemm` sibling
project along with the code they describe (2026-08-19 split); see its
`docs/notes/` instead.
