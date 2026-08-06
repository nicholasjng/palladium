# Development notes

The project's lab notebook: design explorations, measurement records,
and reward specifications, kept for provenance. Nothing here is user
documentation (that lives one level up in `docs/`), and nothing here is
a stability promise.

- [reward-spec-matmul-emitter.md](reward-spec-matmul-emitter.md):
  reward function for the matmul-emitter optimization arc (v0.1 to
  v0.3), with the full calibration changelog and thermal-measurement
  discipline.
- [reward-engineering-case-study.md](reward-engineering-case-study.md):
  write-up of the optimization arc as a reward-engineering case study
  (0.06 to saturation, what each emitter change bought).
- [query-blocking-scratch.md](query-blocking-scratch.md): why one query
  row per instance was the bottleneck; hand-written MSL measurements
  that motivated the R-row cooperative model.
- [simdgroup-matmul-design.md](simdgroup-matmul-design.md): the
  standalone simdgroup-matmul experiment (retired) and the MMA findings
  that moved into the emitter.
- [reward-spec-simdgroup-matmul.md](reward-spec-simdgroup-matmul.md):
  retired reward spec for that experiment.
- [emitter-simplifications.md](emitter-simplifications.md): running
  list of deferred work with the measurements gating each item;
  includes the Metal compiler carry-permutation bug record.
- [whole-jaxpr-lowering-plan.md](whole-jaxpr-lowering-plan.md): scoping
  plan for lowering plain jax.jit jaxprs (Tiers 1 and 2), shelved.
- [production-readiness-plan.md](production-readiness-plan.md): the
  plan this docs layout came from.
- [coop-colvec-and-transposed-lhs.md](coop-colvec-and-transposed-lhs.md):
  emitter work for GEMM-shaped training, gated on the MLP example's
  measurements (2-D output tiling, transposed-lhs dots, colvec bias
  layout).
