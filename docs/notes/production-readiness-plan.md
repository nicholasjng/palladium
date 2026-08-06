# Production readiness plan for the palladium emitter

Written 2026-08-18, against the post-restructure tree (emit/ package,
cooperative MMA model, FFI cooperative dispatch; 91 tests green).

"Production-ready" here means: a third party can `pip install` palladium,
author a Pallas kernel within a documented subset, get either a fast
kernel or a clear error, and trust the package not to change under them
without a version bump. It does not mean an upstream JAX backend
(ROADMAP stretch 11) or whole-jaxpr lowering
(`whole-jaxpr-lowering-plan.md`, shelved). Those stay out of scope.

The plan is four phases. Phases 1 and 2 are the substance; 3 and 4 are
release mechanics. Each phase has an acceptance gate so it is checkable,
in the spirit of the reward specs.

## Where the gap actually is

The happy path is strong: traced-vs-oracle testing on every rule, golden
MSL snapshots, a differential fuzzer, byte-identical-MSL discipline for
refactors, and a CI canary against jax head. The production gap is
concentrated at the edges:

1. **Failure behavior.** A kernel one step outside the supported subset
   either raises a bare `NotImplementedError` from deep in a rule, or
   worse, silently succeeds slowly: one MMA-ineligible dot drops the
   whole kernel from the cooperative model to the classic one, a 5-10x
   cliff with no signal beyond the undocumented `BoundKernel.cooperative`
   flag.
2. **Contract definition.** The supported subset (dtypes, primitives,
   block shapes, the cooperative eligibility rules) exists only as code
   and internal notes. No consumer-facing statement of what is in, what
   is out, and what falls back.
3. **Concurrency and process hygiene.** `MetalCallable.cache`,
   `FfiCallable._cache`, and FFI registration have no locks. Fine for a
   research script, not for a library imported under a threaded server
   or `jax` async dispatch.
4. **Distribution.** `metal-runtime` is a `[tool.uv.sources]` path
   dependency, version is 0.1.0 with no changelog, no wheels are
   published, and `docs/` is internal exploration notes rather than user
   documentation.

## Phase 1: contract and diagnostics

The theme: every kernel a user writes gets one of three outcomes, and
the outcome is always explainable. (a) cooperative fast path, (b)
classic path with a discoverable reason, (c) a trace/emit-time error
naming the offending primitive and the nearest supported alternative.

1. **Explainability API.** `coop_rows`/`_coop_walk` currently return
   `None` on the first `_CoopUnsupported`; capture the reason instead.
   Surface it two ways:
   - `MetalCallable.explain(*args)` (and the same on `FfiCallable`):
     traces, reports chosen model, launch geometry, per-rejection
     reasons, and the emitted MSL length. No compile, no dispatch.
   - `PALLADIUM_EXPLAIN=1` env var: one stderr line per compiled kernel
     (`model=cooperative rows=8 grid=...` or
     `model=classic reason="dot_general lhs is computed, not device"`).
   This is the top backlog item in `emitter-simplifications.md` and the
   single highest-leverage production change.
2. **Error taxonomy.** One exception hierarchy: `PalladiumError` base,
   `TraceError` (unsupported pallas_call structure), `EmitError`
   (unsupported primitive/shape/dtype, already exists), `DispatchError`
   (runtime failures). Every `raise` names the jaxpr equation
   (primitive, avals) and, where a designated exit exists, says what to
   do ("rank-1 dot: reshape lhs to (1, k)"). Audit all raise sites in
   `trace.py`, `emit/core.py`, `emit/rules.py`, `emit/coop.py`.
3. **Input validation at the call boundary.** `MetalCallable.__call__`
   should check dtype/shape agreement against the traced spec before
   dispatch and raise a `DispatchError` naming the mismatched argument,
   instead of whatever the runtime does with a wrong-sized buffer.
   Reject non-C-contiguous inputs explicitly (or copy, and say so in
   the docstring; decide once, document it).
4. **Correctness holes with known fixes** (both small, both listed in
   `emitter-simplifications.md`):
   - Honor `preferred_element_type` in both dot rules, or reject
     kernels that set it to something other than the output dtype.
   - Canonicalize rank-1 `dot_general` (matvec) as `(1, k) @ (k, n)`.
5. **FAST-math contract.** `math_mode=FAST` is the default and changes
   transcendental results vs the interpret oracle. State the tolerance
   contract in the `metal_call` docstring and the user docs; keep SAFE
   as the documented escape for compensated arithmetic.

Acceptance gate: a test file (`test_18_diagnostics.py`) that asserts
(a) `explain()` output for one cooperative and one fallback kernel
names the correct model and reason, (b) five representative
out-of-subset kernels each raise the right exception type with the
primitive name in the message, (c) a dtype-mismatched call fails before
dispatch. Scored kernels' MSL stays byte-identical (existing hashes).

## Phase 2: robustness

1. **Thread safety.** A lock per cache (`MetalCallable`,
   `FfiCallable`), double-checked around trace/emit/compile so the
   compile happens once per shape even under concurrent first calls.
   `_register()` is already `functools.cache` but is not atomic under
   threads; guard it with the same lock. Test with a
   ThreadPoolExecutor hammering one callable with two shapes.
2. **Dtype coverage tests.** The CTYPES table claims f32/f16/bf16/i32/
   u32/bool; only f32 (and i32/u32 via RNG) are exercised end to end.
   Add oracle-diff tests per dtype for load/store, elementwise, and
   reductions; document which dtypes the dot rules accept (f32 only
   today) and enforce that with a clear error rather than wrong code.
3. **Fuzzer expansion.** `test_fuzz_emitter.py` predates cond/while and
   the logical ops; extend the grammar to cover them and the
   cooperative model (attention-shaped programs with varying R, W).
   Keep it opt-in (`-m fuzz`) but run it as a documented release step.
4. **Sabotage pass.** Per the validation doctrine: deliberately break
   each new diagnostic and thread-safety mechanism and confirm a test
   catches it (e.g. drop the lock, force a wrong fallback reason).
5. **Resource limits.** Decide and document behavior for oversized
   requests: threadgroup memory over 32KB (the MMA staging tile and
   `_coop_elementwise_tg_dst` allocate it), grids over device limits.
   Today these surface as Metal compile or dispatch errors; wrap them
   with the actual numbers and the limit that was hit.

Acceptance gate: dtype matrix tests green, fuzz suite green over a
fixed seed corpus including control flow, concurrency test green under
`pytest -x -m ""` locally, sabotage checklist recorded in the test
file's docstring.

## Phase 3: versioning and API surface

Distribution is shelved by decision (2026-08-18): no wheels, no PyPI,
no dependency-source change. `[tool.uv.sources]` path dependency on
metal-runtime stays as the development setup; the pin swap happens
when metal-runtime pushes a v0.1.0 tag, and wheel/release mechanics
follow from there. What remains in scope now:

1. **Versioning.** Bump to 0.2.0, start a CHANGELOG.md (the reward-spec
   changelog is the raw material for the 0.1 -> 0.2 entry), adopt
   "minor may break, patch may not" pre-1.0 semver, and state the jax
   pin policy (`>=0.11,<0.12`, canary-widened) in the README.
2. **Public API freeze.** `__all__` is the contract: `metal_call`,
   `metal_call_jit`, `debug_msl`, `trace`, `emit_msl`, `bind`, `rule`,
   the result classes. Everything in `emit.core`/`emit.coop` is
   private; say so once in the package docstring.

Acceptance gate: version/changelog/pin policy present; `__all__`
matches the documented surface.

## Phase 4: documentation

1. **Split docs/ into `docs/` (user-facing) and `docs/notes/`
   (exploration record).** The reward specs, scratch files, design
   docs, and this plan move to notes/ with a one-line index; they are
   the project's lab notebook, not its manual.
2. **User docs, four pages:**
   - Getting started: install, first kernel, `debug_msl`, `interpret`
     oracle workflow.
   - Supported subset: the contract from Phase 1 as a table
     (primitives, dtypes, block shapes, control flow), plus the
     cooperative eligibility rules and what `explain()` tells you.
   - Performance guide: when the cooperative/MMA path triggers, the
     ~0.14ms dispatch floor and what kernel sizes amortize it, `pin()`
     for repeated calls, `metal_call_jit` for jit composition, and the
     thermal-measurement discipline (paired interleaved device-time
     comparisons) as advice to anyone benchmarking.
   - Extending: the `rule` registration API with a worked example.
3. **README rewrite** around the library story (currently it tells the
   exercise/ODE story): what it is, the subset, one attention-shaped
   example with the measured 2.4x-vs-JAX-CPU number and its caveats,
   links to the four docs pages.
4. **Example hygiene.** `examples/` stay, but each gets a one-line
   header saying which docs page it illustrates; the flash-attention
   example is the performance guide's companion.

Acceptance gate: a reader who has never seen the repo can go from
install to a working custom kernel using only README plus the four
pages (dry-run this by following them literally in a clean checkout).

## Sequencing and effort

Phases 1 and 2 are ordered; 3 and 4 can interleave after Phase 1 (the
docs need the contract to exist before describing it). Rough sizes,
calibrated against this project's history:

| phase | size | dominated by |
|---|---|---|
| 1 contract and diagnostics | 2-3 sessions | explain() plumbing through the coop walker; raise-site audit |
| 2 robustness | 2-3 sessions | dtype matrix and fuzzer grammar work |
| 3 versioning and API surface | small | changelog write-up |
| 4 documentation | 1-2 sessions | supported-subset table is the long pole |

Standing constraints, unchanged from the exploration phase: scored
kernels' emitted MSL stays byte-identical through refactors (hash
check), every behavior change lands with an oracle-backed test, and no
performance claim enters a doc without a paired interleaved
measurement behind it.

## Explicitly out of scope

- Whole-jaxpr lowering Tiers 1/2 (own plan, shelved).
- Cooperative cond/while, batched dot_general, coop Threefry, G=1
  unification: stay in `emitter-simplifications.md` behind their
  motivating-kernel gates. Production readiness does not grow the
  subset; it fences and documents the one that exists.
- Linux/x86, Intel Macs, Python < 3.12.
- Autodiff through `metal_call_jit` (documented as not differentiable;
  `custom_vjp` remains the user-side recipe).
