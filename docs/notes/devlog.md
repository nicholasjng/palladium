# Devlog

Per-card narrative from the tutor arc: what landed, why it was next, and
the one decision worth remembering. Status checkboxes live in
`.tutor/progress.md`; this is the prose companion. Newest at the bottom.

## A — Per-invocation scratch operands   (2026-08-24)

**Done:** `scratch_shapes=` kernels now trace, emit, and run. `trace.py`
lifts `grid_mapping.scratch_avals` into a new `ScratchInfo`
(`shape`/`dtype` only) on `KernelSpec.scratch`; `emit_msl` declares each
entry as a `thread`-space local and appends its `CVal` to `ref_vals`,
with the invar-count gate widened to operands + scratch. Coverage in
`tests/test_22_scratch.py`, diffed against the interpret oracle.

**Why:** unblocks Part B — threadgroup-shared scratch is the same
declare-and-bind path in a different address space, so the plumbing was
worth settling while no cross-thread hazards were in play.

**Note:** scratch is *not* an operand — no `[[buffer(k)]]`, no signature
entry, and `dispatch.py`/`BoundKernel` needed zero changes. That is what
keeps the change to ~40 lines. Scalar (shape-`()`) scratch keeps shape
`()` rather than borrowing the operand path's axis-1 convention: operands
are pointers and scratch is not, so `CVal.at()` absorbs the rank-0 case
directly. Storage is left uninitialized, matching upstream Pallas.
`_rule_get`/`_rule_swap` needed no changes, confirming they were already
generic over Ref-shaped `CVal`s — including ref stores inside a
`fori_loop` body.

## B — Threadgroup-shared scratch, barriers, cooperative reductions   (2026-08-25)

**Done:** `palladium.threadgroup_memory(shape, dtype)` requests
`threadgroup`-space scratch; `barrier()`, `thread_index()`, and
`threads_per_threadgroup()` are the cooperative primitives around it.
`ScratchInfo` gained a `space` tag, `emit_msl` interpolates it as the
declaration qualifier and conditionally adds the two position builtins,
and `bind()` requires an explicit `threadgroup=` for cooperative kernels.
19 tests in `tests/test_23_threadgroup.py`, sabotage-checked five ways.

**Why:** the shared tier of Part A's scratch work, and the first time
palladium emits a kernel whose threads are not independent.

**Note:** the plan's B.0 premise (threadgroup size must become an
emit-time constant) was wrong and the shipped design differs — see "What
actually shipped" in `scratch-and-threadgroup-plan.md`. Three things
worth remembering. The barrier's JAX *effect* is load-bearing: without it
DCE deletes the primitive before palladium sees the jaxpr. Metal rejects
a signature mixing scalar and vector thread-position builtins, so `_tid`
and `_tpt` are `uint3` because `_pid` is. And the interpret oracle does
not apply to cooperative kernels at all — interpret models each instance
as a threadgroup of one, so the reference has to be NumPy.

## B.5 — Cooperative kernels on the jax.ffi path   (2026-08-25)

**Done:** `metal_call_jit(..., threadgroup=N)` now dispatches kernels
using `threadgroup_memory`, including nested in `jax.jit`. The blanket
rejection became the same explicit-threadgroup check `bind()` enforces,
and `normalize_threadgroup` in `diagnostics.py` is now the single place
the `int | tuple | None` knob is normalized for all four entry points.

**Why:** the FFI path is the production surface; leaving cooperative
kernels eager-only would have made threadgroup memory a demo feature.

**Note:** this was scoped as "new FFI attrs plus C++" and turned out to
be Python-only — `palladium_ffi.cpp` had bound `threadgroup_x/y/z` and
set them on the launch descriptor all along. Worth remembering as a
pattern: the native handler is more capable than the Python that calls
it, so check the C++ before scoping C++ work.

## API ergonomics pass   (2026-08-25)

**Done:** six consumer-facing additions, none of which change emitted
code. `verify()` on both callables (differential check against the
interpret oracle or an explicit `reference=`, returning the GPU output);
per-instance storage accounting surfaced through `explain()` and the
stack-overflow error; a `"simdgroup"` threadgroup sentinel plus device
budget checks; `.primitive` on `UnsupportedPrimitiveError` and a new
`StackOverflowError` carrying `.stack_bytes`; `FfiCallable.pin` refusing
with a reason; and LRU bounds on all three kernel caches (the native
pipeline cache, and the Python spec caches on both callables).

**Why:** the interpret-oracle diff was hand-rolled in ~40 places across
the suite, and the two facts that make it correct (FAST-math tolerance,
and that cooperative kernels have no interpret oracle) lived only in
prose. `verify()` is where that knowledge now lives.

**Note:** two bugs found by writing the tests rather than the code. The
`"simdgroup"` sentinel resolved in diagnostics but reached
`mr.Batch.add` as a raw string, because `bind()` normalized only for the
*check* and stored the raw value -- normalization now happens once, at
the top of `bind`. And `metal_call_jit` silently forwarded `cache_size`
to `pallas_call`. Also worth remembering: the stack figure is not
liveness-aware, but neither is Metal's own pipeline check, so it
over-counts in exactly the same places.
