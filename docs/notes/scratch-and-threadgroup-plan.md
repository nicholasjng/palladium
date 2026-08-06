# Scratch operands and threadgroup emission: scoping plan

**Status update, 2026-08-25: Parts A and B are implemented.** Part B
landed on a design this document did not scope -- see "What actually
shipped" at the bottom, which supersedes B.0-B.6 below. The original
scoping is kept for provenance, including the premise that turned out
to be wrong.

**Scoping document, 2026-08-21.** Two related features, in dependency
order: per-invocation scratch Refs (small, self-contained), then
threadgroup-shared scratch + barrier emission (bigger, and it reuses
Part A's machinery rather than duplicating it). Nothing here is
implemented.

---

## Part A: per-invocation scratch (~100-150 lines incl. tests)

Pallas's `scratch_shapes=` kernel argument: extra `Ref`s appended to
the kernel jaxpr's invars past `*ins, *outs`, backed by no caller
array. Currently rejected outright — `trace.py`'s `_block_infos` never
reads `grid_mapping.scratch_avals`, so `KernelSpec` has nowhere to put
them, and `emit_msl` (`core.py:442-447`) catches the resulting
jaxpr-invar-count mismatch and raises `EmitError`. This tier is
*within one kernel invocation only*: allocated, used, and discarded
inside the single Metal thread running that program instance — no
cross-thread visibility, so none of Part B's hazards apply yet.

### A.1 `trace.py`: capture scratch avals (~20-30 lines)

- New `ScratchInfo` dataclass: `shape: tuple[int, ...]`, `dtype:
  np.dtype`. No `array_shape`/`index_map_jaxpr` — there is no backing
  array or BlockSpec, unlike `BlockInfo`.
- `trace()`: read `grid_mapping.scratch_avals`, build a
  `tuple[ScratchInfo, ...]`, add a `scratch` field to `KernelSpec`.

### A.2 `core.py`: allocate and bind (~30-40 lines)

- In `emit_msl`'s operand loop, after the input/output refs: for each
  `ScratchInfo`, emit a `thread`-space declaration the same way
  `declare()` already produces storage for computed intermediates
  (`thread {ctype} {name}[{size}];`), and append its `CVal` to
  `ref_vals`. No `[[buffer(k)]]` binding — scratch is never
  host-visible, so `dispatch.py`/`BoundKernel` need no changes.
- Relax the invar-count check to expect `len(operands) +
  len(spec.scratch)` instead of `len(operands)`.
- `_rule_get`/`_rule_swap` need no changes: they're already generic
  over any Ref-shaped `CVal`.
- Semantics: leave storage uninitialized (garbage), matching upstream
  Pallas's default (no `poison_buffers` support — out of scope).

### A.3 Tests (~50-80 lines)

- One kernel using `scratch_shapes` for a local accumulator, checked
  against the interpret oracle (matches the differential-fuzz pattern
  already used for control flow, per the recent fuzzer test additions).

---

## Part B: threadgroup-shared scratch + barrier emission

The interesting case from the "TPU all-reduce" conversation: real
communication between grid points requires Metal `threadgroup` memory
plus `threadgroup_barrier`, and only works *within one threadgroup* —
there is no cross-threadgroup barrier in a single dispatch. Scope this
to "reduce over a grid axis that fits in one threadgroup," not a
whole-grid all-reduce (that's a two-dispatch pattern, orthogonal to
the emitter — no new machinery needed for it beyond calling a
`BoundKernel` twice).

The design connects to Part A rather than sitting beside it:
threadgroup-shared scratch is still "extra Ref the kernel body reads
and writes," just declared in `threadgroup` address space instead of
`thread`. Reuse `ScratchInfo`/the declare-and-bind path; add what's
new (address space, barriers, and the size coupling below) around it.

### B.0 The real cost: threadgroup size becomes an emit-time fact

Today `threadgroup=` is a `bind()`-time knob (`dispatch.py:263`,
`BoundKernel.threadgroup`), decided *after* `emit_msl` has already
produced text, and the emitted MSL currently has zero knowledge of
it — the kernel body only ever reads `_pid`
(`thread_position_in_grid`). But a `threadgroup`-space array needs a
compile-time size, and a reduction needs to know exactly how many
threads are folding into it. That means threadgroup size must move
from a `bind()`-time parameter to something `trace()`/`emit_msl` see —
plumbed through `KernelSpec`, most likely — before any of the stages
below can be correct. This is the actual scoping risk in Part B; the
line counts past this point are the easy part.

### B.1 `CVal.space = "threadgroup"` (~20 lines)

Additive third value alongside `"thread"`/`"device"` (`core.py:82-85`
already documents `space` as the thing pointer-cast qualifiers key
off). Audit every `{cval.space}` f-string site (the vectorized
dot_general helpers cast on `rhs.space`/`lhs.space` today) to confirm
`"threadgroup"` produces valid MSL there too — most are already
space-parametric, so this should mostly just work.

### B.2 Kernel signature: expose thread-in-group position (~10-20 lines)

Add `uint3 _tid [[thread_position_in_threadgroup]]` (and
`_group_size`/`threads_per_threadgroup` if the reduction loop needs an
explicit bound rather than the emit-time constant from B.0) alongside
the existing `uint3 _pid [[thread_position_in_grid]]` in `emit_msl`'s
`params` list. Mirrors the existing `_PID = ("_pid.x", ...)` tuple.

### B.3 Explicit barrier, not auto-inferred (~30-40 lines)

Auto-inserting barriers from a hazard analysis (write-then-read across
threads) is real analysis work and easy to get subtly wrong. Start
with a `palladium`-specific traced marker — a zero-output primitive
the kernel author calls explicitly, lowered verbatim to
`threadgroup_barrier(mem_flags::mem_threadgroup)` wherever it appears
in the body — same division of responsibility as hand-written
Metal/CUDA (the programmer places barriers, not the compiler). Defer
automatic placement to a later pass once there's real kernel usage to
learn the hazard patterns from.

### B.4 Threadgroup-tagged scratch request (~40-60 lines)

Pallas's own `MemorySpace` enum (`ANY`/`DEFAULT`/`ERROR`/`INDEX`/`KEY`)
has no "threadgroup" member, and hijacking one of those to mean
something Metal-specific would be misleading on every other backend.
Cleaner: a palladium-specific scratch-request helper outside Pallas's
enum entirely, extending `ScratchInfo` from Part A with a
`space: Literal["thread", "threadgroup"]` tag. `core.py` declares
`threadgroup`-tagged entries as `threadgroup {ctype} {name}[{size}];`
using the B.0 emit-time size instead of `thread`-space arrays.

### B.5 Extent/threadgroup-size contract check (~20-30 lines)

Validate that the reduced grid axis's extent matches (or evenly
divides, if the kernel strides) the emit-time threadgroup size from
B.0. Getting this wrong is a silent-wrong-answer bug (partial
reductions, not a crash), so this should be a loud `EmitError` at
trace/emit time, not a runtime surprise from the Metal compiler or
(worse) wrong numbers with no error at all.

### B.6 Test kernel (~40-60 lines)

A block-sum or block-max reduction over one grid axis, checked against
the interpret oracle, plus a case that deliberately mis-sizes the
threadgroup to confirm B.5's check actually fires.

---

## Summary

| Part | Rough size | Real cost driver |
|------|-----------|-------------------|
| A: per-invocation scratch | ~100-150 lines | none — mechanical, self-contained |
| B: threadgroup scratch + barriers | ~150-250 lines | B.0's threadgroup-size coupling (trace/emit must learn about a currently dispatch-only knob) and B.4's API design (how a kernel author requests threadgroup-space scratch) — both design decisions, not line count |

Do Part A first regardless: it's needed either way (threadgroup
scratch reuses its declare-and-bind path), and it's a good exercise in
isolation before Part B's cross-thread correctness questions.


---

## What actually shipped (2026-08-25)

Part B landed as **static threadgroup arrays plus a dynamic loop bound**,
not the emit-time-constant threadgroup size B.0 assumed. Three
measurements drove the change.

**1. The B.0 premise was wrong.** B.0 asserted that threadgroup size
"must move from a `bind()`-time parameter to something `trace()`/`emit_msl`
see". It doesn't. A scratch request's shape is the *shared* array's size
and does not scale with thread count, so the allocation was already
emit-time knowable from `ScratchInfo` alone. What genuinely needs to be
dynamic is the *reduction loop bound*, and MSL supplies that at runtime
via `[[threads_per_threadgroup]]`. The two were conflated.

**2. Non-uniform dispatch makes the dynamic bound the correct choice,
not merely the convenient one.** Dispatching 100 threads with
`threadgroup=32`, `[[threads_per_threadgroup]]` reports `[32,32,32,32,4,4]`
-- the true size of the partial tail group. A baked-in constant folds in
slots no thread ever wrote. This *inverts* B.5: rather than needing a
divisibility check to stay correct, the hardware handles the tail and
the residual contract is one-sided (group size must not exceed the
declared extent). That check lives in `bind()`, which is the only place
that sees both the spec and the threadgroup value.

**3. Dynamic `[[threadgroup(i)]]` allocation was unnecessary.** It would
have required plumbing `threadgroup_memory` byte sizes through
`native/ffi/palladium_ffi.cpp` as variable-length XLA FFI attrs. Static
declaration avoids that entirely, keeps `emit_msl` free of any coupling
to dispatch, and produces a known-size object the Metal compiler can
reason about.

Consequences for the rest of the plan:

- **B.1** was a one-word change: `ScratchInfo` gained a `space` tag and
  `emit_msl` interpolates it as the declaration qualifier. Both spaces
  are compile-time-sized local arrays; nothing else differs. The claim
  that threadgroup scratch "reuses Part A's declare-and-bind path" was
  more true than scoped, not less.
- **B.2/B.4** the request API is `palladium.threadgroup_memory`, built on
  `pl.MemoryRef`'s `memory_space` field being typed `Any` upstream, so a
  palladium sentinel rides through Pallas tracing untouched. No
  `pl.MemorySpace` member was hijacked, as intended.
- **B.3** the barrier is an effectful JAX primitive. The effect is
  load-bearing: without it JAX's DCE removes the zero-output primitive
  from the kernel jaxpr before palladium sees it. Registration in
  `control_flow_allowed_effects` and a no-op CPU MLIR lowering are both
  required for `interpret=True` to run at all.

### Two things the tests cannot check

**The interpret oracle does not apply.** Interpret runs program
instances sequentially with no threadgroup concept, so it models each as
a group of one. A cooperative kernel's interpret result is a *different
computation*, not a reference for its GPU result. Cooperative kernels are
validated against NumPy references instead.

**A missing barrier is not observable here.** Deleting the barrier
emission entirely still produced correct results at every threadgroup
size tried, up to 1024. The race is latent, not observable, so no
numerical assertion catches it and the MSL-text assertion in
`test_23_threadgroup.py` is the only real guard. It is marked
non-deletable for that reason.

### The jax.ffi path (resolved same day)

Initially scoped as needing new FFI attrs plus C++. That was wrong: the
native handler has always bound `threadgroup_x/y/z` and set them on the
launch descriptor, so the restriction was purely that `metal_call_jit`
never accepted a `threadgroup=` kwarg to forward. Landed as Python only
-- the kwarg, the same explicit-threadgroup check `bind()` enforces, and
a shared `normalize_threadgroup` helper so the two paths cannot
normalize the knob differently. The stale `(32,1,1)` comment in
`palladium_ffi.cpp` was reconciled at the same time.

Cooperative kernels now run on both paths, including nested in
`jax.jit`.

### Still open

- **`jax.vmap` over a cooperative kernel is unexercised.** The FFI path
  only permits the sequential vmap methods, which dispatch once per
  batch element, so each element gets its own threadgroups and the model
  should hold -- but no test covers it.
- **The extent contract is one-sided and unenforced.** `bind()` and
  `ffi.py` require an *explicit* threadgroup; neither checks it against
  the declared extent, because palladium cannot tell which scratch array
  a kernel indexes by `thread_index()`. Exceeding the extent is still
  the author's error to make, the same as in hand-written Metal.
- **Cross-threadgroup communication remains a two-dispatch pattern.**
  Metal has no device-wide barrier; nothing here changes that.
