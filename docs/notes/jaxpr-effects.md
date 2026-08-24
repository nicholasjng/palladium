# Jaxpr effects: what they offer palladium codegen

Status: items 1, 3, and 4 implemented; items 2 and 6 corrected after
reading the emitter, see below. (2026-08-25)

## The situation

Every jaxpr equation and every jaxpr carries an `effects` frozenset. For a
Pallas kernel jaxpr the relevant members are the state effects from
`jax._src.state.types`: `get` carries `ReadEffect(ref)`, `swap` carries
`WriteEffect(ref)`, `addupdate` carries `AccumEffect(ref)`, each keyed to a
specific `Ref` var. Verified on a palladium trace: the per-eqn sets arrive
exactly like that, and the jaxpr-level set aggregates them, including up
through control flow. A `swap` buried in a `scan` body surfaces in the outer
equation's effects, so may-read/may-write sets per ref come for free, with
no recursive sub-jaxpr walk.

Before item 1 landed, palladium ignored all of it: ref classification is
still positional (`readonly=k < n_in` in `emit/core.py`), no emitter pass
consumes the write sets yet, and an effectful primitive we cannot honor
used to fail only at emit time as a generic unsupported primitive.

Caveat: the effect types are private API, the same fragility class as the
`NDIndexer` import in `emit/core.py`. Same mitigation: isolate the imports
in one shim module, lean on the `jax<0.12` pin, and let the JAX-head canary
catch breakage once CI exists.

## Items, in suggested order

### 1. Trace-time gate on foreign effects (done)

`trace()` rejects any kernel jaxpr whose effects contain non-`RefEffect`
members, as a `TraceError` naming the effect types. The private imports
live only in `palladium/effects.py`, which also provides the
`read_refs`/`written_refs` sets items 2-5 build on. Verified: both
`jax.debug.print` and `pl.debug_print` arrive as `DebugEffect` and are
caught at trace time.

### 2. Loop hoisting and alias relaxation (re-scoped, smaller than first thought)

The original sketch assumed every `get` copies. Reading the emitter
corrected that: indexed gets of read-only (input) refs already bind the
pointer view with zero copy (`emit/rules.py`, `_rule_get`). What still
copies is (a) full-block gets, deliberately, as a per-thread cache, and
(b) any get of a ref palladium considers writable, i.e. every output ref
(classification is positional, `readonly=k < n_in` in `emit/core.py`).

The remaining effect-driven wins are therefore narrower:

- (done, the scan-xs case) a full-block get consumed only as scan xs
  binds the ref directly instead of copying: each element is read once
  per scan, so the copy bought nothing and cost stack. See
  `_consumed_only_as_scan_xs` in `emit/rules.py`; consts keep the cache
  copy on purpose.
- full-block loads repeated inside a loop body could be hoisted when the
  body has no `WriteEffect` on that ref and the load is un-indexed;
- gets of output refs could alias instead of copy when no write to that
  ref follows while the view is live. Effects give the may-write sets
  (including through sub-jaxprs, via the outer eqn's aggregated
  effects), but the liveness half needs the emitter's def-use info, and
  a swap that consumes the view is itself a same-eqn hazard.

The rest is worth doing when a profile shows the copies.

### 3. In-place kernels via input_output_aliases (done)

`pallas_call(..., input_output_aliases=...)` is now honored end to end:
the eager path binds the slot's input buffer as the aliased output (the
fresh output allocation disappears; `PendingResult.wait` returns a copy
instead of a live view of the shared buffer, and `pinned()` is
rejected), and the ffi path forwards the pairing to
`jax.ffi.ffi_call(..., input_output_aliases=...)` so XLA may donate.

Two findings sharpened the safety conditions beyond the sketch:

- The interpret oracle does NOT alias inside the kernel body: the input
  ref keeps its pre-call values throughout, verified directly. One
  shared buffer therefore only matches the oracle when every read of
  the input executes before the first write of the aliased output.
  `_validate_aliases` checks exactly that on effect order over the
  kernel eqns (sub-jaxprs included via their aggregated effects) and
  rejects the rest as a typed TraceError. This is the effects payoff
  the sketch predicted, in a different place than predicted.
- The aliased pair must slice the shared buffer identically: same array
  shape, dtype, block shape, and BlockSpec index map (compared by their
  jaxpr pretty-print, which normalizes var names).

The emitter also drops the readonly view optimization for aliased
inputs (`emit/core.py`): a bound view could otherwise observe the
in-place write after the ordering check passed at eqn granularity.
Tests: `tests/test_19_io_aliasing.py`.

### 4. Parallel-safety validation (done, best-effort by design)

`_validate_parallel_writes` in `trace.py` rejects the provable case: a
grid axis of extent > 1 that neither the output's BlockSpec index map
nor any top-level write index depends on, so every instance along that
axis writes identical locations. That catches the classic silent race
(default whole-array out spec plus a grid, full-block write), including
the default constant-zero index map pallas installs when out_specs is
omitted. Deliberately unchecked, erring permissive: writes hidden in
sub-jaxprs (found via the eqn's aggregated WriteEffects, then skipped
rather than guessed at), and non-injective maps that do use the axis
(injectivity of an arbitrary jaxpr is not decidable here). Tests:
`tests/test_20_write_races.py`.

### 5. Hazard-tracked batching (later, dispatcher-level)

The pipelined FFI loop uses one command buffer per batch element. Write/
read sets would let the handler encode the whole batch into a single
`CommandBatch` with a concurrent encoder and barriers only where write
sets intersect: one commit, the ~130us queue floor amortized over the
batch. For vmap the independence already follows from vmap semantics; the
effect-derived sets are what make the same optimization safe for general
device-resident kernel chains (barrier iff writes(A) intersects reads(B)).

Do this together with device-resident chaining on the dispatcher side:
`BoundKernel.launch` accepting `mr.Buffer`s or a `PendingResult`'s
outputs as inputs, skipping the upload and the host round-trip between
chained kernels. Both features hang off the same write/read-set hazard
logic and the same CommandBatch encoding, so they are one work package,
not two.

### 6. AccumEffect lowering (moot for now)

Verified: `o_ref[...] += x` in a Pallas kernel decomposes to
get/add/swap before palladium ever sees it, so `addupdate` (and with it
`AccumEffect`) is unreachable from ordinary kernel code. Revisit only if
atomics (`pl.atomic_add`) come into scope.
