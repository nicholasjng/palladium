# Emitter feature sketches

Review follow-up, 2026-09-06. These are proposed designs, not implemented
features or supported-subset promises. The initial bug fixes accompanying
this note cover fractional bfloat16 literals, scalar/singleton reshapes,
used swap results, zero integer powers, and rejection of strided ref
accesses that previously generated contiguous addresses.

## 1. Typed lowering and preflight validation

Problem: primitive-name templates do not capture dtype-dependent semantics.
For example, integer reduce-max currently starts at `-INFINITY` and uses
`fmax`, and signed logical-right-shift emits signed `>>`.

Sketch:

- Keep the existing rule registry and Cursor. Add shared helpers for
  literal formatting, numeric conversion, bitcasts, binary operations,
  and reduction identities, selected by input/output dtype.
- Represent supported dtype/parameter combinations explicitly. Walk nested
  jaxprs before emission to reject unsupported combinations with primitive,
  shape, dtype, parameter, and source-location context.
- Define integer min/max/abs, logical shifts, oversized shift counts,
  division edge cases, and conversion boundaries against the installed
  JAX oracle. Do not assume floating-point templates are valid for integers.
- Treat NaN behavior, signed zero, half/bfloat rounding, and FAST versus
  SAFE behavior as part of the contract. Bitcasts that change element width
  need shape-aware lowering or an explicit rejection.

First increment: integer min/max/reduce-max and logical shift, plus typed
errors for unsupported output and intermediate dtypes. Tests should cover
integer extrema, values around 2**24, shift counts around 32, and fractional
low-precision constants. Run source checks without a device and semantic
checks on Metal in SAFE mode.

## 2. Explicit storage and strided views

Problem: CVal currently uses logical shape to decide whether its expression
is a scalar or indexable storage. Views also assume contiguous row-major
layout, while BlockInfo discards the positions of squeezed dimensions.

Sketch:

- Separate scalar expressions from storage-backed values. A storage view
  carries base identity, element offset, logical shape, element strides,
  address space, readonly status, and guaranteed alignment.
- Preserve full BlockSpec dimensions and squeezed-axis positions in the
  traced spec rather than reconstructing missing axes at the front.
- Centralize element addressing. Scalar extraction, reshape, broadcast,
  transpose, and ref indexing all use that accessor.
- Alias reshapes only when the physical layout permits it. Broadcasts
  use zero strides; transposes permute strides. Materialize otherwise.
- Make vectorized dot eligibility a query against layout and alignment.
  Introduce this incrementally without changing existing optimized paths
  until their source and performance have been checked.

First increment: storage identity and scalar/array distinction, followed
by general ref views. Gate on rows, columns, middle-axis squeezing,
singleton dimensions, indexed writes, and mutation after reads. Snapshot
semantics must survive alias optimizations and loop carry updates.

## 3. Masked loads/stores and edge blocks

Problem: irregular sizes need safe partial tiles; noncontiguous blocks also
prevent conventional two-dimensional tiling.

Sketch:

- Build strided blocks on the view representation above.
- Lower supported masked access primitives to per-element guarded reads
  and writes. A masked load's fallback value must match its frontend
  contract; the invalid address must never be dereferenced.
- Define BlockSpec boundary handling explicitly, using the installed
  Pallas interpreter as the reference for supported cases. Do not silently
  clamp addresses or assume all out-of-bounds reads return zero.
- Keep an unguarded path where a static proof shows the tile is in bounds.

Gate: sizes smaller than a block and sizes one element beyond a block
multiple, across axes and dtypes. Include sentinel outputs to prove masked
stores preserve untouched elements. Keep negative and dynamic index cases
separate from static edge-tile support.

## 4. Value-level structure and numerical primitive coverage

Problem: indexing an already loaded array can stage unsupported value-level
primitives even when equivalent ref indexing works. Scientific functions
also need a longer numerical vocabulary.

Suggested order:

1. `slice`, `dynamic_slice`, and `iota`: reuse view/addressing helpers,
   materializing only when needed. Preserve dynamic-slice boundary semantics.
2. `concatenate`, `rev`, and simple `pad`: explicit copy loops first.
3. `reduce_min`, `reduce_prod`, `reduce_any`, and `reduce_all`: share axis
   traversal with current reductions and use typed identities. Include
   empty reductions and empty axis subsets.
4. `log1p`, `expm1`, `rsqrt`, finite checks, remaining shifts, and rounding
   operations, driven by a real port rather than table size.
5. Common RNG transformations such as uniform and normal, after inventorying
   their staged primitives and checking key implementation assumptions.

Each increment gets one representative scientific kernel, isolated dtype
tests, and interaction tests inside scan/while/cond. Larger gathers,
scatter atomics, sorting, and generalized dot contractions remain separate
projects with their own semantics and measurements.

## 5. Elementwise fusion and storage diagnostics

Problem: chains of elementwise operations declare an array and emit a loop
for each intermediate. This can consume per-thread storage even when a
consumer could operate on each element immediately.

Sketch:

- Begin with pure, single-consumer chains sharing an iteration domain.
  Emit one loop with typed scalar temporaries for the intermediate values.
- Preserve rounding at each dtype boundary. In particular, half/bfloat
  intermediates cannot simply become unrestricted float expressions.
- Stop fusion at ref effects, control-flow boundaries, or multi-consumer
  values initially. Never move reads across writes based solely on string
  equality of CVal expressions.
- Estimate peak live storage separately from total declared bytes, and
  report the largest values and reasons for materialization in `explain`.
  These are estimates, not promises about Metal register allocation.
- Record whether dots vectorize and whether scan outputs stream, with a
  short reason for fallback. Keep diagnostics out of generated hot loops.

Gate: compare baseline and fused MSL on the existing ODE/SDE kernels;
measure paired runtime and compilation behavior on a Metal device. Reject
changes that only shorten source while increasing runtime or register
pressure. Retain the carry-permutation compiler regression tests.

## 6. Cooperative builtins and SIMD-group reductions

First repair the existing contract:

- Discover builtin requirements recursively from primitive use, independent
  of whether threadgroup scratch exists. Today `thread_index()` alone can
  reference an undeclared `_tid`.
- Either explicitly restrict groups to one dimension or implement linear
  thread index `x + size_x * (y + size_y * z)` and total group size as the
  product of the actual dimensions. Current `.x` emission does not match
  the documented linear/total meaning for multidimensional groups.
- Share geometry validation between eager and FFI paths. State which
  scratch bounds are checked and which remain the author's responsibility.
- Document uniform participation requirements for barriers and collectives;
  divergent per-thread control flow needs separate consideration when it
  contains synchronization.

Then add explicit SIMD-group sum/min/max primitives. Define participation,
supported dtypes, partial-group behavior, and numerical ordering before
choosing intrinsics. Compose larger reductions from SIMD-group partials,
shared storage, and a barrier only after the one-group version is tested.

Gate: explicit NumPy/JAX references rather than Pallas interpret, which
does not model cooperating threads. Test complete and partial groups,
multiple groups, and eager/FFI/vmap entry points. Benchmark against the
existing shared-memory approach before making any automatic substitution.

## 7. Validation infrastructure and remaining correctness work

- Replace the no-GPU module allowlist with explicit GPU test markers so
  pure tracing/emission checks in mixed modules run on every machine.
- Extend fuzz generation to integers, bools, low precision, scalar and
  singleton shapes, mixed indexing, and used swap results. Keep broad
  numerical edge cases separate from the current bounded float32 fuzzer.
- Add a recursive capability inventory fixture for the supported JAX
  version range, so new primitive parameters trigger review.
- Reject traced wrappers that contain computation around `pallas_call`
  until there is a plan for lowering that computation. Extracting the
  kernel from `lambda x: pallas_fn(x * 2) + 3` loses the wrapper operations.
- Audit alias-sensitive optimizations across nested jaxprs and preserve
  full BlockSpec axis metadata before broadening block support.

Suggested sequencing: remaining correctness repairs and typed validation;
storage/views and value-level indexing; masked blocks and primitive tail;
measured fusion; SIMD-group reductions. Whole-jaxpr lowering remains a
separate scope, described in `whole-jaxpr-lowering-plan.md`.

## 8. Captured arrays: frontend work required

Investigation on 2026-09-07 with JAX 0.11.1: Pallas's
`_trace_kernel_to_jaxpr` rejects captured non-Ref constants itself, before
palladium's emitter sees the kernel. Merely accepting jaxpr constvars in
the emitter cannot enable closure arrays through `metal_call`.

A future implementation needs a frontend that traces the kernel in its
grid/Ref environment, lifts captured arrays into hidden read-only operands,
and preserves those operands through eager and FFI dispatch. It should
handle constants captured by nested jit/control-flow bodies too, without
patching JAX globals or inspecting Python closure cells as a substitute for
tracing. Public argument arity, BlockSpec ordering, input/output alias
indices, and batching strides must remain consistent after lifting.

Own immutable snapshots per specialization, cache eager uploads, and keep
constants at zero batch stride in the FFI path. Define when captures are
snapshotted and how retracing sees changed captures. Do not put large
coefficient arrays into generated source by default.

Gates: captured coefficients versus explicit operands in eager, pinned,
FFI, jit, and vmap paths; distinct callable captures with identical shapes;
nested captures; buffer lifetime; alias index preservation; unsupported
constant dtype diagnostics. This remains deferred until the frontend
adapter is designed; explicit array operands still work today.

## 9. Initial feature increment (2026-09-07)

Permuted reshape now copies the permuted element order directly into the
final destination, using the same traversal as materialized transpose.
Identity permutations retain the ordinary reshape path. No intermediate
transposed array is allocated.

`select_n` now accepts int32/uint32 indices and arbitrary case counts,
including one case. A balanced comparison tree follows the installed
JAX lowering; scalar indices choose whole arrays and array indices choose
per element. Boolean selection retains its existing emission. Out-of-range
integer indices choose endpoint cases in palladium, but JAX's public
contract leaves this behavior implementation-defined.
