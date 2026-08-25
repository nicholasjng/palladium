# Supported subset

The contract for what a Pallas kernel may contain. Anything outside it
raises a typed error at trace or emit time (never wrong code); see the
error taxonomy in `docs/getting-started.md`.

## Kernel structure

| Construct | Status |
|---|---|
| Grids | Up to rank 3 (`Metal grids are 3D`); gridless calls run one instance |
| BlockSpecs | Contiguous blocks only: every dim after the first must cover its array dim (only the leading block dim may be partial). Dims may be ints, `pl.Blocked`, or `pl.Squeezed`; `pl.Element`, `pl.Indirect`, and `pl.BoundedSlice` are rejected |
| Index maps | Any jaxpr over grid indices (they are lowered, not pattern-matched) |
| Ref indexing | `x_ref[...]`, integer and `pl.dslice` indices, slices with unit stride; a partial slice only on the first kept dim |
| Multiple outputs | Yes |
| `scratch_shapes` | Yes. `pl.MemorySpace.*` requests are `thread`-space (private to one program instance); `palladium.threadgroup_memory(shape, dtype)` requests `threadgroup`-space storage shared across the group |
| `PrefetchScalarGridSpec` | Rejected |
| Captured arrays (jaxpr constvars) | Rejected; close over Python scalars or pass arrays as operands |
| Multiple `pallas_call`s per traced function | Rejected; trace them separately |

## Dtypes

| dtype | Status |
|---|---|
| float32 | Full support |
| float16 | Load/store, elementwise, reductions, dots; `preferred_element_type=float32` accumulates in f32 |
| bfloat16 | Load/store, elementwise, reductions. NumPy cannot pass ml_dtypes arrays over DLPack, so eager `metal_call` ships the bytes as uint16 and relabels the buffer; a lossless, measured-free reinterpretation |
| int32, uint32 | Load/store, elementwise, bitwise, reductions |
| bool | Load/store, logical ops, `select_n` predicates |
| float64, int64 | Rejected with a hint (`jax_enable_x64` is the usual cause); Metal has no doubles |

## Primitives

Elementwise: `add sub mul div min max pow neg abs exp log sin cos sqrt
tanh integer_pow clamp select_n convert_element_type
bitcast_convert_type`, comparisons (`lt le gt ge eq ne`), logical and
bitwise `and or xor not`, `shift_right_logical`. Operands broadcast
NumPy-style against the output shape.

Structure: `broadcast_in_dim`, `reshape` (row-major reinterpretation;
a `dimensions` permutation is rejected), `transpose` (materialized
copy, except a rank-2 transpose consumed only as a dot rhs, which fuses
lazily and costs nothing), `program_id`, inlined `jit` calls.

Matmul: `dot_general` with the standard contraction (lhs dim 1 against
rhs dim 0), operands up to rank 2. Rank-1 operands canonicalize, so
`jnp.dot` matvec/vecmat/vecvec all work. Batch dims and other
contractions are rejected. `preferred_element_type` is honored through
the output dtype.

Reductions: `reduce_sum`, `reduce_max` over any axis subset.

Cooperative: `palladium.barrier()` (lowers to
`threadgroup_barrier(mem_flags::mem_threadgroup)`, placed by the author,
never inferred), `palladium.thread_index()`, and
`palladium.threads_per_threadgroup()`. Kernels using
`threadgroup_memory` must be dispatched with an explicit `threadgroup=`
size no larger than the declared extent. Both `metal_call` and
`metal_call_jit` accept it; leaving it unset is a loud error on either
path, since a runtime-chosen size is commonly larger than the declared
extent and indexing past it corrupts silently. `jax.vmap` composes over
cooperative kernels under every supported `vmap_method`: each batch
element is dispatched with the kernel's own grid and threadgroup, so
batching does not move threadgroup boundaries.

Control flow: `lax.fori_loop` and full `lax.scan` (scanned xs, stacked
ys, and `reverse=True`), `lax.while_loop` (data-dependent, divergent
trip counts are fine), `lax.cond`/`lax.switch` (clamped index
semantics). Stacked ys live on the per-thread stack unless their only
use is a full-block store to an otherwise untouched output ref, in
which case they stream straight to device memory with no stack cost:
the dense-output stepper idiom `_, ys = lax.scan(step, y0, ts);
o_ref[...] = ys` takes the streaming path.

RNG: `jax.random.bits` and `jax.random.fold_in` (Threefry-2x32-20,
bit-exact against jax, 32-bit widths), plus `wrap_key_data`/`key_data`.

Anything else raises `UnsupportedPrimitiveError` naming the primitive;
`docs/extending.md` covers adding a rule.

## Execution model

One Metal thread per program instance, always. Parallelism comes from
the Pallas grid; a divergent trip count in `while_loop`/`cond` inside
one instance costs nothing correctness-wise, since threads are
independent.

Threads stop being independent once a kernel uses `threadgroup_memory`:
those instances communicate within a threadgroup, and ordering is the
author's responsibility via `barrier()`. There is no device-wide
barrier, so cross-*threadgroup* communication still needs two dispatches
(see `examples/04_reaction_diffusion.py`). `interpret=True` cannot model
any of this -- it runs instances sequentially and reports
`thread_index() == 0`, `threads_per_threadgroup() == 1`, so a
cooperative kernel must be validated against a reference implementation
rather than the interpret oracle.

## Checking your kernel

`call.verify(*args)` runs the kernel and diffs it against a reference,
defaulting to this kernel's own `interpret=True` oracle with tolerances
that allow for FAST math. It returns the GPU output, so it can replace
a call site directly. Kernels using `threadgroup_memory` must pass an
explicit `reference=`: interpret has no threadgroups and models each
program instance as a group of one, so the oracle would be comparing
against a different computation.

`call.explain(*args)` reports launch geometry plus the per-instance
storage the kernel declares -- `thread_bytes` (the per-thread stack
figure to shrink when pipeline creation fails) and, for cooperative
kernels, `threadgroup_bytes` against the device budget. It emits MSL to
measure; it compiles and dispatches nothing.

## Hard limits

- Per-thread stack: loaded blocks and intermediates are thread-local
  arrays, so per-instance block sizes are bounded by the per-thread
  stack (roughly a few KB of live arrays). Oversized kernels fail at
  compile time with an `EmitError` saying to shrink the blocks via the
  grid.
- FAST math (the default) reorders float arithmetic and uses
  approximate transcendentals; see the tolerance note in
  `docs/getting-started.md`.
