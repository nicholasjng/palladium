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
| `scratch_shapes` | Rejected |
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

Control flow: `lax.fori_loop` and pure-carry `lax.scan` (stacked ys,
scanned xs, and reverse are rejected; carry avals must match),
`lax.while_loop` (data-dependent, divergent trip counts are fine),
`lax.cond`/`lax.switch` (clamped index semantics).

RNG: `jax.random.bits` and `jax.random.fold_in` (Threefry-2x32-20,
bit-exact against jax, 32-bit widths), plus `wrap_key_data`/`key_data`.

Anything else raises `UnsupportedPrimitiveError` naming the primitive;
`docs/extending.md` covers adding a rule.

## Execution model

One Metal thread per program instance, always. Parallelism comes from
the Pallas grid; a divergent trip count in `while_loop`/`cond` inside
one instance costs nothing correctness-wise, since threads are
independent.

## Hard limits

- Per-thread stack: loaded blocks and intermediates are thread-local
  arrays, so per-instance block sizes are bounded by the per-thread
  stack (roughly a few KB of live arrays). Oversized kernels fail at
  compile time with an `EmitError` saying to shrink the blocks via the
  grid.
- FAST math (the default) reorders float arithmetic and uses
  approximate transcendentals; see the tolerance note in
  `docs/getting-started.md`.
