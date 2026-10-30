# Supported JAX functionality

Palladium traces one Pallas kernel and lowers it to MSL. It requires
`jax>=0.11,<0.12`; JAX 0.11.2 is tested. Cases outside the subset below
raise an error during tracing, lowering, or compilation.

## Pallas kernels

| Feature | Supported behavior |
|---|---|
| Kernel entry | One `pallas_call` per traced callable; multiple calls and `PrefetchScalarGridSpec` are rejected |
| Grid | Rank 1–3, or gridless for one program instance |
| Blocks | Contiguous or strided `BlockSpec` tiles, squeezed axes, partial edge tiles; `pl.Element`, `pl.Indirect`, and `pl.BoundedSlice` are unsupported |
| Ref indexing | Full refs, scalar indices, `pl.dslice`, positive static slice strides |
| Scratch | Per-instance `pl.MemorySpace` scratch |
| Captures | Python scalar constants; captured arrays are unsupported, so pass them as operands |
| Outputs | Multiple outputs and Pallas input/output aliases, subject to backend limitations |

Out-of-range edge lanes are guarded: invalid loads use Pallas-compatible
padding values and invalid stores do nothing. Arbitrary out-of-range indices
are not supported. Strided and guarded reads materialize contiguous values;
read snapshots are preserved across later writes.

## JAX operations in kernels

- **Elementwise:** add/subtract/multiply/divide, power, abs/negation,
  exp/exp2/expm1/log/log2/log1p, sqrt/rsqrt, sin/cos/tan and their inverses,
  sinh/cosh/tanh and their inverses, atan2, erf/erf_inv, logistic, square,
  floor/ceil/round, is_finite, sign/remainder/clamp, min/max, comparisons,
  casts, bitcasts, logical and bitwise operations, shifts, and broadcasting.
  erf, erf_inv, expm1, and log1p are float32 helper functions emitted with
  the kernel; the rest map onto MSL builtins.
- **Array structure:** reshape, permuted reshape, transpose, selection via
  `select_n`, `broadcasted_iota`/`arange`, `concatenate`, and `dynamic_slice`
  (start indices clamped as JAX defines).
- **Cumulative:** `cumsum`, `cumprod`, `cummax`, `cummin` along one axis,
  forward or reverse.
- **Reductions:** sum, min, and max over supported axis subsets, and
  `lax.reduce` with any combiner over several operands, such as `jnp.minmax`
  or a value-and-index argmax.
- **Dot:** standard rank-2 matrix multiplication and rank-1 matvec, vecmat, or
  vecvec. Batched dots and other contraction axes are rejected.
- **Control flow:** `fori_loop`, `scan` (including reverse scans and stacked
  outputs), `while_loop`, `cond`, and `switch`.
- **Random:** 32-bit `random.bits` and `fold_in`, plus key-data conversion,
  using Threefry-2x32-20.
- **Effects:** Ref reads and writes. Debug printing and arbitrary JAX
  effects are unsupported.

`CompilerParams(program="threadgroup")` runs each program as one threadgroup
over a block of rows instead of one thread. The kernel stays plain Pallas on
a 1D grid: refs are `(rows, columns)` blocks spanning all columns, `(rows,)`
per-row vectors, `(columns,)` vectors such as weights, or per-tile outputs:
`(1, columns)` blocks of a `(programs, columns)` array and `(1,)` blocks of a
`(programs,)` array. Full-block reads and stores, elementwise operations,
broadcasts, reshapes by unit dimensions, sum, max or min over the columns,
and `cumsum`, `cumprod`, `cummax` or `cummin` along the columns (forward or
reverse, for rows of at most 16 elements per thread) are supported. Slices,
pads and concatenations along the columns, as in `jnp.roll`, `jnp.pad` and
`jnp.diff`, read shifted columns; scan results cannot be shifted. Sum, max
or min over the rows of a tile (or over the whole tile) accumulate across
the rows and must go to per-tile outputs; tiles must divide the rows, and
values that vary by row cannot be combined with these results. Nested `jit`
and `custom_jvp_call` (as in `jax.nn.softmax`) are inlined. Broadcasts or
shifts across rows, indexed reads and other primitives raise EmitError.
Each block is read into registers once per row, so a row softmax, layer norm
or cumsum reads and writes its data once. On an M2 they ran at 91 GB/s, about
98% of the copy bandwidth and within 2% of MLX's fused kernels and
`mx.cumsum` (`benchmarks/bench_row_programs.py`).

On both paths, tiled matmuls on a 2D or 3D grid lower through Metal 4
TensorOps by default (`CompilerParams(dot_general="auto")`); other dots keep
the one-thread-per-program emitter. `dot_general="tensorops"` requires the cooperative lowering and
raises where it does not apply; `dot_general="default"` forces the primitive
path. `dot_general="simdgroup"` also requires the cooperative lowering, and
emits float32 matmuls with `simdgroup_matrix` instead of Metal 4 `matmul2d`
when the operands are not transposed, M and N divide evenly into tiles, K is
a multiple of 16, the tile has 16, 32, 64 or 128 rows and a multiple of 16
columns, and at most 8192 elements. Everything else keeps `matmul2d`. On an
M2 this ran float32 4096³ at 2.54 TFLOPS against 1.97 for `matmul2d` with
the same 64x64 tiles; see `notes/apple-gpu-kernel-recipe.md`. The lowering covers a standalone, tiled rank-2 matmul or matching rank-3
batches of matrices. The batched form uses arrays
`[B, M, K] @ [B, K, N]`, one batch per Pallas grid coordinate, and one
threadgroup per output tile. Batch sizes must match and each Pallas tile must
contain the full K dimension. Output tile dimensions must be multiples of 16;
partial M/N edge tiles are masked. The generated matmul processes K in internal
chunks and handles its tail. Matching float32, float16, and bfloat16 buffers are
supported, with float accumulation and narrowing stores for half/bfloat outputs.
Rank-2 inputs may be transposed. Recognized scalar elementwise epilogue chains,
such as ReLU or tanh, can be fused; same-shape residuals and rank-2 column bias
are also supported. Batch broadcasting, split-K across Pallas programs, and
arbitrary `dot_general` batch dimensions are not recognized. This is a grid of
2D products, not a batched `dot_general`
primitive. The attention form requires rank-4 `[batch, sequence, heads, D]`
buffers and a static key-tile scan; query and key/value lengths may differ for
noncausal attention. Causal attention requires matching lengths. Q/output use
query tiles, while K/V are full-sequence blocks. One Metal threadgroup computes
each query tile and runs the score and value matmuls with Metal TensorOps. Tile
sizes and head dimensions must be multiples of 16, and both sequence lengths
must divide evenly into their tiles. Short query sequences need padding and
output cropping. Unsupported jaxpr forms and layouts raise `EmitError`.
The mps lowering follows the same policy: cooperative kernels reach jax-mps
with their scaled thread grid and required threadgroup, and run under `jax.jit`
on the `mps` platform (tested on an M1 Pro with JAX 0.11.2 and the `palladium.dispatch` handler
from the `palladium-dispatch` branch of jax-mps).

The backend supports float32, float16, bfloat16, int32, uint32, and bool, with
operation-specific limits:

| Type | Supported operations |
|---|---|
| float32 | All listed arithmetic, reductions, and dots |
| float16 | Load/store, elementwise, reductions, and dots; f32 accumulation when requested |
| bfloat16 | Load/store, elementwise, reductions, and cooperative TensorOps matmul |
| int32, uint32 | Load/store, arithmetic, bitwise operations, shifts, and reductions |
| bool | Load/store, logic, and selection |

float64 and int64 are rejected. `preferred_element_type` controls
dot accumulation through the output dtype. Under SAFE math, floating extrema
preserve NaNs and signed-zero ties; FAST permits compiler reassociation and
does not promise those IEEE edge semantics. See [performance guidance](performance.md)
for numeric comparisons and measurement caveats.

## JAX transformations and backends

| Path | JIT | vmap | Gradients |
|---|---|---|---|
| `pl.pallas_call` on `mps` | Yes: palladium is the registered Pallas backend for the platform (`palladium.CompilerParams` for options) | JAX's own `pallas_call` batching | `palladium.with_vjp` |
| `metal_call` | Yes, through CPU FFI to Metal | One FFI call per batch; nested levels run one element at a time | `palladium.with_vjp` or `jax.custom_vjp` |

The mps lowering supports FAST math only. On other platforms a `pallas_call`
keeps JAX's own behavior: `interpret=True` runs the Pallas interpreter.
`metal_call` dispatches to Metal through the CPU FFI target and does not need
jax-mps.

Unsupported primitives raise `UnsupportedPrimitiveError`; unsupported
primitive cases raise `EmitError`. Unsupported Pallas call structure raises
`TraceError`. See [getting started](getting-started.md)
for diagnostics and examples.
