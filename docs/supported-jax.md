# Supported JAX functionality

Palladium traces one Pallas kernel and lowers it to MSL. It requires
`jax>=0.11,<0.12`; JAX 0.11.2 is tested. This page describes
the supported kernel language and how each call path composes with JAX. Cases
outside this subset raise an error during tracing, lowering, or compilation.

## Pallas kernels

| Feature | Supported behavior |
|---|---|
| Kernel entry | One `pallas_call` per traced callable; multiple calls and `PrefetchScalarGridSpec` are rejected |
| Grid | Rank 1–3, or gridless for one program instance |
| Blocks | Contiguous or strided `BlockSpec` tiles, squeezed axes, partial edge tiles; `pl.Element`, `pl.Indirect`, and `pl.BoundedSlice` are unsupported |
| Ref indexing | Full refs, scalar indices, `pl.dslice`, positive static slice strides |
| Scratch | Per-instance `pl.MemorySpace` scratch and explicit shared `palladium.threadgroup_memory` |
| Captures | Python scalar constants; captured arrays are unsupported, so pass them as operands |
| Outputs | Multiple outputs and Pallas input/output aliases, subject to backend limitations |

Out-of-range edge lanes are guarded: invalid loads use Pallas-compatible
padding values and invalid stores do nothing. Arbitrary out-of-range indices
are not supported. Strided and guarded reads materialize contiguous values;
read snapshots are preserved across later writes.

## JAX operations in kernels

- **Elementwise:** add/subtract/multiply/divide, power, abs/negation,
  exp/log/log2/sin/cos/sqrt/tanh, sign/remainder/clamp, min/max, comparisons,
  casts, bitcasts, logical and bitwise operations, shifts, and broadcasting.
- **Array structure:** reshape, permuted reshape, transpose, and selection via
  `select_n`.
- **Reductions:** sum, min, and max over supported axis subsets, including
  `jnp.minmax`.
- **Dot:** standard rank-2 matrix multiplication and rank-1 matvec, vecmat, or
  vecvec. Batched dots and other contraction axes are rejected.
- **Control flow:** `fori_loop`, `scan` (including reverse
  scans and stacked outputs), `while_loop`, `cond`, and
  `switch`.
- **Random:** 32-bit `random.bits` and `fold_in`, plus
  key-data conversion, using Threefry-2x32-20.
- **Effects:** Ref reads and writes, plus Palladium's explicit cooperative
  barrier effect. Debug printing and arbitrary JAX effects are unsupported.

For `metal_call` and `metal_call_jit`, tiled matmuls on a 2D or 3D grid lower
through Metal 4 TensorOps by default (`dot_general="auto"`); dots the
cooperative lowering does not recognize keep the one-thread-per-program
emitter. `dot_general="tensorops"` requires the cooperative lowering and
raises where it does not apply; `dot_general="default"` forces the primitive
path. The lowering covers a standalone, tiled rank-2 matmul or matching rank-3
batches of matrices. The batched form uses arrays
`[B, M, K] @ [B, K, N]`, one batch per Pallas grid coordinate, and one
threadgroup per output tile. Batch sizes must match, M and N must tile evenly,
and each tile must contain the full K dimension. Rank-2 inputs may be
transposed. Fused epilogues support ReLU, a same-shape residual, or column bias;
the rank-3 path supports ReLU and a same-shape residual. Batch broadcasting,
edge tiles, split-K, and arbitrary `dot_general` batch dimensions are not
recognized. This is a grid of 2D products, not a batched `dot_general`
primitive. The attention form requires rank-4 `[batch, sequence, heads, D]`
buffers and a static key-tile scan; query and key/value lengths may differ for
noncausal attention. Causal attention requires matching lengths. Q/output use
query tiles, while K/V are full-sequence blocks. One Metal threadgroup computes
each query tile and runs the score and value matmuls with Metal TensorOps. Tile
sizes and head dimensions must be multiples of 16, and both sequence lengths
must divide evenly into their tiles. Short query sequences need padding and
output cropping. Unsupported jaxpr forms and layouts raise `EmitError`.
`mps_call_jit` does not use this path.

The backend supports float32, float16, bfloat16, int32, uint32, and bool, with
operation-specific limits:

| Type | Supported operations |
|---|---|
| float32 | All listed arithmetic, reductions, and dots |
| float16 | Load/store, elementwise, reductions, and dots; f32 accumulation when requested |
| bfloat16 | Load/store, elementwise, and reductions |
| int32, uint32 | Load/store, arithmetic, bitwise operations, shifts, and reductions |
| bool | Load/store, logic, and selection |

float64 and int64 are rejected. `preferred_element_type` controls
dot accumulation through the output dtype. Under SAFE math, floating extrema
preserve NaNs and signed-zero ties; FAST permits compiler reassociation and
does not promise those IEEE edge semantics. See [performance guidance](performance.md)
for numeric comparisons and measurement caveats.

## Cooperative kernels

`thread_index()`, `threads_per_threadgroup()`,
`barrier()`, and shared scratch support one to three dimensions.
Any cooperative operation requires an explicit `threadgroup=`.
Thread indices are linearized with x fastest. Barriers must be reached
uniformly; a conservative analysis rejects divergent barrier control flow.
Palladium checks direct top-level `scratch[thread_index()]` bounds,
but does not prove general shared-memory race freedom.

The Pallas CPU interpreter runs each program instance as a group of one. It
cannot validate cooperative results; compare these kernels with an independent
reference instead.

## JAX transformations and backends

| Path | JIT | vmap | Gradients |
|---|---|---|---|
| `mps_call_jit` | Yes, through jax-mps | Unsupported | Pair forward/backward calls with `custom_vjp`, or supply a pure-JAX reference VJP |
| `metal_call_jit` | Yes, through CPU FFI to Metal | Pipelined by default; sequential methods also available | Pair with `custom_vjp` |
| `metal_call` | Eager NumPy interface | Not a JAX transformation | No automatic differentiation |

`mps_call_jit` uses FAST math and the Pallas interpreter on non-MPS
platforms by default for independent kernels; `fallback="error"`
requires MPS. Cooperative kernels reject interpreter fallback.
`metal_call_jit` dispatches to Metal through the CPU FFI target and
does not need jax-mps.

Unsupported primitives raise `UnsupportedPrimitiveError`;
unsupported primitive cases raise `EmitError`. Unsupported Pallas
call structure raises `TraceError`. See [getting started](getting-started.md)
for diagnostics and examples.
