# Pallas-to-MSL TensorOps

## Design decisions

- Keep the Pallas grid and its `BlockSpec` mappings as logical program
  semantics. Select execution scope separately: one MSL thread per program for
  scalar kernels, or one threadgroup per program for cooperative work.
- Keep Metal-specific operations in target lowering: TensorOps calls,
  cooperative ownership, barriers, threadgroup limits, and MSL emission.
  Share low-level MSL TensorOps primitives across cooperative lowerings.
- Every cooperative kernel declares its group width with a
  `// palladium: threadgroup width N` line (`_kernel_source`); the launch
  reads it and nothing else.

## Current lowering

`compile_kernel(spec)` (`src/palladium/emit/tensorops`) picks
`lower_attention` when the kernel contains a scan and `lower_matmul`
otherwise, both matching directly on the jaxpr. A plan/import/layout pass
pipeline and standalone cooperative reduction and pointwise lowerings existed
until 2026-09-30; they were unreachable because only kernels with a dot reach
TensorOps. Row programs (`notes/threadgroup-programs.md`) now cover that
ground without a dot.

`metal_call`, `metal_call_jit`, `debug_msl`, and `emit_msl` route tiled
matmuls and supported attention here automatically;
`dot_general="default"` forces the primitive path, and untiled dots keep it
because they carry no tiled cooperative schedule. The matmul lowering
supports FP32, FP16, or BF16 rank-2 matmuls with lazy transposes, rank-3
batched FP32 matmuls, full-K or K-tail accumulation, and partial M/N edge
tiles, with elementwise epilogues, residuals and column bias.
`dot_general="simdgroup"` swaps the `matmul2d` main loop for
`simdgroup_matrix` on supported float32 shapes
(`notes/apple-gpu-kernel-recipe.md`).

Scalar epilogues run on each thread's MPP destination cooperative tensor,
then store the tile. Epilogues that read another tile use threadgroup staging
and a barrier. Reading an MPP threadgroup result as a flat array gives the
wrong element layout; the cooperative tensor path is the fix.

## JAX API review

`jax.extend.core` provides the Jaxpr types and traversal helpers this
frontend uses, including `subjaxprs`, `jaxprs_in_params`, and `jaxpr_as_fun`.
The custom nested-region walker remains useful because it also handles
`ClosedJaxpr` values and nested containers. Pallas `BlockMapping` exposes
canonical block shapes, transforms, and an interpret-only start-index
evaluator; it does not produce symbolic MSL offset expressions, so our
block-map checks and offset emission remain target-specific. Mosaic TPU and
Mosaic GPU have useful per-primitive lowering registries and cooperative
scope models, but their rules emit MLIR for TPU or NVIDIA. They are
architectural references, not drop-in MSL lowerings.

## Attention schedule

Both MPP matmuls stay; the work between them is spread over the group. SIMD
group `sg` owns score rows `sg, sg + 4, ...`; lane `l` owns columns
`l, l + 32, ...`. Each lane reads its BK/32 scaled scores once into
registers, `simd_max` gives the block max, probabilities are computed in
registers and written back in place for the value matmul, `simd_sum` gives the
block sum, and lane 0 commits the row carries plus the accumulator rescale
factor to `row_scale`. Causal masking and partial key tiles (BK < 32) are lane
guards on the same path. (The first schedule gave each score row to one lane,
idling most of the group.)

The output accumulator is the value matmul's destination cooperative tensor,
declared before the key loop and held in registers throughout. MPP exposes
each element's coordinates through `get_multidimensional_index`, which a probe
on an M2 confirmed are exact and distributed over all 128 threads (16
elements per thread for a 32×64 tile, index order (column, row)). The per-step
rescale and the final normalizing store use those coordinates, so the
accumulator never touches threadgroup memory. Barriers stay at three per key
step.

Measured on the M2 (`[1, 4096, 4, 64]`, f32): 16.6 to 13.8 ms at 32×32, 17.0
to 11.6 ms at 32×64, and 9.3 ms at 16×128, which the smaller footprint newly
allows. Staging Q in threadgroup memory was tried and rejected (21.4 ms
against 14.1 ms at 32×32) because the extra shared bytes cost occupancy.
