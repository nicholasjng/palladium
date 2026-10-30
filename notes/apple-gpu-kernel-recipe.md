# A Pallas matmul and attention recipe near the Apple GPU roofline (step 2 done)

Goal: hand-written Pallas kernels that reach a large share of the GPU's
throughput, told as a roofline story like the scaling book and the Mosaic GPU
matmul tutorial. `benchmarks/bench_roofline.py` measures the ceilings and the
current matmul; this note records them and the steps from there.

## The roofline (Apple M2, 10-core GPU, 16 GB; 2026-10-29)

| Ceiling | Measured |
|---|---:|
| `simdgroup_matrix` multiply-accumulate, float32 / float16 | 3.38 / 3.41 TFLOPS |
| Theoretical peak (public spec, not measured) | ~3.6 TFLOPS |
| Device memory bandwidth (copy) | 92.6 GB/s |
| Ridge point | ~36 FLOP/byte |

The matrix instruction runs on the same ALUs at the same rate for float16 and
float32: M1/M2-class GPUs have no separate matrix units. A scalar FMA
microbenchmark reached only 1.4 TFLOPS in float32 and stays unexplained; real
matmuls exceed it, so it is not a ceiling.

Where kernels land (TFLOPS, share of the 3.38 matrix peak):

| Kernel | float32 | float16 |
|---|---:|---:|
| torch-MPS matmul 4096³ | 2.91 (86%) | 3.22 (95%) |
| MLX matmul 4096³ | 2.72 (80%) | 3.31 (98%) |
| palladium Pallas matmul 4096³, best tile | 2.12 (63%), 128x64 | 3.06 (91%), 64x32 |
| palladium Pallas matmul 2048³, best tile | 2.48 (73%), 64x32 | 3.30 (98%), 64x32 |
| MLX attention, 4096 tokens, 32 heads of 64 | 1.99 (59%) | 2.55 (75%) |
| torch-MPS attention, same | 1.74 (51%) | 2.48 (73%) |
| palladium flash attention, same, `matmul2d`, 64x64 tiles | 1.48 (44%) | not measured |
| palladium flash attention, same, `simdgroup_matrix`, 64x16 tiles | 2.10 (62%) | not measured |

The vendor rows were measured with torch 2.13 and MLX 0.32.1 directly;
`bench_vendor_matmul` measures `jnp.dot` through jax-mps instead, when the
mps platform is installed. Attention counts 4·heads·tokens²·64 FLOPs.

Targets: float32 matmul at vendor parity (≥ 2.7 TFLOPS at 4096³), float16
matmul ≥ 3.2, attention ≥ 2.0 at float32.

## Why float32 lags

Today one program computes a `tm x tn` output tile and streams full `K` strips
of both operands from device memory. Its arithmetic intensity is
`2·tm·tn / ((tm + tn)·bytes)`: 16 FLOP/byte for 64x64 in float32, 32 in
float16. Float16 sits at the ridge point and runs near peak. Float32 sits at
half of it and depends on cache hits, which is why it drops from 2.48 TFLOPS
at 2048³ to 2.12 at 4096³, and to 0.75 with 32x32 tiles. The fix is a larger
effective tile per device-memory read: stage operands in threadgroup memory,
share them across SIMD groups, and order programs so neighbours reuse cache
lines.

## Step 2 results (2026-10-29, M2, float32 4096³, GPU time)

| Variant | TFLOPS |
|---|---:|
| palladium `matmul2d`, 128x64 tile, 4 SIMD groups (baseline) | 2.12 |
| `matmul2d` 128x128, 4 / 8 / 12 SIMD groups | 0.77 / 1.47 / 1.53 |
| `matmul2d` 128x128, 16 SIMD groups | does not launch: register use caps the threadgroup at 384 threads |
| `matmul2d` 128x128, 8 SIMD groups, K step 16 / 32 / 64 / 128 | 1.35 / 1.47 / 1.33 / 1.47 |
| element-wise threadgroup staging in front of `matmul2d` | 0.49–0.67 |
| grouped program order (groups of 4, 8, 16 row tiles), 64x64 and 128x64 | 1.90–2.12 |
| hand-written `simdgroup_matrix` GEMM, 64x64x16, 2x2 SIMD groups | 0.05 |
| same, with `#pragma clang loop unroll(full)` on the fragment loops | 2.00 |
| same, threadgroup rows padded by 4 elements | 1.97 |
| same, float16 | 2.65 |
| **MLX-style inner loop** (below), 64x64x16, 2x2 SIMD groups | **2.54** |
| MLX-style, 64x64x32 / 128x64x16 | 2.34–2.37 / 2.46–2.47 |
| MLX-style, float16 inputs, float32 accumulation | 2.97–3.09 |
| MLX (same tile configuration on this device class, no swizzle) | 2.72 |

The hand-written variant scripts behind these rows were removed on
2026-10-30. The winning kernel is `emit/tensorops/simdgroup.py`, and
`benchmarks/bench_roofline.py` measures it against `matmul2d`.

What this changes:

- The traffic model below fits the small tiles, but its premise is not the
  whole story: if re-reading operands from device memory were the limit,
  grouped program order should have helped, and it did not.
- Large tiles do not work with `matmul2d` on M1/M2-class GPUs: the
  accumulator spills, and more SIMD groups recover only part of it.
- Staging is not a win by itself, and a hand-written kernel matches
  `matmul2d` (2.0) only once its fragment loops are fully unrolled.
  Without that, the accumulators go to stack memory (40x slower).
- MLX reaches 2.72 with the same tile configuration, so the remaining gap is
  in the inner loop. MLX does not use `simdgroup_load`: each thread reads its
  own fragment elements from threadgroup memory (`BaseMMAFrag::load` in
  `steel/gemm/mma.h`), and accumulators stay thread-local vectors that are
  reinterpreted as `simdgroup_matrix` only for the multiply.

Porting that inner loop (with MLX's interleaved SIMD-group layout,
serpentine product order and contiguous per-thread tile copies) reaches
2.54 TFLOPS, past the 2.5 target and 93% of MLX. The rest of the gap is
unexplored: MLX's vectorized tile loader, its epilogue, or compiler flags.

The TensorOps lowering now emits this kernel for float32 tiled matmuls
under `CompilerParams(dot_general="simdgroup")`
(`src/palladium/emit/tensorops/simdgroup.py`); float16, transposed operands,
edge tiles, K not a multiple of 16 and tiles above 8192 elements keep
`matmul2d`. A plain Pallas `jnp.dot` through palladium, M2, 3 repetitions
(`bench_roofline.py`):

| float32, tile | `matmul2d` | `simdgroup` |
|---|---:|---:|
| 4096³, 64x64 | 1.97 | **2.54** |
| 4096³, 128x64 | 2.00 | 2.47 |
| 4096³, 32x32 | 0.68 | 2.09 |
| 2048³, 64x32 | 2.35 | 2.64 |

`"auto"` now picks `"simdgroup"` on Apple GPU families 7 and 8 (M1, M2), read
from `palladium.device.device_limits()["apple_gpu_family"]`: 4096³ float32 end
to end through `metal_call` ran at 2.42 TFLOPS against 1.89. Measured on the
M2 only; confirm on an M1, and measure an M3 or M4 before widening it. GPUs
with matrix accelerators (family 10) keep `matmul2d`.

## Steps

1. **Roofline benchmark** (done): `benchmarks/bench_roofline.py`.
2. **Threadgroup staging.** A threadgroup memory space for Pallas scratch,
   barriers, and several SIMD groups per program cooperating on one larger
   tile (`notes/threadgroup-programs.md`). Measured (see Step 2 results):
   larger tiles and staging alone do not reach the target; a hand-written
   kernel with MLX's inner loop does. Done: palladium emits it for a plain
   Pallas `jnp.dot` under `dot_general="simdgroup"`, 2.54 TFLOPS at 4096³.
3. **Double-buffered K loop.** Metal has no public counterpart of TMA or
   `cp.async`, so overlap comes from two staging buffers and barriers, not
   from asynchronous copies. This is the main difference from the Mosaic GPU
   recipe. Done when the loop's load and compute phases overlap in a GPU
   trace and float32 gains over step 2.
4. **Program-order swizzle and tile search.** Group programs so neighbouring
   threadgroups share operand tiles in cache, then sweep tile shapes (attention
   already showed 2x from tile shape alone). Grouped order showed no effect
   at 4096³ (Step 2 results); MLX also uses none on this device class. Done
   at the targets above.
5. **Attention (done for float32).** Following MLX's steel attention: both
   products on `simdgroup_matrix`, K and V staged in threadgroup memory, and
   scores, probabilities and output in each SIMD group's registers, so a key
   step has two barriers and the softmax rows reduce with two shuffles. 2.10
   TFLOPS without a mask and 1.93 causal at 64x16 tiles, against 1.48 and 1.26
   for `matmul2d`. Masking only the key blocks that cross the diagonal took
   causal from 1.83 to 1.93. Head sizes need only be multiples of 8 on this
   schedule: 56-wide heads ran in 59.8 ms with 64x8 tiles, against 65.7 ms
   padded to 64. Probabilities times V in float16 was not tried: it changes
   the results of a float32 kernel, so it would need an opt-in precision
   setting, and nothing asks for one.
6. **Write-up.** The roofline, each step's change and its number, in the
   style of the Mosaic GPU matmul tutorial.

## Programming model: follow `pl.kernel`

Pallas moved its GPU model away from `pallas_call`: Mosaic GPU kernels are
only written with `plgpu.kernel()` since jax 0.11.2, `program_id` is
deprecated there in favour of `jax.lax.axis_index`, and `pl.core_map` is
deprecated for `pl.kernel`, which in 0.12 lowers through a new `mpmd_map`
primitive over one or more meshes. The Triton backend is deprecated.

For Metal, the matching model is a mesh of threadgroups with SIMD groups (and
lanes) as inner axes, and memory spaces for device, threadgroup and thread
storage. `pallas_call` with BlockSpecs stays the entry point for simple
kernels.

How it could plug in (read from jax 0.12 source, not tried):

- `jax._src.pallas.core.Mesh` is a protocol (`shape`, `default_memory_space`,
  `core_type`, `supported_memory_spaces`, `discharges_effect`,
  `tracing_context`), so palladium can define its own mesh.
- `mpmd_map`'s lowering matches only `cuda`, `cpu`, `rocm` and `tpu`, so
  `pl.kernel` on `mps` fails today. Registering a lowering for `platform="mps"`
  that reuses `_mpmd_map_fallback_lowering` would turn the mesh into a grid
  and reach `pallas_call._pallas_call_lowering`, where palladium's mps
  lowering already hooks in. That function is private, so palladium would own
  a small compatibility layer for it.

Ideas from Mosaic GPU's 0.12 additions worth copying: block-level `sum`,
`max`, `min`, `prod` with an accumulator ILP parameter (the cooperative
reductions in `notes/threadgroup-programs.md`), an explicit `store` with an
`optimized` switch, and `ref.memory_space_cast`.

## jax 0.12 status

- `jnp.minmax` now traces to a variadic `reduce` with a combiner jaxpr. A
  generic `reduce` rule now inlines the combiner per element, with one
  accumulator per operand updated like loop carries; it also covers argmax
  pairs and other multi-operand reductions.
- `pl.DeviceIdType` was removed; palladium does not use it.
- The test suite had not started since metal-runtime removed `device_name()`
  (2026-09-29); the probes now use `device_info()`.
