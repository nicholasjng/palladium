# A Pallas matmul and attention recipe near the Apple GPU roofline (planned)

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
| palladium flash attention, same, 32x128 tiles | 1.34 (40%) | not measured |

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

## Steps

1. **Roofline benchmark** (done): `benchmarks/bench_roofline.py`.
2. **Threadgroup staging.** A threadgroup memory space for Pallas scratch,
   barriers, and several SIMD groups per program cooperating on one larger
   tile (`notes/threadgroup-programs.md`). Done when float32 4096³ reaches
   2.5 TFLOPS with a 128x128 effective tile.
3. **Double-buffered K loop.** Metal has no public counterpart of TMA or
   `cp.async`, so overlap comes from two staging buffers and barriers, not
   from asynchronous copies. This is the main difference from the Mosaic GPU
   recipe. Done when the loop's load and compute phases overlap in a GPU
   trace and float32 gains over step 2.
4. **Program-order swizzle and tile search.** Group programs so neighbouring
   threadgroups share operand tiles in cache, then sweep tile shapes (attention
   already showed 2x from tile shape alone). Done at the targets above.
5. **Attention.** Head sizes padded to multiples of 16 (SAM2's 56 is not
   supported today), larger key tiles, probabilities times V in float16, and
   a look at `exp` throughput. Done at ≥ 2.0 TFLOPS float32.
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
