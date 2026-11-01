# A vertically integrated Metal compute stack for JAX (plan, not started)

Goal: run whole JAX training programs on Apple GPUs through a PJRT plugin
we own, with no MLX involvement, for three workloads: ODE ensembles for
simulation-based inference, SAM2 fine-tuning, and LLM fine-tuning. Planned
as a fresh repository; palladium's emitter and runtime are the starting
material. This complements the kernel facility in
`notes/metal-kernel-facility.md`: that note is about writing kernels, this
one about running whole programs.

## Why now

- Kernels are at parity with MLX on an M2: row softmax, layer norm and
  cumsum at 91 GB/s (within 2%), float32 GEMM at 2.54 TFLOPS (93% of MLX),
  attention at 2.10 TFLOPS against MLX's 1.99.
- Whole programs are not limited by kernels. In the CNF update
  (`xla-metal-replay`, 2026-10-28), jax-mps with async dispatch took 149 µs
  per update of which 61 µs was GPU time; the rest is host glue and
  synchronization, which only the plugin can remove. Replaying XLA's fusions
  from outside the plugin capped out at about 1.3×.
- jax-mps decides more than speed: it pins `jax<0.12` (blocking F4 of the
  facility plan), and palladium's custom call is a patch on top of it.

## What the plugin has to execute

A StableHLO census of representative training steps (2026-10-30, jax 0.12;
SBI: RK4 ensemble plus a mixture-density posterior; LLM: decoder block with
RoPE, causal attention, SwiGLU and cross-entropy; SAM2-like: conv patch
embedding, windowed attention, max-pooling, transposed-conv upsampling,
focal plus dice loss; all with AdamW) found 31–38 distinct ops per program,
about 50 in total:

| Group | Ops | Kinds | Work |
|---|---|---|---|
| Generated | elementwise math, compare, select, logic and shifts, convert and bitcast, iota, constant, broadcast, reshape, transpose, slice, concatenate, pad, reverse, dynamic_update_slice, reduce | ~38 | One fusion code generator; palladium's emitter and row programs already cover the jaxpr equivalents |
| Control | while, call | 2 | Compiler and runtime |
| Kernels | dot_general, convolution, gather, scatter, reduce_window, select_and_scatter | 6 | Tuned kernel families |

Where the kernels come from: GEMM everywhere (LLMs need bf16/f16 at
scale); convolutions from SAM2's patch embedding, transposed-conv upsampling
and their gradients (implicit GEMM covers them); gather and scatter-add from
embeddings and `take_along_axis` and their gradients; reduce_window and
select_and_scatter from max-pooling and its gradient. The real workloads add
a few: searchsorted for spline flows, small linear algebra (Cholesky,
triangular solves) for Gaussian posteriors, dropout RNG (integer
elementwise code), and sort or top-k only for sampling.

What JAX hands a new platform (2026-11-01, jax 0.12, lowered for a stub
`metal` backend with no device behind it):

- The three census programs lower with no custom calls. The LLM step is
  identical to the CPU lowering; the SBI step differs only in threefry,
  whose default rule unrolls the 20 rounds instead of looping, which suits
  fusion.
- Linear algebra: Cholesky and triangular solves arrive as
  `stablehlo.cholesky` and `stablehlo.triangular_solve`; LU is a JAX-built
  loop of gathers, scatters and triangular solves, so `solve`, `inv`,
  `slogdet` and Gaussian log densities need only those two kernels. QR
  arrives as XLA's internal custom calls `@Qr` and
  `@ProductOfElementaryHouseholderReflectors`. `eigh` and `svd` (also `eig`,
  `schur`, `hessenberg`, `tridiagonal`) have no default rule and fail;
  jax-mps registers its own `eigh`, `qr` and `svd` rules for `mps`.
- Sort and top-k arrive as `stablehlo.sort`, cumulative sums as
  `stablehlo.reduce_window`, FFTs as `stablehlo.fft`, and the `rbg` PRNG
  as `stablehlo.rng_bit_generator`.
- Host callbacks fail: `pure_callback` and `io_callback` check a fixed set
  of platforms (`jax/_src/callback.py`), and `jax.debug.print`,
  `debug_callback` and `checkify` have no default rule. Supporting them
  needs our own rules and a host-callback handler in the plugin.
- Pallas: `pallas_call` has no default rule, and `mpmd_map`'s raises for
  unknown platforms. Rules registered from Python with
  `mlir.register_lowering(..., platform="metal")` work for both and emit a
  `stablehlo.custom_call` whose typed-FFI backend config must be a
  dictionary; that is how palladium's kernels reach the plugin.

Without MLX, the long tail can fall back to Apple's system frameworks:
Metal Performance Shaders (matrix multiply, CNN convolutions, Cholesky/LU
and solves), which encode into our own command buffers; MPSGraph, which has
the broadest op set but its own scheduling and compile overhead (Apple's
MPSGraph-based jax-metal went stale); and Accelerate on the CPU for rare,
small ops, which unified memory makes copy-free.

## Architecture

1. PJRT plugin: client, devices, buffers on unified memory with donation,
   async execution, one command buffer per executable step.
2. Compiler: StableHLO in; fusion passes; generated MSL for fused
   elementwise and reduction regions; calls into the kernel families;
   Pallas kernels as native custom calls.
3. Runtime: palladium's C API (`native/runtime`) or its successor.
4. Kernels: GEMM (`simdgroup_matrix`, and Metal 4 `matmul2d` where the GPU
   has matrix accelerators), attention forward and backward, implicit-GEMM
   convolution, gather/scatter, windowed reductions.

## Milestones

On the M4 Pro Mac mini (64 GB, Apple GPU family 9) unless noted. Each one
is measured against an established baseline on the same machine.

| # | Milestone | What it shows | Measured against | Target |
|---|---|---|---|---|
| M0 | Baselines | Where the ceilings and competitors sit | Roofline benchmark (bandwidth; simdgroup peak in f32/f16/bf16; `matmul2d` vs `simdgroup` GEMM); the workloads on jax-mps, XLA:CPU, MLX/mlx-lm and PyTorch MPS | A results table; settles the GEMM path for family 9 |
| M1 | Minimal plugin | `jax.jit` on our platform: zero-copy buffers, async execution, generated elementwise and reduction kernels | jax-mps on the same programs; CPU results | Matches CPU; per-call overhead clearly below jax-mps |
| M2 | SBI end to end | The execution-model premise: whole steps on the GPU without host glue | jax-mps, XLA:CPU, diffrax | Same losses as CPU; GPU time is most of wall time; faster than jax-mps |
| M3 | Kernels for LLM training | Kernel parity where MLX is strongest | MLX | bf16 GEMM, gather and scatter-add, attention forward and backward at MLX speed (the backward is new) |
| M4 | LLM LoRA fine-tune | Beating MLX at its own use case | mlx-lm LoRA, same 0.5–1B model and data | Same loss curve; tokens/s at or above mlx-lm; comparable peak memory |
| M5 | SAM2 fine-tune | Generality: convolutions, pooling, larger graphs | PyTorch MPS (qgis-segment) and jax-mps | Same mask quality; steps/s above PyTorch MPS |
| M6 | Compiler decision | Whether Python codegen hits its limits | Compile times and fusion quality from M2–M5 | A measured case for an MLIR dialect, or none |

M0 comes first: every later target is relative to this machine. M2 is the
earliest real proof; if owning execution does not close much of the gap
between 149 µs and 61 µs per CNF-like update, the plugin's premise fails
within weeks. M4 is the result that matters most. Throughout, record
first-step compile time, peak memory, and agreement with a CPU or XLA
reference: a faster step that compiles for a minute or drifts numerically is
not a win.

M1 so far (2026-11-01):

- Plugin, `native/pjrt/`, against the PJRT C API 0.116 that jaxlib 0.12
  pins. One `metal` device with one `device` memory, buffers in shared
  MTLBuffers (host data is copied in, strided arrays included), copies to
  the host and other devices, memory stats, and `apple_gpu_family` as a
  device attribute. `palladium._pjrt` registers it through the
  `jax_plugins` entry point, only when JAX_PLATFORMS names `metal`.
- Compiler in Python: `PJRT_Client_Compile` calls back into
  `palladium.stablehlo`, which deserializes the portable artifact with
  jaxlib's MLIR bindings and returns buffer slots and kernel launches; the
  plugin compiles the MSL once and runs every launch in one command buffer.
  Elementwise ops (through palladium's scalar templates) and index maps
  (broadcast, reshape, transpose, slice, reverse, iota) fuse into one kernel
  per output shape, with calls and composites inlined; other ops are
  UNIMPLEMENTED. Fifteen programs match the CPU backend.
- First numbers on the M2, `tanh(2x + 1) * x`: 162 µs per call at 1,024
  elements against 3 µs on CPU, 0.99 ms at 4M elements against 0.92 ms.
  Execution is synchronous and allocates every output, so the small case
  measures a command buffer round trip plus allocation.
- Lessons: the 0.12 header defines `PJRT_Error` and `PJRT_Memory` as
  function tables callers use directly; jaxlib calls some slots without a
  null check (`PJRT_Client_TopologyDescription`, `PJRT_Error_ForEachPayload`,
  `PJRT_TopologyDescription_Fingerprint` for cache keys,
  `PJRT_LoadedExecutable_GetDeviceAssignment`, whose proto is encoded by
  hand); programs arrive as VHLO portable artifacts.

Next: asynchronous execution with completion events and reused output
buffers, then reductions, `dot_general` and `while`, which the SBI step of
M2 needs.

Work that costs time without proving anything: a JAX port of SAM2's image
encoder and mask decoder (SAM2 is PyTorch) for M5, and a small JAX LoRA
training stack for M4.

## MLIR, an `applegpu` dialect, and PDLL

Compile in Python first (decided 2026-11-01): jaxlib's MLIR bindings read
the StableHLO the plugin receives, so the plugin needs no LLVM, MLIR or
StableHLO build and no StableHLO pin of its own, and palladium keeps one
emitter. C++ passes inside the plugin, built against a pinned LLVM/StableHLO
prefix, wait for M6 to show that Python codegen limits compile time or
fusion. Add an `applegpu`
dialect when fused StableHLO and Pallas kernels need one shared lowering
target, around M3–M5, not before: the wins so far came from schedule
decisions (registers versus threadgroup memory, barrier counts, tile shapes),
not from IR infrastructure, and the planned expression-tree refactor
(`notes/expression-trees.md`) fixes the Python emitter's main weakness for
far less. If Pallas should lower into the dialect, follow the Mosaic
backends: jaxpr to MLIR from Python through jaxlib's MLIR bindings.

Model in the dialect what produced the wins: 8x8 SIMD-group fragments and
their lane layout, threadgroup memory and barriers, SIMD-group collectives,
and Metal 4 tensor operations for GPUs with matrix accelerators.

Emit MSL text, not AIR: MSL is documented and compiles at runtime without
Xcode (the managed laptop has only the Command Line Tools); AIR is
undocumented and needs the Xcode Metal toolchain. The community Triton
backends for Apple GPUs (out of tree, e.g. the triton-ext PRs that map
`tt.dot` to an Apple MMA encoding and emit MSL from TTGIR) take the MSL
route too; their MMA encoding is worth reading before designing ours.

PDLL suits small declarative op-to-op rewrites, but the decisions that
matter (what to fuse, layout assignment, tiling, scheduling) want C++
passes, which is also what Triton uses. Start with C++ `RewritePattern`s.

## Risks

- JAX releases monthly; owning the plugin means keeping up with StableHLO
  bytecode compatibility at every jaxlib bump.
- Matrix accelerators (M5 and later) change which kernels win; the
  `matmul2d` path is the likely one there and is untested.
- One person: each milestone is weeks to months. Let the three workloads
  decide what gets built, and stop at M2 if the premise does not hold.
