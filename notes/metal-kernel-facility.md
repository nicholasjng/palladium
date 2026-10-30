# A Mosaic-style kernel facility for Apple GPUs (F0–F1 built, rest gated)

Goal: write Apple GPU kernels in Pallas with explicit control of the memory
hierarchy and the matrix instructions, as `jax.experimental.pallas.mosaic_gpu`
(`plgpu`) does for NVIDIA, and reach the vendor kernels' throughput by hand.
The recipe in `notes/apple-gpu-kernel-recipe.md` is the first user and the
measuring stick.

## What to build, and what not

Mosaic GPU is large because NVIDIA hardware has to be programmed explicitly:
asynchronous copies (TMA, `cp.async`), mbarriers, warpgroup matrix
instructions, tensor memory, clusters, plus an MLIR dialect with layout
inference that lowers to PTX. Apple exposes much less. MSL is the lowest
public level, Apple's compiler does register allocation and scheduling, and
there is no public asynchronous copy engine. What is worth exposing:

| Hardware feature | MSL | Pallas surface |
|---|---|---|
| Threadgroup memory | `threadgroup T x[n]` | a `THREADGROUP` memory space for scratch |
| Barriers | `threadgroup_barrier`, `simdgroup_barrier` | `plmtl.barrier()`, `plmtl.simdgroup_barrier()` |
| SIMD groups and lanes | `simdgroup_index_in_threadgroup`, `thread_index_in_simdgroup` | mesh axes, read with `lax.axis_index` |
| SIMD-group collectives | `simd_sum`, `simd_max`, shuffles | `plmtl.sum` / `max` over a lane axis, `plmtl.shuffle` |
| 8x8 matrix instruction | `simdgroup_matrix`, `simdgroup_multiply_accumulate` | `plmtl.mma` on fragment-layout arrays |
| Metal 4 tensor operations | `tensor_ops::matmul2d` (MPP) | `plmtl.tensor_matmul`, the path for GPUs with matrix accelerators |
| Specialization | function constants | static kernel parameters |

Not built: an MLIR dialect, a layout-inference solver, asynchronous copy
primitives. The emitter stays Python producing MSL, with the expression trees
of `notes/expression-trees.md` for structure. Layouts are few and explicit:
replicated, lane-strided, and the 8x8 fragment layout (MLX's
`BaseMMAFrag` coordinates).

The name `plmtl` is a placeholder for a `palladium.metal` module.

## Programming model

Follow where Pallas is going (`pl.kernel` over meshes; see the recipe note),
and keep `pallas_call` with BlockSpecs for simple kernels:

```python
mesh = plmtl.Mesh(grid=(m // 64, n // 64), simdgroups=4)

@functools.partial(
    pl.kernel,
    out_type=jax.ShapeDtypeStruct((m, n), jnp.float32),
    mesh=mesh,
    scratch_types=[plmtl.THREADGROUP((64, 20), jnp.float32),
                   plmtl.THREADGROUP((16, 68), jnp.float32)],
)
def matmul(a_ref, b_ref, out_ref, a_tile, b_tile):
    i, j = lax.axis_index("x"), lax.axis_index("y")
    acc = plmtl.zeros_fragments((64, 64), jnp.float32)  # fragment layout
    def step(kb, acc):
        plmtl.copy(a_ref.at[pl.ds(i * 64, 64), pl.ds(kb * 16, 16)], a_tile)
        plmtl.copy(b_ref.at[pl.ds(kb * 16, 16), pl.ds(j * 64, 64)], b_tile)
        plmtl.barrier()
        acc = plmtl.mma(acc, a_tile, b_tile)  # serpentine, unrolled
        plmtl.barrier()
        return acc
    acc = lax.fori_loop(0, k // 16, step, acc)
    out_ref[pl.ds(i * 64, 64), pl.ds(j * 64, 64)] = acc
```

This is the kernel `emit/tensorops/simdgroup.py` emits today, written in
Pallas. `plmtl.copy` is the cooperative vectorized tile copy, and `plmtl.mma`
is the per-lane fragment load plus serpentine multiply-accumulate. An
epilogue is plain `jnp` on the fragment array.

## Milestones

Estimates are guesses for one person. Status as of 2026-10-30, measured on an
M2 unless noted.

- **F0, baseline (done).** Roofline benchmark; float32 GEMM at 2.54 TFLOPS
  through `dot_general="simdgroup"`.
- **F1, threadgroup scope (done).** Built as row programs
  (`program="threadgroup"`, `notes/threadgroup-programs.md`): plain Pallas
  kernels whose block ops lower cooperatively, with column reductions and
  scans, reductions over the rows of a tile, and shifts along the columns.
  Softmax, layer norm and cumsum run within 2% of MLX at 91 GB/s. The
  explicit primitives first planned here (user-declared `THREADGROUP`
  scratch, barriers, lane indices) moved to F2/F4; 2D stencils with halos
  across rows go with them; histograms need atomic scatter, which core Pallas
  lacks.

What F0–F1 showed: the speed so far came from the emitter (recognized
matmuls, row programs), not from user-written schedules, and the largest
remaining costs were outside the kernels. Those come first:

- **N1, per-call overhead (done).** `metal_call` lost 3–4 ms per 268 MB call
  to the GPU's page-table setup for each new no-copy wrap, and over 128 MB
  wraps silently copied (XNU splits mappings into 128 MB entries).
  metal-runtime now caches validated wraps and walks all entries; a row
  softmax over 128 MB went from 6.4 to 3.4 ms end to end.
- **N2, `"auto"` uses `simdgroup_matrix` on Apple families 7 and 8 (done).**
  4096³ float32 end to end: 2.42 TFLOPS against 1.89 for `matmul2d`. To do:
  confirm on an M1 (with the metal-runtime that reports `apple_gpu_family`):
  `uv run mew run --random-interleaving -k "float32-n4096|attention"
  benchmarks/bench_roofline.py` should show `simdgroup` ahead of `tensorops`
  in every pair. Measure an M3 or M4 before widening the families.
- **N3, attention in the recognizer (done).** At 4096 tokens and 32 heads of
  64, float32: 2.10 TFLOPS (1.93 causal) with `simdgroup_matrix` for both
  products and the softmax in registers, against 1.48 for `matmul2d` and
  MLX's 1.99. Taken by `"auto"` on families 7 and 8 for query tiles of 32 or
  more rows. This met the old F5 target without F2–F4.

The explicit model is gated on a kernel that needs it:

- **F2, fragments (2–3 weeks).** A fragment-layout array type, `plmtl.mma`,
  `plmtl.copy`, and elementwise epilogues on fragments. Its done condition
  (the hand-written matmul above at the emitted kernel's 2.5 TFLOPS) gains
  no speed by itself; it pays off only for kernels the recognizers cannot
  match, such as dequantizing matmuls or attention variants. Start when such
  a kernel has a user.
- **F3, pipelining: dropped.** Metal has no asynchronous copies, and
  threadgroup staging and program-order swizzles did not help the GEMM
  (`notes/apple-gpu-kernel-recipe.md`). Revisit only if a GPU trace shows
  loads and compute not overlapping.
- **F4, `pl.kernel` on Metal (1–2 weeks), with F2.** A `plmtl.Mesh`
  implementing `jax._src.pallas.core.Mesh`, and `mpmd_map` lowerings for
  `mps` (and `cpu` for `metal_call`). Blocked on the mps side until jax-mps
  supports jax 0.12 and takes the `mps.metal_kernel` custom call.
- **F5, attention written with F2–F4: not needed for speed.** N3 met its
  target inside the recognizer; an explicit version would only serve
  attention variants the recognizer cannot match.
- **F6, tensor operations (open).** `plmtl.tensor_matmul` over MPP, chosen on
  GPUs with matrix accelerators (Apple family 10). metal-runtime reports the
  family as `device_info()["apple_gpu_family"]`. Needs such a machine.
- **F7, docs.** A tutorial in the style of the Mosaic GPU matmul guide:
  roofline, each step, its number. Can start with row programs and the
  `simdgroup` GEMM.

Cross-cutting, from F1 on:

- `interpret=True` keeps working: every new primitive gets a discharge or
  interpret rule (barriers as no-ops in sequential execution), so the Pallas
  interpreter stays the correctness oracle, plus the differential fuzz tests.
- `KernelDiagnostics` reports achieved TFLOPS and percent of the roofline.
- Every generated fragment loop is fully unrolled.

## Dependencies and risks

- Pallas internals: `pl.kernel`, `mpmd_map` and the `Mesh` protocol changed
  between jax 0.11 and 0.12 and are partly private. Keep the adapter small and
  test it against every jax release.
- jax-mps: the mps path needs the generic `mps.metal_kernel` custom call
  upstream (patch in `~/Code/jax-mps-metal-kernel.patch`) and a jax-mps that
  supports the jax version in use (today it pins `jax<0.12`).
- Apple's compiler is opaque: unrolling, spills and scheduling are decided
  there. The roofline benchmark and GPU traces are the feedback loop.
- No public asynchronous copies: pipelining may gain little (F3 decides).
- Hardware: only an M2 measured; GPUs with matrix accelerators change the
  ceilings, and F6 is untested.
