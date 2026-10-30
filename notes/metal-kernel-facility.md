# A Mosaic-style kernel facility for Apple GPUs (planned, not built)

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

Estimates are guesses for one person.

- **F0, baseline (done).** Roofline benchmark; float32 GEMM at 2.54 TFLOPS
  through `dot_general="simdgroup"`.
- **F1, threadgroup scope (3–4 weeks).** A `pallas_call` option running each
  program as one threadgroup, a `THREADGROUP` scratch space, barriers, SIMD
  group and lane indices (`notes/threadgroup-programs.md`). Cooperative
  elementwise ops and row reductions (`plmtl.sum` and `max` with SIMD-group
  collectives). Done when a row softmax and a layer norm written this way
  match MLX's fused kernels within 20%.
  Status 2026-10-29: done as row programs (`program="threadgroup"`): column
  reductions and scans, reductions over the rows of a tile to per-tile
  outputs, and shifts along the columns (roll, pad, diff). Softmax, layer
  norm and cumsum run within 2% of MLX at 91 GB/s. Moved out: user-declared
  `THREADGROUP` scratch, explicit barriers and lane indices go to the
  explicit model with F2/F4; 2D stencils with halos across rows likewise;
  histograms need atomic scatter, which core Pallas lacks.
- **F2, fragments (2–3 weeks).** A fragment-layout array type, `plmtl.mma`,
  `plmtl.copy`, and elementwise epilogues on fragments. Done when the
  hand-written Pallas matmul above reaches 2.5 TFLOPS float32 at 4096³, the
  same as the emitted one.
- **F3, pipelining (1–2 weeks).** `plmtl.emit_pipeline`: a double-buffered K
  loop over two threadgroup buffers. Done when it is measured against F2;
  keep it only if it gains.
- **F4, `pl.kernel` on Metal (1–2 weeks).** A `plmtl.Mesh` implementing
  `jax._src.pallas.core.Mesh`, and `mpmd_map` lowerings for `mps` (and `cpu`
  for `metal_call`). Done when the example above runs on both paths.
- **F5, attention (2–3 weeks).** Flash attention written with F1–F3; head
  sizes not multiples of 16 padded. Done at ≥ 2.0 TFLOPS float32 (MLX parity
  at 4096 tokens, 64-wide heads).
- **F6, tensor operations (open).** `plmtl.tensor_matmul` over MPP and a
  device capability query in metal-runtime, to choose it on GPUs with matrix
  accelerators. Needs such a machine.
- **F7, docs.** A tutorial in the style of the Mosaic GPU matmul guide:
  roofline, each step, its number.

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
- Hardware: only M1/M2-class GPUs measured; GPUs with matrix accelerators
  change the ceilings, and F6 is untested.
