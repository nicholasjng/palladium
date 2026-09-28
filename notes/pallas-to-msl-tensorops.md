# Pallas-to-MSL TensorOps

## Design decisions

- Keep the Pallas grid and its `BlockSpec` mappings as logical program
  semantics. Select execution scope separately: one MSL thread per program for
  existing scalar kernels, or one threadgroup per program for cooperative work.
- Preserve jaxpr equations, loops, and effects through planning. Lower
  primitives compositionally; Pallas-authored schedules such as online
  softmax remain explicit and do not need an attention-pattern matcher.
- Give values a logical shape, address space, and ownership distribution.
  Layout assignment is a later pass; inputs and outputs begin as device views,
  and scratch uses its declared thread or threadgroup space.
- Keep Metal-specific operations in target lowering: TensorOps calls,
  cooperative ownership, barriers, threadgroup limits, and MSL emission.
  Share low-level MSL TensorOps primitives across cooperative lowerings.

## First implementation slice

The TensorOps API is `plan_kernel` -> `import_kernel` -> `assign_layouts` ->
`compile_kernel` (`src/palladium/emit/tensorops`). The importer retains nested jaxpr
regions, including scans. The initial layout pass labels dot results as
TensorOps-owned and row reductions as row-owned in cooperative scope.

`compile_kernel` exposes the planning API. `metal_call`, `metal_call_jit`,
`debug_msl`, and `emit_msl` route tiled matmuls and supported attention to
TensorOps automatically; `dot_general="default"` forces the primitive path.
Untiled dots retain the primitive emitter because they do not carry a tiled
cooperative Pallas schedule. For threadgroup attention, TensorOps
analyzes the imported scan region, checks TensorOps and row ownership, and
builds an online-softmax lowering plan directly from its operations. It emits
its MSL body directly. Standalone matmuls lower from imported IR, including
elementwise chains, residuals, and column bias. The lowerer supports FP32,
FP16, or BF16 rank-2 matmuls with lazy transposes, rank-3 batched FP32
matmuls, full-K or K-tail accumulation, and partial M/N edge tiles. Other
cooperative scans and generic cooperative Pallas kernels remain unsupported.
Scalar epilogues run on each thread's MPP destination cooperative tensor, then store
the tile to the output. Epilogues that read another tile still use threadgroup
staging and a barrier. The device correctness benchmark exposed that directly
reading an MPP threadgroup result as a flat array gives the wrong element
layout; the cooperative tensor path fixes fused ReLU and ReLU-plus-bias.

This removes intermediate threadgroup reads/writes and barriers for chained
epilogues. It does not reduce HBM traffic versus another already-fused
single-kernel matmul: the materialized values were on-chip. To measure the HBM
benefit of fusion itself, `benchmarks/bench_pallas_matmul_epilogue_fusion.py`
compares this path with a two-kernel baseline that writes and rereads the
matmul result in a device buffer. Its byte counter reports those logical
device-buffer writes and reads; it does not claim they all reach DRAM, since
caches may serve some accesses.

Before the K-loop, edge-tile, and dtype extensions, a user device run passed
correctness for fused ReLU-plus-bias and reported 0.76 ms in its `Real` column.
The pasted output did not include split-path median counters, so it confirms
that earlier fused path but does not establish a speedup over the two-kernel
baseline. The device correctness benchmark passed all three extended cases:
FP32 with a K tail and partial M/N tiles, FP16 with a multi-step K loop, and
BF16 with a K tail. The reported `Real` times were 0.24 ms, 0.23 ms, and
0.26 ms respectively; these are coverage-run timings, not a comparison against
another lowering. Run it on a Metal machine with:

```sh
JAX_PLATFORMS=mps,cpu uv run mew run benchmarks/bench_pallas_matmul_tensorops_coverage.py
```

The cooperative row reducer supports standalone FP32 `reduce_sum` and
`reduce_max` on rank-2 inputs with full-width row tiles. Lanes within each
SIMD group accumulate strided columns and use `simd_sum`/`simd_max`; SIMD
groups stride over rows, and partial final row tiles are guarded. This is a
generic reduction-jaxpr path, separate from attention. Coverage now includes
short and 1025-column rows; rerun
`benchmarks/bench_pallas_reduction_tensorops_coverage.py` after this parallelization
to check device numerics and timing.

The following cooperative path lowers standalone FP32 pointwise chains over
equally shaped rank-1 to rank-3 buffers. Each threadgroup lane processes a
strided subset of the tile. It currently accepts one output and straight-line
scalar elementwise operations. The ReLU-scale-residual coverage check is
`benchmarks/bench_pallas_elementwise_tensorops_coverage.py`. Its first device run
exposed incorrect global addressing for column tiles: flattened tile indices
were added directly to the tile base, ignoring the full-array row stride. The
lane offset now reconstructs tile coordinates and applies the full-array
strides. The corrected kernel passed device validation at 0.19 ms in the
`Real` column; this is a coverage-run timing, not a comparison against another
lowering.

Pointwise edge tiles are now masked by checking each lane's global coordinate
before loading or storing. The coverage benchmark uses a 65x97 array and 16x32
tiles to exercise partial edges on both axes. The user device run passed at
0.18 ms in the `Real` column; this is a coverage-run timing, not a comparison
against another lowering.

The next pointwise extension accepts scalar operands and trailing row-vector
bias operands. Jaxpr `broadcast_in_dim` is represented as a per-lane alias,
while scalar and vector reads use their own block maps and address strides.
`benchmarks/bench_pallas_elementwise_tensorops_coverage.py` now checks both broadcast
forms together with the irregular edge tiles. The user device run passed
numerics and reported unchanged timing relative to the preceding elementwise
case. The coverage benchmark compiles the cooperative kernel through
`compile_kernel(scope=ProgramScope.THREADGROUP)` and dispatches the emitted
source directly.

Example after tracing a Pallas callable:

```python
from palladium.emit.tensorops import ProgramScope, compile_kernel

compiled = compile_kernel(spec, scope=ProgramScope.THREADGROUP)
msl = compiled.source
```

Choose `THREADGROUP` when one Pallas program cooperatively owns a tile. Scalar
one-thread-per-program kernels use the default scope. Public emission routes
tiled dot kernels and supported attention to TensorOps automatically; untiled
dots and non-matmul kernels use the primitive emitter. `dot_general="default"`
is available as a matmul fallback override. Cooperative reductions and
elementwise kernels remain available through the explicit `compile_kernel`
planning API. The legacy TensorOps matchers and emitters have been removed.

## Performance comparison

The comparison uses noncausal `[1, 4096, 4, 64]`, tiles `32 x 64`. After TensorOps
took ownership of MSL assembly, both paths still emitted byte-identical
4,687-byte MSL and requested 16,640 bytes of threadgroup storage. On the
current arm64 host, over 30 warmed codegen calls, median Python-side time was
1.000 ms for TensorOps and 0.197 ms for the specialized emitter. The added cost is
TensorOps planning, IR import, layout analysis, and its separate MSL assembly.

The migration comparison benchmark has been removed now that TensorOps is the
sole Metal 4 cooperative path. The local development environment has no Metal
device, so GPU results below were reported by the user from device runs.
Reported pair times were 22.85 ms at `32 x 64` and 18.65 ms at `16 x 64`,
implying about 11.43 ms and 9.33 ms per launch respectively. The
MSL sources are byte-identical, so these results show no kernel speedup. Use
`--format json` to see the benchmark's per-emitter median-latency counters.

In a later device run, the user reported legacy/TensorOps medians of 10.183/10.115 ms,
identical MSL, and a 1.0067x paired ratio. This again shows that the current TensorOps
work changes frontend/emitter structure, not the generated attention schedule.

The latest `tq16-tk64` run reported 9.294 ms legacy versus 9.316 ms TensorOps
(0.9976x), effectively parity with identical MSL. V2 codegen took 1.064 ms
versus 0.286 ms legacy; Metal compilation took 1.46 us versus 9.46 us, a
small absolute difference. The measured TensorOps frontend currently adds codegen
cost without improving this kernel's runtime.

Previously recorded specialized attention measurements were 11.51 ms at
`32 x 64` and 9.44 ms at `16 x 64`, consistent with the per-launch estimates.
The next performance opportunity is to improve the plan or schedule (for
example layout choice or loop pipelining), then measure on-device with resident
buffers and identical correctness tolerances.

## Size and final review

Before replacement, the TensorOps package was about 1,941 source lines and the legacy
TensorOps matmul and attention matcher/emitter about 1,519 lines. Those legacy
files are now removed; the cooperative path uses TensorOps and shared TensorOps MSL
primitives. LOC is a rough proxy, and TensorOps also supports standalone cooperative
elementwise and reduction kernels.

The final cleanup consolidated compiler dispatch, corrected the matmul output
view's dtype metadata, and formatted the TensorOps modules. Further large reductions
would mean changing the deliberately narrow attention and matmul recognizers,
which risks weakening their validation contracts.

## JAX API review

The installed environment has JAX 0.11.2. `jax.extend.core` already provides
the Jaxpr types and traversal helpers this frontend uses, including
`subjaxprs`, `jaxprs_in_params`, and `jaxpr_as_fun`. The custom nested-region
walker remains useful because it also handles `ClosedJaxpr` values and nested
containers. Pallas `BlockMapping` exposes canonical block shapes, transforms,
and an interpret-only start-index evaluator; it does not produce symbolic MSL
offset expressions, so our block-map checks and offset emission remain
target-specific. Mosaic TPU and Mosaic GPU have useful per-primitive lowering
registries and cooperative scope models, but their rules emit MLIR for TPU or
NVIDIA. They are architectural references, not drop-in MSL lowerings. No
installed JAX API replaces the MPP descriptors/views or the MSL expression
renderer.
