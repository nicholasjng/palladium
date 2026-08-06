# Changelog

Pre-1.0 semver: a minor version may break the public API, a patch may
not. The public API is `palladium.__all__`; everything else (the
`emit.core`/`emit.rules`/`emit.coop` internals in particular) is
private and may change between minor versions.

The jax dependency is pinned to a tested range (`>=0.11,<0.12`); a
weekly CI canary tests against jax head and the pin widens when it
passes. metal-runtime is consumed as a path dependency during
development; the pin moves to a tagged release when one exists.

## 0.2.0 (2026-08-18)

### Performance

- Simdgroup-cooperative execution model: kernels whose dot_generals fit
  a one-SIMD-group-per-instance layout (R query rows per instance,
  values distributed over 32 lanes) are detected and lowered onto
  SIMD-group functions and the SIMD-group matrix units
  (simdgroup_float8x8), with query blocking expressed through ordinary
  BlockSpecs. The bundled flash-attention example runs 2.8x/3.8x over
  jax.jit on CPU at seq 1024/4096 (f32, head dim 64, M1 Pro), and a
  jax.jit-composed hybrid (attention on the GPU, the rest on CPU) holds
  2.4x/2.0x end to end.
- Vectorized float4 paths for dot_general, gated on a gcd-composed
  element-alignment analysis (`CVal.align`) instead of a boolean.
- The cooperative MMA dot column-tiles wide outputs (fixed register
  pressure per tile, threadgroup result tile up to 16KB), lifting the
  previous output-width cap; narrow dots emit byte-identical code.
- A specialized multi-SIMD-group GEMM lowering for bare matmul kernels
  (load/load/matmul/store): several SIMD-groups per threadgroup share
  a staged rhs slab with adaptive geometry, storing fragments straight
  to the output. Bare f32 GEMMs now beat jax.jit-on-CPU 1.3-1.6x
  (550-790 GFLOP/s at batch 4096 on M1 Pro); the MLP training example
  (examples/08_mlp_training.py) went from ~0.1x of the CPU step to
  0.4-0.9x, now bound by dispatch and host-transpose overhead rather
  than kernel throughput.
- `MetalCallable.pin(*args)` uploads inputs once and returns a
  zero-argument callable for repeated dispatch on unchanging inputs.

### Features

- `metal_call_jit`: kernels registered as a jax.ffi target, traceable
  and jittable, composing with surrounding jax.numpy code inside
  jax.jit. Dispatches both execution models with the correct launch
  geometry. `jax.vmap` supported via `vmap_method="sequential"` (whole-
  batch methods are rejected: the launch grid is baked per unbatched
  shape); `jax.grad` via the `jax.custom_vjp` pairing recipe in
  `examples/07_custom_vjp.py`, scaled up to a full jitted MLP training
  step in `examples/08_mlp_training.py` (correct and gated against a
  jnp reference; not yet fast, see its docstring).
- bfloat16 end to end on both paths. NumPy cannot pass ml_dtypes
  arrays over DLPack, so the eager path ships the bytes as uint16 and
  relabels the buffer: a bit-lossless reinterpretation, measured
  indistinguishable from a native-dtype upload.
- Control flow in kernels: `lax.cond`/`lax.switch` (clamped index
  semantics) and `lax.while_loop` with divergent per-instance trip
  counts, plus `and`/`or`/`xor`/`not` and `clamp`.
- Rank-1 `dot_general` (matvec, vecmat, vecvec) canonicalized to the
  rank-2 rules. `preferred_element_type` honored: accumulation runs in
  the output dtype, mixed-precision products are promoted before the
  multiply, and the vectorized/MMA paths require f32 end to end.
- In-kernel counter-based RNG (`jax.random.bits`/`fold_in`,
  Threefry-2x32-20, bit-exact against jax).

### Diagnostics

- `MetalCallable.explain(*args)` / `FfiCallable.explain(*args)`: which
  execution model a kernel gets, its launch geometry, and, when it
  falls back to the classic one-thread-per-instance model, the first
  reason the cooperative model was rejected. `PALLADIUM_EXPLAIN=1`
  prints one stderr line per newly compiled kernel.
- One exception hierarchy under `PalladiumError`: `TraceError` (also a
  ValueError), `EmitError`, `UnsupportedPrimitiveError` (also a
  NotImplementedError), `DispatchError` (also a TypeError). Errors name
  the offending primitive, argument, or limit.
- Call-boundary validation: strict dtype and shape checks against the
  traced spec (no silent casts), unsupported element types rejected
  before tracing with the fix in the message, threadgroup-memory
  budgets checked at emission with the numbers.

### Correctness

- Worked around a Metal shader compiler bug that miscompiles carry
  permutations in loops whose bodies hold three or more thread-local
  array temporaries (wrong results from the second iteration on, SAFE
  and FAST math alike, reproduced in hand-written MSL). Permuted scan/
  while carries now copy back through volatile hazard endpoints; the
  permutation-free path is unchanged. Found by the differential
  fuzzer; also fixed a previously known strict-xfail case with the
  same root cause.
- Scans with stacked ys or scanned xs are rejected structurally
  (version-tolerant of jax's scan param layout) instead of
  miscompiling.
- Thread safety: compile caches are locked (concurrent first calls
  compile once per shape), kernel launches serialize their upload
  phase, and metal-runtime's `Batch.wait()` now blocks every caller
  rather than only the first.

### Internal

- The emitter is a package (`emit.core` machinery, `emit.rules` classic
  model, `emit.coop` cooperative model) with a shared jaxpr walker; the
  unused standalone simdgroup_matmul module is gone.
- Differential fuzzer extended to control flow and cooperative
  dot-anchored kernels.

## 0.1.0

Initial version: trace/emit/dispatch pipeline for Pallas kernels
(block load/store, elementwise ops, grids and BlockSpecs, fori_loop and
pure-carry scan, indexed refs, reductions, transpose, dot_general),
`metal_call` with the `interpret` oracle, golden-MSL snapshots, ODE
example suite (Lotka-Volterra ensembles, per-thread adaptive stepping,
SDE Monte Carlo, Gray-Scott).
