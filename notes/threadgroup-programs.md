# Threadgroup programs (planned, not built)

The hand-written cooperative API (`thread_index`, `threads_per_threadgroup`,
`barrier`, `threadgroup_memory`) was removed on 2026-09-30. It had no users,
made kernels palladium-only, and `interpret=True` could not check it (the
interpreter runs each program as a group of one). The use cases it served are
real, though: block reductions and softmax wider than one thread, prefix sums
(particle-filter resampling, stream compaction), stencils that stage a tile
and halo in shared memory, histograms, and reducing ensemble statistics
without a round trip to JAX.

## Direction: keep the kernel plain Pallas, change the execution scope

A `CompilerParams` option runs each Pallas program as one Metal threadgroup
instead of one thread, the way the TensorOps path already does. Block-level
`jnp` ops then lower cooperatively across the group:

- elementwise ops: lanes stride over the block (flat ownership)
- `jnp.sum` / `jnp.max` along an axis: SIMD-group reductions per row,
  `simd_sum` / `simd_max` plus a threadgroup combine for wide rows
- `jnp.cumsum`: a SIMD-group scan plus a threadgroup pass
- loads and stores: lanes stride over the block, with edge masks

Kernels stay portable and `interpret=True` remains a valid oracle, because the
jaxpr is ordinary Pallas.

## Starting points in history

- The standalone cooperative row-reduction and pointwise lowerings
  (`emit/tensorops/reduction.py`, `elementwise.py`, deleted in change
  `synqwkxu`) already emit this code for a program-per-threadgroup scope.
  They were unreachable only because nothing selected the scope for a
  kernel without a dot.
- The removed API and its checks (`threadgroup.py`, `_validate_barriers` in
  `trace.py`) are in the parent of change `knslnlpo`. The barrier-uniformity
  check is not needed here: barriers become the emitter's job, placed
  between the cooperative stages it generates.
- `emit/tensorops/softmax.py` has the SIMD-group row-reduction pattern the
  attention lowering uses.

## Open questions

- Selection: an explicit `CompilerParams(program="threadgroup")`, or
  inferred from block sizes? Explicit first.
- How values that do not fit the cooperative patterns (arbitrary `scan`
  bodies over block values) are handled: reject with an EmitError at first.
- Measure against the one-thread-per-program emitter on a real workload
  (SIR summary statistics, a particle-filter resample) before generalizing.
