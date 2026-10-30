# Threadgroup programs (row programs built 2026-10-29)

Built: `CompilerParams(program="threadgroup")` lowers row programs
(`src/palladium/emit/threadgroup.py`): 1D grids of `(rows, columns)` blocks,
elementwise operations, broadcasts, and sum, max or min over the columns.
Narrow rows (up to 256 columns) get one SIMD group each; wider rows the whole
threadgroup, which declares its width to the launch with a
`// palladium: threadgroup width N` line. Each block is read into registers
once per row. Softmax and layer norm run at 91 GB/s on an M2 (98% of copy
bandwidth, within 2% of MLX); `tests/runtime/test_row_programs.py`.

Scans along the columns (`cumsum`, `cumprod`, `cummax`, `cummin`, forward and
reverse) are built too. Each slot of the row's threads is a contiguous
chunk, scanned by SIMD-group shuffles; with a whole threadgroup on the row,
the first SIMD group combines every chunk's partial totals at once, three
barriers per row. Also at 91 GB/s and MLX parity. SIMD-group partials are
stored as 32-bit patterns, so integer sums stay exact past 2**24.

Reductions over the rows of a tile are built: each thread accumulates its
columns across the row loop in registers; in narrow mode the four SIMD
groups combine through threadgroup memory after the loop, and SIMD group 0
runs the epilogue. Everything that depends on such a reduction is deferred
to the epilogue and must go to per-tile `(1, columns)` or `(1,)` outputs;
tiles must divide the rows. Shifts along the columns (`slice`, `pad`,
`concatenate`, so `jnp.roll`, `jnp.pad`, `jnp.diff`) are column remaps:
shifted block columns are read from device memory (L1 hits) instead of
registers, with clamped indices and a select chain per piece.

Moved out of row programs (`notes/metal-kernel-facility.md`): user-declared
threadgroup scratch, explicit barriers and lane indices go to the explicit
model (F2/F4); 2D stencils with halos across rows need staged tiles and the
same; histograms need atomic scatter, which core Pallas does not express.
The text below is the original plan.

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
