# Threadgroup programs (row programs built 2026-10-29)

Built: `CompilerParams(program="threadgroup")` lowers row programs
(`src/palladium/emit/threadgroup.py`): 1D grids of `(rows, columns)` blocks,
elementwise operations, broadcasts, and sum, max or min over the columns.
Narrow rows (up to 256 columns) get one SIMD group each; wider rows the whole
threadgroup, up to 1024 threads. Like every cooperative kernel, it declares
its width with a `// palladium: threadgroup width N` line. Each block is read
into registers once per row. Softmax and layer norm run at 91 GB/s on an M2 (98% of copy
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

Why plain Pallas: a hand-written cooperative API (`thread_index`,
`barrier`, `threadgroup_memory`) existed until 2026-09-30 and was removed. It
had no users, made kernels palladium-only, and `interpret=True` could not
check it. Row programs keep the kernel an ordinary jaxpr, so the interpreter
stays a valid oracle, and only the execution scope changes.

Open: measure on a real workload (SIR summary statistics, a particle-filter
resample) against the one-thread-per-program emitter. On synthetic rows the
row programs are 4–8× faster in GPU time, except cumsum on 256-column rows
(1.1×); reductions over rows and column shifts are not benchmarked yet.
