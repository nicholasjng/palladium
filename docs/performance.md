# Performance guide

What makes a palladium kernel fast, what the fixed costs are, and how
to measure honestly on Apple silicon. Reference numbers are from an
M1 Pro; treat them as indicative.

## Where the speed comes from

palladium's model is one thread per program instance, with parallelism
coming from the Pallas grid, not from any per-instance vectorization
trick: a large ensemble of independent integrator instances (one RK4
trajectory, one SDE path, one stencil cell) maps directly onto the
GPU's thread count, which is where the win comes from. The bundled
Lotka-Volterra RK4 ensemble example (`examples/01_ode_ensembles.py`) is
the reference workload: N=100,000 independent trajectories, 500 fixed
RK4 steps each. Measured against `jax.jit(vmap(diffeqsolve))` on CPU
(adaptive Tsit5) it runs tens of times faster at that batch size.

Practical guidance:

- Grid size is the main lever. More independent program instances
  means more GPU threads in flight; a kernel with a small or absent
  grid (one instance doing everything) gets none of this.
- Keep per-instance blocks small. Each instance's blocks live in
  thread-local memory, so per-instance BlockSpecs must stay well under
  the per-thread stack; a gridless call over a large array overflows it
  (and says so in the error).
- Prefer `fori_loop`/`scan` with a static trip count over `while_loop`
  when the count is known: both lower fine, but a data-dependent trip
  count means threads diverge on wall-clock time even though the
  one-thread-per-instance model tolerates divergence correctness-wise.

## Fixed costs

- **Dispatch floor**: a blocking GPU dispatch costs about 136 us on an
  M1 Pro regardless of kernel size (measured decomposition in the
  metal-runtime devlog). It overlaps across in-flight batches (down to
  roughly 33 us at depth 16) but not for a caller that blocks per call.
  Consequence: sub-millisecond kernels are dominated by it; batch more
  work per call if you can.
- **FFI overhead**: `metal_call_jit` adds about 0.25 ms per call in
  total (zero-copy buffer wrapping; measured 0.19 ms at 16 KB buffers,
  0.25 ms at 1 MB). At multi-millisecond kernel sizes the FFI path runs
  within 3 percent of the direct one.
- **Input upload**: eager `metal_call` copies inputs per call. For
  repeated calls on unchanging inputs use `call.pin(*arrays)`, which
  uploads once and returns a zero-argument callable.

## Math mode

FAST (default) is measurably faster for transcendental-heavy kernels
and is what all quoted numbers use. SAFE preserves IEEE ordering; it is
required for compensated arithmetic (FAST deletes the error terms of
two-sum tricks) and useful when diffing against the oracle bit-nearly.

## Measuring on Apple silicon

Sustained GPU load down-clocks an M1 Pro by up to 3x, which is enough
to invert an A/B comparison run sequentially. Hard-won rules, each of
which caught a wrong conclusion in this project's history:

1. Compare candidates paired and interleaved (A, B, A, B, ...), never
   in separate runs or long sequential passes.
2. Prefer device time (`metal_runtime.Batch.gpu_time`) over wall time
   for kernel-vs-kernel comparisons, so the fixed dispatch floor does
   not flatter the slower kernel.
3. Treat absolute numbers as lower bounds unless the machine was rested;
   ratios from interleaved pairs are the only stable currency.
4. Rerun with repetitions and take medians; single runs are noise.
