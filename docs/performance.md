# Performance guide

What makes a palladium kernel fast, what the fixed costs are, and how
to measure honestly on Apple silicon. Reference numbers are from an
M1 Pro; treat them as indicative.

## Where the speed comes from

The emitter's fast path is the cooperative execution model (see
`docs/supported-subset.md` for its eligibility rules): one SIMD-group
per program instance, R rows each, dots on the SIMD-group matrix units.
The bundled flash-attention example (`examples/06_flash_attention.py`)
is the reference workload: f32, head dim 64, one fused kernel with a
streaming online softmax, blocked over queries (R = 8) and keys.
Measured against `jax.jit` on CPU it runs about 2.8x at seq 1024 and
3.8x at seq 4096; composed inside `jax.jit` through `metal_call_jit`
with CPU code around it, the end-to-end win is about 2.4x and 2.0x.

Practical guidance:

- Check `explain()` first. One ineligible dot (a computed rhs is the
  common cause) silently drops the whole kernel to the classic model,
  which is several times slower for dot-heavy work. The reported reason
  tells you what to restructure.
- Give dot lhs a blocked shape. `q` blocks of 8 rows
  (`BlockSpec((8, head_dim), ...)`) are what let R = 8 and the MMA path
  engage; one row per instance forecloses both.
- Let transposes fuse. `jnp.dot(q, k.T)` costs nothing when `k` is an
  input block: the transpose lowers to an indexing convention (and a
  hardware transposed load on the MMA path), never a copy.
- Alignment matters for the float4 paths: they engage when block
  offsets are provably 4-element aligned, which they are for
  power-of-two block widths.

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
