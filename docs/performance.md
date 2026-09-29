# Performance and example results

Palladium is aimed at kernels with substantial work per independent Pallas
grid instance: an ODE trajectory, SDE path, or stencil cell. It fuses that work
into one Metal dispatch. The model is one Metal thread per program instance;
parallelism comes from the grid.

## Recorded results

These local Apple-GPU runs show what the approach can do. They are workload-
and machine-specific, not general speedup guarantees. The training results
used JAX 0.11.1; the recorded notes did not consistently retain the GPU model.

| Workload | Palladium | Comparison |
|---|---:|---:|
| Lotka–Volterra, 100,000 trajectories × 500 RK4 steps, MPS custom call | 1.059 ms | 63.008 ms for `jit(vmap(scan))` on MPS |
| Lotka–Volterra training, 4,096 trajectories × 100 steps, full Adam update | 0.482 ms, reverse VJP, checkpoint interval 10 | 39.173 ms for pure JAX on MPS |
| 2D CNF, 256 points, width 4, 16 RK4 steps, full optimizer update | 0.504 ms, reverse VJP, interval 4 | 17.609 ms for JAX on MPS; 1.392 ms for JAX on CPU |

Training measurements are warmed, synchronized medians over 15 samples with
rotating variant order. Updates start from identical optimizer states; input
transfer and first-call compilation are excluded. Run the Mew benchmarks with
`JAX_PLATFORMS=mps,cpu uv run mew run --random-interleaving benchmarks/`.
See the reproducible scripts for [RK4](../benchmarks/bench_jax_mps_rk4.py),
[ODE training](../benchmarks/bench_mps_training.py), and
[CNF training](../benchmarks/bench_cnf_training.py). The experimental
[Pallas attention benchmark](../benchmarks/bench_pallas_flash_attention.py)
checks causal and noncausal Pallas TensorOps lowering against a NumPy reference
and measures resident-buffer runs at sequence lengths from 1,024 to 4,096. It
needs an MSL 4-capable runtime. The [MLX SDPA comparison](../benchmarks/bench_attention_vs_mlx_sdpa.py)
times the same kernel through `mps_call_jit` under `jax.jit` against
`jax.nn.dot_product_attention`, which jax-mps routes to MLX's fused kernel; run it
with the same `mew run --random-interleaving` command as the other benchmarks.

On an M1 Pro, f32 runs with shape `[1, sequence, 4, 64]` and 32×64 query/key
tiles measured:

| Sequence | Causal | Pallas-generated MSL |
|---:|:---:|---:|
| 1,024 | No | 0.96 ms |
| 1,024 | Yes | 0.95 ms |
| 2,048 | No | 3.07 ms |
| 2,048 | Yes | 3.09 ms |
| 4,096 | No | 11.29 ms |
| 4,096 | Yes | 11.09 ms |

With 32×32 query/key tiles on the same M1 Pro (JAX 0.11.2), the noncausal
4,096 case measured 4.96 ms, so the tile choice matters more than the mask.

Those M1 Pro rows predate the current schedule. The attention lowering now
owns each score row by one SIMD group (lanes across columns, `simd_max` and
`simd_sum` for the row reductions, scores read once into registers) and keeps
the output accumulator in the value matmul's cooperative tensor instead of
threadgroup memory, addressed through MPP element coordinates. Threadgroup
memory per group fell from `BQ×BK + BQ×D + 2·BQ` to `BQ×BK + 3·BQ` floats, which
is what makes 128-wide key tiles affordable. On an M2 (8-core GPU) through
metal-runtime, f32, `[1, 4096, 4, 64]`, medians of resident-buffer runs:

| Tiles (q×k) | Before | After |
|:---:|---:|---:|
| 32×32, full | 16.6 ms | 13.8 ms |
| 32×32, causal | 9.6 ms | 7.7 ms |
| 32×64, full | 17.0 ms | 11.6 ms |
| 32×128, full | n/a | 10.7 ms |
| 16×128, full | n/a | 9.3 ms |
| 32×128, causal | n/a | 6.0 ms |

At `[1, 1024, 4, 64]` the 32×128 tiles measured 0.96 ms against 2.1 ms before.
Wide key tiles amortize the three barriers per key step; 64×128 exceeds the
32 KB threadgroup budget. Staging the query tile in threadgroup memory was
measured and rejected: it costs 1.5x at 4,096 through lost occupancy.

These are steady-state medians with resident inputs; the generated kernel
passed the benchmark's NumPy correctness check for both mask modes. Causal and
noncausal timings are within the variation between runs, with no consistent
overhead from causal masking at these sizes. These are workload-specific
results, not a general TensorOps speedup guarantee.

A separate JAX 0.11.2 forward-only probe ran 4,096 CNF trajectories for 64
RK4 steps through CPU FFI to Metal. Warm medians were 0.510 ms at width 4,
3.347 ms at width 16, and 9.276 ms at width 32. It used five samples and had
overlapping CPU activity, so treat the values as rough scaling data.

The symplectic example measures numerical behavior rather than throughput. Over
5,000,000 Kepler Verlet steps, float32 energy error grows with a fitted log-log
slope of +0.50, consistent with accumulated round-off. Compensated df32 stays
near the integrator's step-size error bound. Run
[the example](../examples/symplectic_longrun.py) on Metal to reproduce it.

## Costs and practical choices

- A blocking Metal dispatch has measured at about 136 μs on M1 Pro, regardless
  of kernel size. Pipelined dispatches overlap this queue latency; very short
  kernels benefit from fusing work in each instance or batching work in the
  Pallas grid.
- Keep each instance's blocks and intermediates small: they occupy per-thread
  storage and can exceed Metal's stack limit.
- `metal_call` uploads NumPy inputs on each call. Use
  `call.pin(*arrays)` when inputs stay unchanged across repeated
  dispatches, and `call.iterate(*arrays, steps=n)` for state-update
  kernels (stencils, PDE steps, agent models): all `n` dispatches go into
  one command buffer and each step's outputs refill the next step's inputs
  on the device. `examples/reaction_diffusion.py` runs 2,000 Gray-Scott
  steps this way in 20 ms on an M2.
- `metal_call_jit` adds CPU-FFI setup and buffer wrapping. The
  measured passthrough overhead was 0.19 ms for 16 KB buffers and 0.25 ms for
  1 MB; the common path wraps eligible XLA memory without copying. These are
  end-to-end FFI costs, not bufferization-only timings.
- Use static `fori_loop` or `scan` when practical. A
  data-dependent `while_loop` is correct, but instances may finish
  at different times.

For comparisons, warm compilation first, synchronize every sample, and
interleave candidates (A, B, A, B) to reduce thermal bias. Report the device,
JAX version, workload shape, and whether timing includes transfers or
compilation. The [JAX functionality guide](supported-jax.md) describes the
available call paths and math modes.
