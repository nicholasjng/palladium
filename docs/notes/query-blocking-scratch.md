# Query blocking in flash attention: a 2x the emitter can't express yet

> **Outcome (2026-08-14, same day):** implemented. The cooperative model
> generalized to `m == R` dot_generals (columns-per-lane layout, no
> threadgroup memory needed after all — see `emit.py`'s cooperative
> section), `examples/06` gained `block_q`, and the ratio transferred:
> generated block_q=8 runs 0.693ms device at the scored block_kv=16
> (1.76x over one-row) and 0.468ms at block_kv=32 (2.6x, within 11% of
> this doc's hand-written 0.422ms). Reward v0.2.0 scores
> **1.000 (saturated)** at 1.42x over jax.jit wall time, heat-soaked.
> The "second execution-model fork" this doc predicted for BQ>8 reuse
> remains unbuilt — unnecessary at this shape, since the knee at 8 held
> for generated code too (block_q=16 regressed to 1.146ms).

**Scratch, 2026-08-14.** Findings from a dispatch-overhead investigation in
`metal-runtime`, written up here because the conclusion is about this repo's
emitter, not that runtime. Nothing here is implemented. The kernels measured
are hand-written MSL standing in for what the emitter *would* produce, so
treat the ratios as the result and the absolute numbers as indicative — see
"What transfers" at the end.

## The question, and why the obvious answer was wrong

The starting hypothesis was that per-call dispatch overhead explains the
flash-attention gap: `examples/06` runs ~1.5 ms against `jax.jit` CPU's
1.351 ms (`docs/notes/reward-spec-matmul-emitter.md`), a ~150 us gap, and the
empty-kernel dispatch floor was known to be ~0.2 ms
(`docs/notes/simdgroup-matmul-design.md`). Those numbers line up so well that the
gap looks fully explained.

It is explained, and it is not actionable. Measured properly in
metal-runtime (`benchmarks/bench_overhead.py`, `bench_phase` and
`bench_submit_path`), one blocking dispatch costs 136 us, split:

| phase | time |
|---|---|
| `Batch()` create | 2.4 us |
| `add()` encode | 0.3 us |
| `commit()` | 2.2 us |
| `wait()` | **131 us** |

and `wait()` splits further, by the command buffer's own timestamps, into 9 us
inside the driver, 54 us submitted-but-not-yet-running, 5 us of actual GPU
execution, and 77 us of this thread being rescheduled after completion. An
*empty* kernel costs the same 133 us, so none of it is work. About 130 us is
platform floor that no API choice removes: unretained command buffers,
completion handlers, shared events, and residency sets all measured *worse*
than the baseline, and Metal 4 is 19% better (110 us) at the cost of the
entire binding model.

The one lever there is that this is latency, not throughput — it overlaps
across command buffers, so per-dispatch cost falls to 83/51/39/33 us at 2/4/8/16
in flight. That is unavailable to a caller that blocks per call, which
`BoundKernel.__call__` does by construction.

So dispatch overhead is a ~9% fixed tax with roughly 20 us of reachable
savings in it. Parity has to come from the kernel.

## Fusion is already maximal, so "fuse more" is not the move

Worth stating explicitly, because it was the first instinct. `examples/06` is
already one `pallas_call`, one `mr.Batch`, one `batch.add`, one dispatch
(`dispatch.py:155-162`), with a streaming online softmax over KV blocks
(`examples/06_flash_attention.py:107-139`) that never materialises the
`(seq, seq)` score matrix. Both `dot_general`s emit inline into the
surrounding kernel with results in thread-local registers; there are no
intermediate device buffers anywhere in `emit.py`, no `threadgroup` address
space, and no barriers (`scratch_shapes` is still rejected outright,
`emit.py:519`). It pays the 130 us floor exactly once.

There is no fusion left to add. The useful question is what the fusion
*costs*.

## What the fusion costs: `m == 1`

`grid=seq_len` puts one query row in each Pallas program instance. Under the
cooperative model that becomes one 32-lane SIMD group per query row
(`dispatch.py:149-151`), and every `dot_general` in the kernel has `m == 1`.
That single property causes three separate problems:

1. **It forecloses `simdgroup_matrix`.** An 8x8 MMA tile with `m == 1` wastes
   7/8 of itself — exactly the reasoning that left the intrinsics stranded in
   the standalone `palladium.simdgroup_matmul`
   (`docs/notes/simdgroup-matmul-design.md`, steps 2-4 outcome).
2. **It multiplies K/V traffic by the query count.** Each instance streams all
   of K and V past itself, so K and V are re-read 1024 times at
   `seq_len=1024`.
3. **It makes register pressure the binding constraint.** The three dead ends
   already recorded in `emit.py` are all downstream of it: vectorizing the
   n=64 PV accumulator *regresses* 8-10% because "the extra float4 registers
   compete with the online-softmax carry (m, l, o)" (`emit.py:1163-1181`);
   register-hoisting the query row was 2-2.4x slower via stack spill; and
   threadgroup-memory K staging measured 0.43-0.79x and was reverted. Each was
   read as a local failure. They are one structural symptom.

Real flash attention blocks over queries *and* keys. `examples/06` blocks only
over KV.

## The measurement

Two hand-written MSL kernels at this repo's shapes (seq 1024, head_dim 64,
f32), in `metal-runtime/benchmarks/bench_attention.py`, both validated against
a float64 NumPy oracle before timing:

- `rowwise` — today's shape: one 32-lane SIMD group per query row, one
  `simd_sum` per score.
- `blocked<n>` — one SIMD group per `n` query rows, staging Q and the softmax
  probabilities `P` in threadgroup memory so each loaded K/V element serves
  `n` rows. Loops are k-outer / row-inner so the reuse is in registers.

Rested, `--random-interleaving --repetitions 7`, device-time medians:

| variant | SIMD groups | GPU | vs rowwise | of FP32 peak |
|---|---|---|---|---|
| `rowwise` | 1024 | 850 us | 1.00x | 6% |
| `blocked2` | 512 | 627 us | 1.36x | |
| `blocked4` | 256 | 434 us | 1.96x | |
| `blocked8` | 128 | **422 us** | **2.02x** | 12% |
| `blocked16` | 64 | 704 us | 1.21x | |
| `blocked32` | 32 | 5120 us | 0.17x | |

Query blocking is worth **2.0x of device time at BQ=8**. Adding the 130 us
dispatch floor puts `blocked8` near 550 us against `jax.jit` CPU's 1351 us —
not parity, 2.5x, which saturates the `min(t_jax/t_metal, 1.0)` reward
outright.

## Why it peaks at 8, and the ceiling that implies

Reuse and parallelism are coupled *inversely* in this design, because a
threadgroup is exactly one SIMD group. Blocking by `n` cuts K/V traffic by `n`
and also divides the SIMD-group count by `n` — and there are only 1024 query
rows to divide. BQ=32 leaves 32 SIMD groups for 16 GPU cores, two per core,
with nothing left to hide memory latency behind; it is 6x *slower* than
`rowwise`. The knee at 8 is where those two curves cross, and it will move
with `seq_len`: this is a property of the shape, not a constant.

Getting reuse past 8 therefore needs threadgroups of several SIMD groups
sharing one staged K/V block, so the reuse factor and the SIMD-group count
stop being reciprocal. That means splitting the KV loop across SIMD groups and
combining their partial `(m, l, o)` triples through threadgroup memory — a
*second* execution-model fork stacked on the cooperative one, not a codegen
tweak. Worth knowing before anyone treats BQ as a free tuning knob.

`blocked8` at 12% of the M1 Pro's ~5.2 TFLOP/s FP32 peak also says the MMA
intrinsics remain the larger untapped win. BQ=8 is precisely the block size
that makes them usable.

## What this asks of the emitter

Query blocking is expressible in the Pallas kernel today: widen the `q`
`BlockSpec` and `out_specs` from `(1, head_dim)` to `(block_q, head_dim)`
(`examples/06_flash_attention.py:133,137`) and the grid from `seq_len` to
`seq_len // block_q` (line 131). `jnp.dot(q, kj.T)` then has `m == block_q`,
and the `(m, l, o)` carry becomes per-row.

What happens next is the gap. `_coop_walk` raises
`_CoopUnsupported("dot_general with m != 1")` (`emit.py:1613`), so the whole
kernel drops back to one-thread-per-instance, where `_rule_dot_general` is the
naive O(M*N*K) scalar loop with "no threadgroup-memory tiling"
(`emit.py:1084`). The result would be correct and much slower. So the change
is not a `BlockSpec` edit; it is cooperative support for `m > 1`, which needs:

- output sharding over lanes for an `(m, k) @ (k, n)` shape, not just `(1, k)`;
- `threadgroup`-space declarations and barriers — item 1 of
  `docs/notes/simdgroup-matmul-design.md`, built-then-reverted for the `m == 1` case
  where it lost 0.43-0.79x, and now wanted again for the case where the reuse
  actually pays;
- a carry layout for `(m, l, o)` that is per-row rather than per-instance.

This is ROADMAP stretch 12's gating condition — "explicitly gated on an
MMA-heavy kernel entering the benchmark set" — being met from the other
direction. Stretch 12 waited for such a kernel to show up; this says the
benchmark set already has one, and `m == 1` is the only reason it doesn't look
like it.

## Methodology: interleave, or don't bother

The un-interleaved numbers are worse than useless, they are *inverted*. Same
binaries, three orderings, three answers for `blocked8` (device time
throughout, so the rows are comparable):

| how it was run | `blocked8` | `rowwise` |
|---|---|---|
| 4-benchmark pass, cool | 485 us | 1144 us |
| 12-benchmark sequential pass, warm | 1206 us | 963 us |
| rested, random-interleaved, median of 7 | 422 us | 850 us |

The middle row says query blocking is a regression. It is an artifact of
thermal drift across a 12-benchmark run, and taken at face value it would have
killed the whole direction. This is the same hazard already recorded in
`docs/notes/simdgroup-matmul-design.md` (0.523-0.862 ms at 512 for identical code)
and in `_M1_VECTORIZE_MAX_N`'s comment, which is now three independent
sightings. Assume it applies to every GPU comparison in this project, and
prefer device-time (`Batch.gpu_time`) over wall time for kernel comparisons so
the fixed ~130 us floor does not flatter whichever kernel is slower.

## What transfers

The 2.0x **ratio** between shapes. Not the absolutes: `rowwise` is a
hand-written stand-in, and at 850 us of device time it is already
comfortably faster than the ~1.5 ms this repo measures for its emitted
equivalent. That difference is unexplained and worth a look on its own — it
suggests the emitted kernel carries overhead beyond its shape, which no amount
of query blocking will remove.

Two things this scratch does **not** establish: that a *generated* `m == 8`
kernel hits the same 422 us (the emitter's register allocation and index
arithmetic are not the hand-written version's), and that BQ=8 is right at any
other `seq_len` or `head_dim`. Both need re-measuring after the emitter change,
interleaved, before any of this goes in a reward spec.

## References

- Dispatch decomposition, pipeline-depth sweep, dead-end table:
  `metal-runtime/notes/devlog.md`, card "Dispatch overhead: measured, and
  mostly not ours" (2026-08-13).
- The attention kernels and the block-size sweep:
  `metal-runtime/benchmarks/bench_attention.py`; devlog card "Flash attention:
  fusion is maximal, query blocking is the lever".
- `Batch.timestamps` (the driver/queue/GPU/wake split) is new in metal-runtime
  and is what made the `wait()` breakdown reproducible from Python.
