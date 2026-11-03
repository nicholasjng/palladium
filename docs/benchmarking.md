# Benchmarking

How to measure palladium on a Mac and compare it with XLA:CPU, jax-mps and
MLX. Every benchmark is a [mew](https://github.com/nicholasjng/mew) suite
under `benchmarks/`: runs write `.jsonl` result files, and `mew compare`
lines rows up across files. The cross-backend benchmarks measure whatever
backend their process has and keep it out of their row names, so the same
file runs once per backend and the comparison is one command.

## The suite

| File | Measures | Backends |
|---|---|---|
| `bench_training.py` | whole training steps, pipelined: the SBI step and its two halves, the census LLM step, the CNF update in plain JAX and with Palladium's adjoint kernels; compile plus first call | default JAX device: `metal`, `cpu`, `mps` |
| `bench_dispatch.py` | per-call overhead (1,024 floats, pipelined) and elementwise bandwidth (256 MiB) | default JAX device |
| `bench_kernels.py` | GEMM in float32, float16 and bfloat16; attention forward and backward; embedding gather and scatter-add | `metal`, `mps`, or MLX directly; skips `cpu` |
| `bench_roofline.py` | copy bandwidth, `simdgroup_matrix` peak, Pallas matmuls through `matmul2d` and `simdgroup_matrix`, `plmtl` matmuls, flash attention; GPU time from the command buffer | Palladium's runtime |
| `bench_row_programs.py` | softmax, layer norm and cumsum as row programs, in GB/s | Palladium's runtime |
| `bench_pallas_flash_attention.py` | the `matmul2d` attention schedule across lengths, head sizes and tiles | Palladium's runtime |
| `bench_cnf_training.py` | CNF updates per variant and checkpoint interval, on `metal` and `cpu`, waiting for each update | `metal,cpu` |
| `bench_rk4_ensemble.py` | the Lotka–Volterra RK4 ensemble as a Pallas kernel, through `metal_call`, and in plain JAX | `metal,cpu` |
| `bench_sir_simulators.py` | the SIR simulator of `examples/sbi_sir.py` in NumPy and PyTorch (eager on CPU and MPS, `torch.compile`) | `uv run --with torch` |

Every benchmark checks its result against NumPy or a JAX reference before
timing, so a failure is a correctness bug, not a slow run.
`benchmarks/conf.py` stamps each run's context with the commit (jj change
and dirty state), `JAX_PLATFORMS` and the installed jax, jaxlib, jax-mps,
mlx and palladium versions; `mew compare` prints them above its table.

On `metal`, `bench_training.py` also reports the plugin's counters per step:
`gpu_fraction` (GPU busy time over wall time), `launches`, and `encode_us`
(host time inside Execute). `bench_kernels.py` reports `flops` per second
and `bench_elementwise` `bytes` per second. `mew run` prints counters under
each row and the `.jsonl` files keep them, but `mew compare` (0.3) compares
times only.

## 1. Setup

Requirements: macOS 26 or later on Apple silicon, [uv](https://docs.astral.sh/uv/),
CMake and Ninja (`brew install cmake ninja`), and network access on the first
build. Xcode is not needed for palladium.

```sh
git clone https://github.com/nicholasjng/palladium.git
cd palladium
uv sync
uv run pytest -q
uv run python -c "from palladium.device import device_limits; print(device_limits())"
```

Keep results outside the checkout so `jj` and `git` ignore them:

```sh
M=m4pro                 # name the files after the machine
R=~/bench/$M && mkdir -p $R
```

## 2. Palladium against XLA:CPU

`JAX_PLATFORMS` decides the backend; `metal,cpu` makes `metal` the default
device and keeps CPU available for setup.

```sh
JAX_PLATFORMS=metal,cpu uv run mew run benchmarks/bench_training.py benchmarks/bench_dispatch.py -o $R/metal.jsonl
JAX_PLATFORMS=cpu       uv run mew run benchmarks/bench_training.py benchmarks/bench_dispatch.py -o $R/cpu.jsonl
uv run mew compare $R/metal.jsonl $R/cpu.jsonl
```

`mew compare` measures every file against the last one, so put the baseline
last. Rows a backend cannot run (Palladium's CNF kernels on CPU) are
skipped and reported as missing.

## 3. jax-mps, from a venv in its own checkout

jax-mps pins `jax<0.12` while palladium needs 0.12, so they do not share an
environment, and the jax-mps rows need no palladium: they are plain JAX
programs. Create the venv inside the jax-mps checkout, add mew to it, and run
palladium's benchmark files with that venv's mew, from palladium's directory:

```sh
cd ../jax-mps
./scripts/setup_deps.sh                   # once: builds LLVM/MLIR, StableHLO and MLX; long
uv sync                                   # builds jax-mps into ../jax-mps/.venv
uv pip install mew-bench numpy
cd ../palladium
JAX_PLATFORMS=mps,cpu ../jax-mps/.venv/bin/mew run \
    benchmarks/bench_training.py benchmarks/bench_dispatch.py benchmarks/bench_kernels.py \
    -o $R/mps.jsonl
uv run mew compare $R/metal.jsonl $R/mps.jsonl
```

Building jax-mps from source needs Xcode with its Metal toolchain (`xcrun
--find metal` must print a path). Without it, a released wheel works the
same way: `uv venv ../jax-mps/.venv && uv pip install --python
../jax-mps/.venv jax-mps mew-bench numpy`. The context line shows which
jax-mps and jax each run used. jax-mps cannot run the CNF update with
Palladium's kernels, which needs palladium; that row is skipped.

## 4. MLX directly

`bench_kernels.py` runs on MLX itself. MLX does not depend on jax, so it
can run from palladium's venv: `uv run --with mlx` adds it for one command
without changing the lock, and `BENCH_BACKEND=mlx` makes the file use MLX
even though jax is installed:

```sh
BENCH_BACKEND=mlx uv run --with mlx mew run benchmarks/bench_kernels.py -o $R/mlx.jsonl
uv run mew compare -k bench_kernels $R/metal.jsonl $R/mps.jsonl $R/mlx.jsonl
```

In that mode the file does not import jax, so the plugin is not loaded. A
venv with `mlx` and no jax needs no `BENCH_BACKEND`.

The `metal` side of `bench_kernels.py` comes from the run in section 2 if it
included the file; otherwise add it to that command. Matmuls on `metal` go
through the plugin's GEMM (Palladium's Pallas emitter), the attention
forward through Palladium's flash attention, and its backward and the
embedding rows through the plugin's compiler. MLX uses `mx.matmul`,
`mx.fast.scaled_dot_product_attention` and its gradient, indexing and
`.at[].add`. Record `mlx` from the context line: on GPUs with matrix
accelerators (M5 and later), newer MLX releases may use them.

## 5. Interleave and repeat

Thermal and load drift bias a run that does all of one backend first. For
numbers to keep, alternate the backends and append the repetitions:

```sh
for i in 1 2 3; do
  JAX_PLATFORMS=metal,cpu uv run mew run benchmarks/bench_training.py --append -o $R/metal.jsonl
  JAX_PLATFORMS=cpu       uv run mew run benchmarks/bench_training.py --append -o $R/cpu.jsonl
done
uv run mew compare $R/metal.jsonl $R/cpu.jsonl
```

`mew compare` reduces repetitions to their median (`--statistic` picks
another). Within one file, `--random-interleaving` shuffles the cases.

## 6. Checklist for the M4 Pro (M2 close-out and M0)

M2 needs its throughput numbers on the milestone machine, and M0 its
baselines (`notes/metal-compute-stack.md`):

1. Sections 2, 3 and 4 above, including `bench_kernels.py` on all three
   backends.
2. The roofline and the row programs:
   ```sh
   uv run mew run --random-interleaving benchmarks/bench_roofline.py -o $R/roofline.jsonl
   uv run mew run --random-interleaving benchmarks/bench_row_programs.py -o $R/rows.jsonl
   ```
3. The long-run agreement (not a timing benchmark; writes a JSON record):
   ```sh
   for w in sbi cnf; do
     JAX_PLATFORMS=metal,cpu uv run python examples/pjrt_long_run.py --workload $w \
         --output notes/metal-long-run-$w-$M-$(date +%F).json
   done
   ```

Send the `.jsonl` files and the two long-run records back; the context lines
carry the machine, commit and versions.

## 7. What to look at

- **Overhead and pipelining.** `bench_dispatch.py::bench_small_call` is the
  per-call floor; `gpu_fraction` in `bench_training.py` near 1 means the GPU,
  not the host, limits a step.
- **`matmul2d` against `simdgroup_matrix`.** The `tensorops-*` and
  `simdgroup-*` rows of `bench_roofline.py::bench_pallas_matmul`. On the M2
  and the M4 Pro, `simdgroup` won for float32 (6.66 against 5.82 TFLOPS on
  the M4 Pro). On a GPU with matrix accelerators, `matmul2d` should lead,
  and float16 `matmul2d` should pass `bench_simdgroup_matrix_peak`, which
  measures the older 8x8 instruction only.
- **Against MLX.** `bench_kernels.py` shows where the plugin stands on
  M3's kernels: GEMM per dtype, attention forward and backward, and
  scatter-add. `gpu_us` near the wall time means the kernel, not dispatch
  or allocation, sets a `metal` row. On the M2 (2026-11-03), with the
  backward kernels, Palladium's attention gradient was 4% (full) and 27%
  (causal) faster than MLX's, and its forward 4–8% faster.
- **Bandwidth.** `bench_copy_bandwidth` and the row programs set the ceiling
  for memory-bound work; `bench_elementwise` should come close to it.

## Run hygiene

- Keep the machine on power, close other GPU users, and let it idle a
  minute before starting.
- For numbers to publish, repeat as in section 5 and compare medians.
- Share the `.jsonl` files with the macOS version (`sw_vers`) and the
  `device_limits()` line.

Reference numbers from earlier machines are in
[notes/apple-gpu-kernel-recipe.md](../notes/apple-gpu-kernel-recipe.md),
[notes/metal-pjrt-plugin.md](../notes/metal-pjrt-plugin.md) and
[performance.md](performance.md).
