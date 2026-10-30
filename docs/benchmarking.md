# Benchmarking on a new Mac

How to measure palladium on a new machine and compare it with MLX: the
roofline, the row programs and attention with palladium alone, and the
vendor rows and MLX's fused attention through jax-mps. Written for the M5
Max (Apple GPU family 10, the first with matrix accelerators in the GPU),
where the open question is the Metal 4 `matmul2d` ("TensorOps") path.

| Benchmark | Measures | Needs jax-mps |
|---|---|---|
| `bench_roofline.py` | copy bandwidth, simdgroup_matrix peak, Pallas matmul through `matmul2d` and `simdgroup_matrix`, attention at 4096 tokens and 32 heads of 64 | only for the `bench_vendor_matmul` rows (MLX's `jnp.dot`) |
| `bench_row_programs.py` | softmax, layer norm and cumsum as row programs, in GB/s | no |
| `bench_pallas_flash_attention.py` | the `matmul2d` attention schedule across sequence lengths, head sizes and tiles | no |
| `bench_attention_vs_mlx_sdpa.py` | palladium attention under `jax.jit` on `mps` against MLX's fused SDPA | yes, built from the `palladium-dispatch` branch |

Palladium's own rows report GPU time from the command buffer. The vendor
rows report wall time per call, which includes dispatch; that is negligible
for the 4096³ matmuls (around 20 ms per call) but not for small problems.

## 1. Palladium

Requirements: macOS 26 on Apple silicon, [uv](https://docs.astral.sh/uv/),
CMake and Ninja (`brew install cmake ninja`), and network access on the
first build, which fetches Apple's metal-cpp headers. Xcode is not needed:
palladium compiles MSL at runtime.

```sh
git clone https://github.com/nicholasjng/palladium.git
cd palladium
uv sync
uv run pytest -q
uv run python -c "from palladium.device import device_limits; print(device_limits())"
```

The last line prints the device name and `apple_gpu_family`; an M5 should
report 10. On family 10, `dot_general="auto"` keeps `matmul2d` for float32
matmuls and attention; families 7 to 9 prefer `simdgroup_matrix`.

## 2. Palladium-only benchmarks

Keep results outside the checkout, or add `/*.jsonl` to
`.git/info/exclude`, so they are not picked up by `jj` or `git`. Name them
after the machine:

```sh
M=m5max
uv run mew run --random-interleaving benchmarks/bench_roofline.py -o ~/bench/roofline-$M.jsonl
uv run mew run --random-interleaving benchmarks/bench_row_programs.py -o ~/bench/rows-$M.jsonl
uv run mew run --random-interleaving benchmarks/bench_pallas_flash_attention.py -o ~/bench/attention-$M.jsonl
```

Without jax-mps, the roofline's `bench_vendor_matmul` rows are skipped with
"no mps platform"; that is expected. `-k <regex>` selects cases, for
example `-k "float32-n4096|attention"`. Each benchmark checks its results
against NumPy before timing, so a failure there is a correctness bug, not a
slow run.

## 3. jax-mps, for the MLX comparisons

jax-mps runs JAX programs on MLX. The comparisons need the
`palladium-dispatch` branch, which carries the `palladium.dispatch` handler
that lets `pl.pallas_call` reach palladium on the `mps` platform; a stock
jax-mps wheel cannot run `bench_attention_vs_mlx_sdpa.py`. Building it
needs Xcode with its Metal toolchain (MLX compiles its kernels with the
Metal compiler):

```sh
xcrun --find metal      # must print a path; otherwise install Xcode's Metal Toolchain

cd ..                   # jax-mps must sit next to palladium
git clone --branch palladium-dispatch https://github.com/nicholasjng/jax-mps.git
cd jax-mps
./scripts/setup_deps.sh # builds LLVM/MLIR, StableHLO and MLX once; long
cd ../palladium
uv sync --group mps --no-group latest-jax
```

jax-mps pins `jax<0.12`, so the `mps` group conflicts with the default
`latest-jax` group and the sync above moves the environment to jax 0.11.
A plain `uv sync` or `uv run` afterwards would move it back and remove
jax-mps, so pass the same groups to every command:

```sh
export JAX_PLATFORMS=mps,cpu
uv run --group mps --no-group latest-jax python -c "import jax; print(jax.devices('mps'))"
```

[INSTALLATION.md](../INSTALLATION.md) has a smoke test that checks the
`palladium.dispatch` path end to end, and troubleshooting for the jax-mps
build.

Then the comparisons:

```sh
uv run --group mps --no-group latest-jax mew run --random-interleaving \
    benchmarks/bench_roofline.py -o ~/bench/roofline-mps-$M.jsonl
uv run --group mps --no-group latest-jax mew run --random-interleaving \
    benchmarks/bench_attention_vs_mlx_sdpa.py -o ~/bench/sdpa-$M.jsonl
```

Record the MLX version jax-mps was built with
(`third_party/mlx/version.txt` in the jax-mps checkout):
on an M5, newer MLX releases may use the GPU's matrix accelerators, and
that decides which baseline the numbers are compared with.

Return to the default environment with `unset JAX_PLATFORMS` and `uv sync`
when done.

## 4. What to look at on an M5

- **`matmul2d` against `simdgroup_matrix`.** Compare `tensorops-*` and
  `simdgroup-*` rows of `bench_pallas_matmul`, float32 and float16. On the
  M2 and the M4 Pro, `simdgroup` won for float32 (6.66 against 5.82 TFLOPS
  on the M4 Pro). If `matmul2d` reaches the matrix accelerators, it should
  now lead clearly, and float16 `matmul2d` should pass the
  `bench_simdgroup_matrix_peak` row, which measures the older 8x8
  instruction only. If `simdgroup` still wins, family 10 belongs in the
  `"auto"` policy in `emit/kernel.py` like 7 to 9.
- **Attention.** Compare `bench_pallas_attention[tensorops-*]` with
  `[simdgroup-*]`; on the M4 Pro the two were within 4%. On an M5 the
  `matmul2d` schedule may pull ahead, and `bench_pallas_flash_attention.py`
  shows which tiles it prefers.
- **Against MLX.** `bench_vendor_matmul` and `bench_attention_vs_mlx_sdpa.py`
  give MLX's numbers on the same machine.
- **Bandwidth.** `bench_copy_bandwidth` and the row programs set the ceiling
  for everything memory-bound; the row programs ran at the copy
  benchmark's bandwidth on the M2 and the M4 Pro.

## 5. Run hygiene

- Keep the machine on power, close other GPU users, and let it idle a
  minute before starting; `--random-interleaving` spreads thermal drift
  across cases rather than removing it.
- For numbers to publish, add `--repetitions 5` and compare medians.
- Share the `.jsonl` files together with the `device_limits()` line, the
  macOS version (`sw_vers`), and for the jax-mps runs the jax-mps commit and
  MLX version.

Reference numbers from earlier machines are in
[notes/apple-gpu-kernel-recipe.md](../notes/apple-gpu-kernel-recipe.md) and
[performance.md](performance.md).
