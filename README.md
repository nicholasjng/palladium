# palladium

Pallas kernels on Apple GPU: trace a `pl.pallas_call`, emit Metal Shading Language,
dispatch through [metal-runtime](https://github.com/nicholasjng/metal-runtime).
The result is hand-written kernels running faster than Diffrax on ODE integration, measured in `examples/`.

## Setup

```sh
uv sync
uv run pytest -m "not exercise"   # baseline sanity check: must be green
```

`uv sync` also compiles `native/ffi/palladium_ffi.cpp` (scikit-build-core,
root `CMakeLists.txt`); needs CMake and Ninja on `PATH` (`brew install
cmake ninja`), same as metal-runtime.

metal-runtime is a git dependency (`pyproject.toml`), no sibling checkout needed.
Developing metal-runtime itself: point `[tool.uv.sources]` at a local `path` instead of a git revision.

## Status

Exercises 1-5 (block load/store, elementwise, grids, loops, the RK4 capstone) and stretches 6-10
(indexed ref access, per-thread adaptive stepping, counter-based RNG, df32 precision, jax.ffi
integration) are done:

```sh
uv run pytest tests/                              # everything green
uv run python examples/02_adaptive_lockstep.py     # the headline result
```

`02_adaptive_lockstep.py` runs a Van der Pol ensemble through Diffrax's
`vmap`-synchronized adaptive solve and palladium's per-thread adaptive
kernel (Bogacki-Shampine 3(2), FSAL, PI controller), and prints
wall-clock and step-count comparisons for both.

## Integrating with JAX

`metal_call(...)` is NumPy-in, NumPy-out: it traces a kernel once,
compiles it, and runs it eagerly. `MetalCallable.__call__` calls
`np.asarray` on its arguments, which fails the moment one is a tracer,
so a `metal_call` result cannot sit inside a jitted computation.

`metal_call_jit(...)` (`src/palladium/ffi.py`) is the fix: same
`pl.pallas_call`-shaped entry point, but dispatch happens through
`jax.ffi.ffi_call` against a registered FFI target instead of eagerly.
CPU stays the JAX-visible platform (`register_ffi_target(...,
platform="cpu")`); Metal is reached through the FFI escape hatch, no new
PJRT client needed. The result is an ordinary JAX primitive: traceable,
jittable, composable with surrounding `jax.numpy` code.

```python
call = palladium.metal_call_jit(kernel, out_shape=jax.ShapeDtypeStruct((8, 8), jnp.float32))

@jax.jit
def composed(x, y):
    return jnp.sum(call(x, y) ** 2)  # a real GPU dispatch inside a jit trace
```

One generic native handler (`native/ffi/palladium_ffi.cpp`) backs every
palladium kernel: MSL source, entry point name, grid, and math mode
travel as FFI attributes (baked into the HLO at trace time), and
`RemainingArgs`/`RemainingRets` cover any input/output count. Nothing to
build separately: the repo root's `CMakeLists.txt` (scikit-build-core)
compiles it and ships `libpalladium_ffi.dylib` inside the `palladium`
package as part of `uv sync`/`pip install`, the same way metal-runtime
ships `libmetal_runtime_c.dylib`. `metal_call_jit` finds it via
`importlib.resources`; `PALLADIUM_FFI_LIBRARY` overrides the path for an
out-of-tree build. Not differentiable: `ffi_call` has no JVP/transpose
rule by default.

## How the emitter was built

Test-driven, one primitive group at a time. The stubs live in
`src/palladium/emit.py`; each names its test file, each test file names
its stub, and every docstring carries the hints.

1. `test_02_emit_copy.py`: block load/store
2. `test_03_emit_elementwise.py`: the elementwise table
3. `test_04_grid_blocks.py`: grids, program_id, BlockSpec offsets
4. `test_05_loops.py`: fori_loop -> C for-loop (the one that matters)
5. `test_06_capstone_rk4.py`: batched RK4 Lotka-Volterra, no new rules
6. `test_07_preflight.py` / `test_08_adaptive.py`: comparisons and
   select_n, then the adaptive controller kernel built on top of them
7. `test_09_indexed_refs.py`: indexed ref access, `x_ref[i, j]`
8. `test_10_random_bits.py`: counter-based RNG, `jax.random` inside a kernel

`uv run pytest tests/ -x` finds the first red test when a new stretch is scaffolded.

Debugging: the emitter's output is just text, read it.
`print(palladium.debug_msl(kernel, *args, **pallas_kwargs))` is the
one-call version; `PALLADIUM_DUMP_MSL=1` (or `=<dir>`) dumps every
kernel's source at compile time, and Metal compile errors arrive with the
line-numbered source attached. The CPU oracle for any kernel is
`metal_call(...).interpret`.

## Machinery

- `tests/golden/` pins the emitted MSL for three canonical kernels;
  regenerate after intended emitter changes with
  `PALLADIUM_REGEN_GOLDEN=1 uv run pytest tests/test_msl_snapshots.py`
  and review the diff.
- `uv run mew run` benchmarks the ensemble solve (palladium vs Diffrax),
  filter with `--tag palladium|diffrax`.
- `uv run pytest -m fuzz` differential-fuzzes the emitter: Hypothesis
  generates random kernels (expression trees, loops with permuted
  carries) and diffs Metal against the interpret oracle under SAFE math.
  Excluded from default runs; scale with `PALLADIUM_FUZZ_EXAMPLES`. On a
  failure, Hypothesis shrinks to a minimal kernel, turn it into a named
  regression test. Doesn't generate indexed-ref kernels yet: stretch 6
  landed without generated coverage, still worth adding.

## Examples

- `01_ode_ensembles.py`: batched RK4 Lotka-Volterra, palladium vs
  jit(vmap(diffeqsolve)).
- `02_adaptive_lockstep.py`: the lockstep tax. Diffrax vs palladium's
  per-thread adaptive controller on a Van der Pol ensemble.
- `03_sde_montecarlo.py`: GBM Monte Carlo, hand-written MSL (pcg_hash)
  next to a Pallas-authored version (`jax.random`, Threefry-2x32-20) for
  comparison, both validated against Black-Scholes.
- `04_reaction_diffusion.py`: Gray-Scott stencil, one thread per grid
  point, halo reads via indexed ref access. Beats the CPU baseline.
- `05_df32_precision.py`: does compensated arithmetic (df32) buy real
  accuracy on the RK4 capstone. FAST vs SAFE vs df32, all measured
  against a true float64 reference, not against each other.

```sh
uv run python examples/02_adaptive_lockstep.py
```
