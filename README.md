# palladium

Pallas kernels on Apple GPU: trace a `pl.pallas_call`, emit Metal Shading
Language, dispatch through [metal-runtime](../metal-runtime). The point is
the benchmark story in `examples/` — hand-written-kernel ODE integration vs
Diffrax — and the emitter that gets us there is deliberately left as
test-driven exercises. See `ROADMAP.md` for the plan.

## Setup

Expects the sibling checkout `../metal-runtime` (path dependency).

```sh
uv sync
uv run pytest -m "not exercise"   # provided baseline: must be green
```

## Working the exercises

```sh
uv run pytest tests/ -x           # first red test = your next exercise
```

The stubs live in `src/palladium/emit.py`; each names its test file, each
test file names its stub, and every docstring carries the hints. Order:

1. `test_02_emit_copy.py` — block load/store
2. `test_03_emit_elementwise.py` — the elementwise table
3. `test_04_grid_blocks.py` — grids, program_id, BlockSpec offsets
4. `test_05_loops.py` — fori_loop -> C for-loop (the one that matters)
5. `test_06_capstone_rk4.py` — batched RK4 Lotka-Volterra, no new rules

Debugging: the emitter's output is just text — read it.
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
- CI (`.github/workflows/ci.yml`) lints, type-checks, and tests on every
  push, plus a weekly canary against jax latest — the early-warning system
  for upstream changes to kernel staging (jax is pinned to one minor for
  exactly that reason). GitHub's hosted macOS runners have no Apple GPU,
  so CI covers the text half only (trace, emitted-MSL snapshots — which is
  everything the canary needs); dispatch correctness is validated locally
  with `uv run pytest` before pushing.
- `uv run mew run` benchmarks the ensemble solve (palladium vs Diffrax),
  filter with `--tag palladium|diffrax`.
- `uv run pytest -m fuzz` differential-fuzzes the emitter: Hypothesis
  generates random kernels (expression trees, loops with permuted
  carries) and diffs Metal against the interpret oracle under SAFE math.
  Excluded from default runs; scale with `PALLADIUM_FUZZ_EXAMPLES`. On a
  failure, Hypothesis shrinks to a minimal kernel — turn it into a named
  regression test. Grow the generators alongside stretch 6: every new
  indexing feature should land with generated coverage from day one.

## Examples

`01_ode_ensembles.py` (activates after the capstone) - `02_adaptive_lockstep.py`
(runs today, measures the vmap lockstep tax) - `03_sde_montecarlo.py` (runs
today, hand-written MSL Monte Carlo) - `04_reaction_diffusion.py` (baseline
today, Metal version gated on stretch exercise 6).

```sh
uv run python examples/03_sde_montecarlo.py
```
