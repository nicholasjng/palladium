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

Debugging: `print(palladium.emit_msl(spec))` — it is just text, read it.
The CPU oracle for any kernel is `metal_call(...).interpret`.

## Examples

`01_ode_ensembles.py` (activates after the capstone) - `02_adaptive_lockstep.py`
(runs today, measures the vmap lockstep tax) - `03_sde_montecarlo.py` (runs
today, hand-written MSL Monte Carlo) - `04_reaction_diffusion.py` (baseline
today, Metal version gated on stretch exercise 6).

```sh
uv run python examples/03_sde_montecarlo.py
```
