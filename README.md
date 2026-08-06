# palladium

Hand-authored ODE/SDE integrator kernels on Apple GPU. Write an ordinary
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) kernel — refs,
BlockSpecs, `jnp` ops, one thread per program instance, control flow via
`fori_loop`/`scan`/`while_loop` — and palladium traces it, emits Metal
Shading Language, and dispatches it through
[metal-runtime](https://github.com/nicholasjng/metal-runtime).

```python
import jax, jax.numpy as jnp
import palladium

def rk4_step(y_ref, dt_ref, o_ref):
    ...  # ordinary Pallas: refs, BlockSpecs, jnp ops, fori_loop/scan/while

call = palladium.metal_call(rk4_step, grid=..., in_specs=..., out_shape=...)
out = call(y0, dt)                       # eager, NumPy in/out
oracle = call.interpret(y0, dt)          # the same kernel on the CPU interpreter
print(call.explain(y0, dt))              # launch geometry, emitted MSL size
```

Two measured results, both reproducible from `examples/` on an M1 Pro:

- **Throughput.** A 100,000-member Lotka-Volterra RK4 ensemble runs tens
  of times faster than `jax.jit(vmap(diffeqsolve))` on CPU, which on
  macOS is the practical Diffrax deployment.
- **Accuracy at long horizons.** Over 5,000,000 Verlet steps of the
  Kepler problem, float32 energy error climbs as `sqrt(steps)` — the
  random walk of round-off, fitted slope +0.50. The same integrator in
  compensated df32 stays flat on its own O(h²) truncation bound.

`docs/performance.md` documents the measurement discipline behind
comparisons like these.

## Setup

macOS on Apple silicon, Python 3.12+, CMake and Ninja
(`brew install cmake ninja`). metal-runtime is a sibling path dependency
(`[tool.uv.sources]`), so check both repos out next to each other:

```sh
git clone https://github.com/nicholasjng/metal-runtime
git clone https://github.com/nicholasjng/palladium && cd palladium
uv sync          # builds metal-runtime and the native FFI handler
uv run pytest -q # must be green on a Metal-capable Mac
```

## Documentation

- [Getting started](docs/getting-started.md) — first kernel, the
  interpret-oracle workflow, `explain()`, error taxonomy.
- [Supported subset](docs/supported-subset.md) — the contract: primitives,
  dtypes, kernel structures.
- [Performance](docs/performance.md) — execution model, dispatch floor,
  `pin()`, FFI overhead, math modes.
- [Extending](docs/extending.md) — new lowering rules via `palladium.rule`.

Pre-1.0 semver: minor may break the public API (`palladium.__all__`),
patch may not. jax is pinned to a tested range (`>=0.11,<0.12`).

## Integrating with JAX

`metal_call` is eager, so its result cannot sit inside a jitted
computation. `metal_call_jit` registers the kernel as a jax.ffi target
instead — CPU stays the JAX-visible platform, Metal is reached through
the FFI escape hatch — and the call becomes a traceable JAX primitive.

```python
call = palladium.metal_call_jit(kernel, out_shape=jax.ShapeDtypeStruct((8, 8), jnp.float32))

@jax.jit
def composed(x, y):
    return jnp.sum(call(x, y) ** 2)  # a real GPU dispatch inside a jit trace
```

One generic native handler (`native/ffi/palladium_ffi.cpp`) backs every
kernel; MSL source, entry point, launch geometry and math mode travel as
FFI attributes. `uv sync` builds it; `PALLADIUM_FFI_LIBRARY` overrides
the path out of tree. `jax.grad` needs a backward kernel paired via
`jax.custom_vjp` (there is no JVP/transpose rule); `jax.vmap` needs
`vmap_method="sequential"`, one dispatch per batch element, with a batch
dim in the Pallas grid as the fast path.

## Examples

```sh
uv run python examples/01_ode_ensembles.py
```

1. `01_ode_ensembles.py` — batched RK4 Lotka-Volterra vs
   `jit(vmap(diffeqsolve))`.
2. `02_adaptive_lockstep.py` — the vmap lockstep tax: Diffrax against a
   per-thread adaptive controller (Bogacki-Shampine 3(2), PI control).
3. `03_sde_montecarlo.py` — GBM Monte Carlo with in-kernel counter-based
   RNG, validated against Black-Scholes.
4. `04_reaction_diffusion.py` — Gray-Scott stencil via indexed ref access.
5. `05_df32_precision.py` — compensated (df32) arithmetic under
   `math_mode=SAFE`, measured against a float64 reference.
6. `06_symplectic_longrun.py` — Kepler over millions of steps: bounded
   energy error vs secular drift, then the round-off walk in float32
   against a flat df32 line.

## Development machinery

- Every kernel-lowering test diffs the GPU result against
  `interpret=True`, the CPU oracle. Rules are verified by sabotage —
  break the rule, watch the test fail — not by passing.
- `tests/golden/` pins emitted MSL. Regenerate with
  `PALLADIUM_REGEN_GOLDEN=1 uv run pytest tests/test_msl_snapshots.py`
  and review the diff.
- `uv run pytest -m fuzz` differential-fuzzes the emitter against the
  oracle under SAFE math (Hypothesis-generated expression trees, loops
  with permuted carries, control flow); scale with
  `PALLADIUM_FUZZ_EXAMPLES`. It has caught real bugs, including a Metal
  compiler miscompile (`docs/notes/emitter-simplifications.md`).
- `uv run mew run` benchmarks the ensemble solve; filter with
  `--tag palladium|diffrax`.
