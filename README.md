# palladium

Hand-authored ODE/SDE integrator kernels on Apple GPU: trace a
`pl.pallas_call`, emit Metal Shading Language, dispatch through
[metal-runtime](https://github.com/nicholasjng/metal-runtime). You
write an ordinary [Pallas](https://docs.jax.dev/en/latest/pallas/index.html)
kernel — control flow via `fori_loop`/`scan`/`while_loop` over one
thread per program instance — and palladium turns it into a compiled
Metal kernel behind a NumPy-in/NumPy-out callable, or a jax.ffi
primitive that composes inside `jax.jit`.

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

The bundled Lotka-Volterra RK4 ensemble example runs tens of times
faster than `jax.jit(vmap(diffeqsolve))` on CPU at N=100,000 (M1 Pro);
see `docs/performance.md` for the measurement discipline behind
comparisons like this.

Note: an earlier iteration of this project also grew a cooperative
SIMD-group/MMA GEMM lowering (matmuls, flash attention, MLP training).
That capability outgrew being a side feature and was split out into its
own sibling project, `mgemm`; palladium stays scoped to hand-authored
control-flow kernels.

## Setup

Requires macOS on Apple silicon, Python 3.12+, CMake and Ninja
(`brew install cmake ninja`). metal-runtime is currently consumed as a
sibling path dependency (`[tool.uv.sources]` in `pyproject.toml`), so
check both repos out next to each other:

```sh
git clone https://github.com/nicholasjng/metal-runtime
git clone <this repo> palladium && cd palladium
uv sync          # builds metal-runtime and the native FFI handler
uv run pytest -q # must be green on a Metal-capable Mac
```

## Documentation

- [Getting started](docs/getting-started.md): first kernel, the
  interpret-oracle workflow, `explain()`, debugging hooks, the error
  taxonomy.
- [Supported subset](docs/supported-subset.md): the contract. Which
  primitives, dtypes, and kernel structures are supported.
- [Performance guide](docs/performance.md): execution models, the
  dispatch floor, `pin()`, FFI overhead, math modes, and how to measure
  on thermally sensitive hardware.
- [Extending](docs/extending.md): registering lowering rules for new
  primitives with `palladium.rule`.

Versioning: pre-1.0 semver (minor may break the public API, patch may
not); the public API is `palladium.__all__`. jax is pinned to a tested
range (`>=0.11,<0.12`) and a weekly CI canary tests jax head before the
pin widens. History in [CHANGELOG.md](CHANGELOG.md).

## Integrating with JAX

`metal_call` is eager and NumPy-based, so its result cannot sit inside
a jitted computation. `metal_call_jit` registers the kernel as a
jax.ffi target instead: CPU stays the JAX-visible platform, Metal is
reached through the FFI escape hatch, and the call is an ordinary JAX
primitive, traceable and composable with surrounding `jax.numpy` code.

```python
call = palladium.metal_call_jit(kernel, out_shape=jax.ShapeDtypeStruct((8, 8), jnp.float32))

@jax.jit
def composed(x, y):
    return jnp.sum(call(x, y) ** 2)  # a real GPU dispatch inside a jit trace
```

One generic native handler (`native/ffi/palladium_ffi.cpp`) backs every
kernel: MSL source, entry point, launch geometry, and math mode travel
as FFI attributes. It is built by `uv sync` via the root
`CMakeLists.txt`; `PALLADIUM_FFI_LIBRARY` overrides the path for an
out-of-tree build. `jax.grad` works by pairing a backward kernel via
`jax.custom_vjp` (`metal_call_jit` kernels have no JVP/transpose rule of
their own); `jax.vmap` works with `vmap_method="sequential"` (one
dispatch per batch element; a batch dim in the Pallas grid is the fast
path).

## Examples

- `01_ode_ensembles.py`: batched RK4 Lotka-Volterra, palladium vs
  `jit(vmap(diffeqsolve))`.
- `02_adaptive_lockstep.py`: the vmap lockstep tax. Diffrax vs a
  per-thread adaptive controller (Bogacki-Shampine 3(2), PI control)
  with divergent per-thread step counts.
- `03_sde_montecarlo.py`: GBM Monte Carlo with in-kernel counter-based
  RNG, validated against Black-Scholes.
- `04_reaction_diffusion.py`: Gray-Scott stencil via indexed ref
  access.
- `05_df32_precision.py`: compensated (df32) arithmetic under
  `math_mode=SAFE`, measured against a float64 reference.

```sh
uv run python examples/01_ode_ensembles.py
```

## Development machinery

- Every kernel-lowering test diffs the GPU result against
  `interpret=True`, the CPU oracle; rules are verified by sabotage
  (break the rule, watch the test fail), not by passing.
- `tests/golden/` pins emitted MSL for canonical kernels; regenerate
  after intended emitter changes with `PALLADIUM_REGEN_GOLDEN=1
  uv run pytest tests/test_msl_snapshots.py` and review the diff.
- `uv run pytest -m fuzz` differential-fuzzes the emitter with
  Hypothesis-generated kernels (expression trees, loops with permuted
  carries, control flow) against the oracle under SAFE math; scale with
  `PALLADIUM_FUZZ_EXAMPLES`. It has caught real bugs, including a Metal
  compiler miscompile (`docs/notes/emitter-simplifications.md`).
- `uv run mew run` benchmarks the ensemble solve, filter with
  `--tag palladium|diffrax`.
- `docs/notes/` is the development lab notebook (design explorations,
  measurement records); `ROADMAP.md` the longer arc.
