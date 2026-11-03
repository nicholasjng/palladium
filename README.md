# Palladium

Palladium runs JAX on Apple GPUs without MLX. It has two parts:

- a PJRT plugin, `metal`, that compiles whole `jax.jit` programs (StableHLO)
  to Metal kernels and runs them asynchronously on unified memory;
- a translator from JAX Pallas kernels to Metal, which makes a plain
  `pl.pallas_call` under `jax.jit` a Metal kernel inside those programs.

Without the plugin, `palladium.metal_call` dispatches the same kernel to Metal
from a CPU `jax.jit` program through a `jax.ffi` target.

## Example

```python
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import palladium

def saxpy(x_ref, y_ref, out_ref):
    out_ref[...] = 2.0 * x_ref[...] + y_ref[...]

n = 4096
block = pl.BlockSpec((1,), lambda i: (i,))
call = pl.pallas_call(
    saxpy,
    grid=(n,),
    in_specs=(block, block),
    out_specs=block,
    out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
    compiler_params=palladium.CompilerParams(),  # optional Metal-side knobs
)
result = jax.jit(call)(x, y)  # x and y are float32 arrays on the metal device
```

Run with `JAX_PLATFORMS=metal,cpu`; the plugin registers only when
`JAX_PLATFORMS` names `metal`, and `metal` first makes it the default device.

Other platforms keep JAX's own `pallas_call` behavior; pass `interpret=True`
to run the same kernel on the Pallas interpreter anywhere. Gradients pair a
forward call with a backward one through `palladium.with_vjp`.

## Install

Palladium requires macOS on Apple silicon, Python 3.12+, CMake and Ninja.
The build compiles its Metal runtime (`native/runtime`) and fetches Apple's
metal-cpp headers. Run:

```sh
uv sync
uv run pytest -q
```

Palladium requires jax 0.12. See [Installation](INSTALLATION.md) for a smoke
test of both paths, and [Benchmarking](docs/benchmarking.md) for measuring
the plugin against XLA:CPU, jax-mps and MLX.

## Documentation

- [Architecture and interview walkthrough](docs/architecture.md): building
  blocks, JAX Pallas integration, both dispatch paths, and a presentation route.
- [Emitter walkthrough](docs/emitter.md): jaxpr lowering, Ref addressing,
  control flow, optimizations, and cooperative TensorOps.
- [API guide](docs/api.md): public entry points, metadata, diagnostics, and errors.
- [Getting started](docs/getting-started.md): Pallas on the `metal` plugin and
  through CPU FFI, with the CPU interpreter as a correctness reference.
- [Supported JAX functionality](docs/supported-jax.md): Pallas constructs,
  primitives, dtypes, and transformation limits.
- [Performance and examples](docs/performance.md): measured workloads,
  timing conditions, and practical guidance.
- [Benchmarking](docs/benchmarking.md): the mew suite and cross-backend
  comparisons.
- [The metal PJRT plugin](notes/metal-pjrt-plugin.md): implementation record,
  status and measurements.
