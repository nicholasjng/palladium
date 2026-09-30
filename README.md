# Palladium

Palladium translates a supported subset of JAX Pallas kernels to Metal. With
the [jax-mps](https://github.com/tillahoffmann/jax-mps) plugin selected, it
is the Pallas backend for the `mps` platform: a plain `pl.pallas_call` under
`jax.jit` lowers to a Metal kernel. Without the plugin, `palladium.metal_call`
dispatches the same kernel to Metal from a CPU `jax.jit` program through a
`jax.ffi` target.

## Example

```python
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import palladium  # registers the mps lowering

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
result = jax.jit(call)(x, y)  # x and y are float32 arrays on MPS
```

Other platforms keep JAX's own `pallas_call` behavior; pass `interpret=True`
to run the same kernel on the Pallas interpreter anywhere. Gradients pair a
forward call with a backward one through `palladium.with_vjp`,
`with_auxiliary_vjp`, or `with_reference_vjp`.

## Install

Palladium requires macOS on Apple silicon, Python 3.12+, CMake, Ninja,
and the sibling [metal-runtime](https://github.com/nicholasjng/metal-runtime)
checkout for the CPU-FFI path. In development, check the repositories
out side by side and run:

```sh
uv sync
uv run pytest -q
```

To run on the `mps` platform, install and select the jax-mps plugin separately. Its
platform name is `mps`; the plugin is not installed by this repository.

## Documentation

- [Getting started](docs/getting-started.md): the mps and CPU-FFI call paths,
  with the CPU interpreter as a correctness reference.
- [Supported JAX functionality](docs/supported-jax.md): Pallas constructs,
  primitives, dtypes, and transformation limits.
- [Performance and examples](docs/performance.md): measured workloads,
  timing conditions, and practical guidance.
