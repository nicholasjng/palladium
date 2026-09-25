# Palladium

Palladium translates a supported subset of JAX Pallas kernels to Metal. Its
main integration is [`mps_call_jit`](docs/supported-jax.md), which sends the
generated kernel to the jax-mps backend. It also provides eager `metal_call`
and a CPU-FFI `metal_call_jit` path.

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
call = palladium.mps_call_jit(
    saxpy,
    grid=(n,),
    in_specs=(block, block),
    out_specs=block,
    out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
    fallback="error",
)
result = jax.jit(call)(x, y)  # x and y are float32 arrays on MPS
```

`fallback="error"` requires the call to lower on MPS. Without it, independent
kernels may use Pallas's interpreter on other platforms. Cooperative kernels
always require MPS execution.

## Install

Palladium currently requires macOS on Apple silicon, Python 3.12+, CMake, Ninja,
and the sibling [metal-runtime](https://github.com/nicholasjng/metal-runtime)
checkout for its eager and CPU-FFI paths. In development, check the repositories
out side by side and run:

```sh
uv sync
uv run pytest -q
```

To use `mps_call_jit`, install and select the jax-mps plugin separately. Its
platform name is `mps`; the plugin is not installed by this repository.

## Documentation

- [Getting started](docs/getting-started.md): eager, FFI, and MPS call paths,
  with the CPU interpreter as a correctness reference.
- [Supported JAX functionality](docs/supported-jax.md): Pallas constructs,
  primitives, dtypes, and transformation limits.
- [Performance and examples](docs/performance.md): measured workloads,
  timing conditions, and practical guidance.
