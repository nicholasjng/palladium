# Installation

Palladium needs an Apple silicon Mac with GPU access, macOS 26 or later,
Python 3.12+, [uv](https://docs.astral.sh/uv/), CMake and Ninja. It requires
jax 0.12. Xcode is not needed: Palladium compiles Metal Shading Language at
runtime, and the build fetches Apple's metal-cpp headers on its first run.

```sh
brew install cmake ninja
git clone https://github.com/nicholasjng/palladium.git
cd palladium
uv sync
uv run pytest -q
```

`uv sync` builds the native libraries: the Metal runtime (`native/runtime`),
the CPU FFI target (`native/ffi`) and the `metal` PJRT plugin
(`native/pjrt`). uv does not rebuild for C++-only changes; after editing
`native/`, run `uv sync --reinstall-package palladium`. The installed library
is stripped; `uv sync --reinstall-package palladium -C install.strip=false`
keeps its symbols for Instruments or `sample`.

## Select the plugin

JAX loads the `metal` plugin only when `JAX_PLATFORMS` names it. Put it first
to make it the default device, and keep `cpu` for host-side setup:

```sh
export JAX_PLATFORMS=metal,cpu
uv run python -c "import jax; print(jax.devices())"
```

## Smoke test

Save this as `smoke.py` in the checkout. It runs one Pallas kernel both ways:
as a plain `pl.pallas_call` inside `jax.jit` on the `metal` plugin, and through
`palladium.metal_call` from a CPU program.

```python
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np

import palladium


def saxpy(x_ref, y_ref, out_ref):
    out_ref[...] = 2.0 * x_ref[...] + y_ref[...]


n = 4096
block = pl.BlockSpec((256,), lambda i: (i,))
kwargs = dict(
    grid=(n // 256,),
    in_specs=(block, block),
    out_specs=block,
    out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
)
x, y = np.arange(n, dtype=np.float32), np.ones(n, np.float32)
expected = 2 * x + y

# 1. pl.pallas_call inside jit on the metal plugin.
metal = jax.devices("metal")[0]
call = pl.pallas_call(saxpy, **kwargs)
lowered = jax.jit(call).trace(x, y).lower(lowering_platforms=("metal",)).as_text()
assert "palladium_dispatch" in lowered, "pallas_call did not lower to Palladium's kernel"
result = jax.jit(call)(jax.device_put(x, metal), jax.device_put(y, metal))
assert result.devices() == {metal}
np.testing.assert_array_equal(np.asarray(result), expected)

# 2. metal_call from a CPU program, through the FFI target.
cpu = jax.devices("cpu")[0]
result = jax.jit(palladium.metal_call(saxpy, **kwargs))(
    jax.device_put(x, cpu), jax.device_put(y, cpu)
)
np.testing.assert_array_equal(np.asarray(result), expected)
print("Palladium kernels ran on the metal plugin and through CPU FFI")
```

```sh
JAX_PLATFORMS=metal,cpu uv run python smoke.py
```

## Run an example

The CNF example trains a continuous normalizing flow with Palladium's
forward and adjoint kernels inside a jitted Adam update, and evaluates the
result with plain JAX on CPU:

```sh
JAX_PLATFORMS=metal,cpu uv run python examples/cnf_density.py --iterations 500
uv run python examples/cnf_density.py --device cpu   # the kernels through metal_call
```

[Benchmarking](docs/benchmarking.md) covers the benchmark suite, including
comparisons with jax-mps and MLX, which run from their own environments.

## If something fails

- **`Unknown backend metal`:** `JAX_PLATFORMS` does not name `metal`, or it was
  set after jax initialized its backends; set it before starting Python.
- **`UNIMPLEMENTED: metal: …`:** the program uses an operation the plugin does
  not compile yet; the message names it. `notes/metal-pjrt-plugin.md` lists
  what is supported.
- **Metal reports no device:** run in a session with access to the GPU. A
  restrictive sandbox may allow device discovery but block execution.
- **Native changes have no effect:** rebuild with
  `uv sync --reinstall-package palladium`.
