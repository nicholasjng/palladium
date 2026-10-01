# Installation: Palladium with jax-mps

This guide sets up the `pl.pallas_call` → `palladium.dispatch` → Metal path used
in the CNF examples. The PJRT handler is C++ code in a modified jax-mps plugin:
installing Palladium alone cannot add it to an existing jax-mps binary.

The measured setup used Python 3.14, JAX/jaxlib 0.11.2, and jax-mps 0.11.1 on
an M1 Pro running macOS 26.7.1. The package metadata requires Python 3.12+ and
JAX/jaxlib 0.11.x. The instructions below pin the source revisions used for
those measurements; they are a development setup, not a stock PyPI install.

## Prerequisites

Use an Apple silicon Mac with GPU access, [uv](https://docs.astral.sh/uv/),
Homebrew, and a recent Xcode installation with its Metal toolchain available.
Check that the selected developer directory provides both the SDK and Metal
compiler:

```sh
xcode-select -p
xcrun --show-sdk-path
xcrun --find metal
brew install cmake ninja
```

If `xcrun` cannot find `metal`, select your Xcode installation and install its
Metal Toolchain component before continuing. The CNF example uses ordinary
float32 kernels; newer TensorOps examples have additional hardware/toolchain
requirements.

## Check out the sources side by side

Palladium resolves `metal-runtime` from a sibling directory. These commands
create a fresh checkout layout:

```sh
mkdir -p ~/Workspaces/palladium-demo
cd ~/Workspaces/palladium-demo

git clone https://github.com/nicholasjng/metal-runtime.git
git clone --branch sbi-simulator https://github.com/nicholasjng/palladium.git
git clone --branch palladium-dispatch https://github.com/nicholasjng/jax-mps.git

git -C palladium checkout 03f39dc0ff894104b20cd56c410e1c9ba5af07e4
git -C jax-mps checkout 6bf58877d252965c15202d92b4dbc6149b55e247
```

The jax-mps revision contains the `palladium.dispatch` handler in
`src/pjrt_plugin/ops/control_flow.cc`. A stock plugin with the same package
version is not sufficient: the version number does not identify this patch.

## Build the native dependencies once

From the jax-mps checkout:

```sh
cd ~/Workspaces/palladium-demo/jax-mps
./scripts/setup_deps.sh
```

This builds the pinned LLVM/MLIR, StableHLO, Protobuf/Abseil, and MLX dependencies
and installs them under `~/.local/jax-mps-deps`. It downloads and compiles a
substantial amount of source; allow time and disk space for the first build.
The script reuses matching installed dependencies on subsequent runs.

The dependency revisions are tied to jaxlib's StableHLO bytecode format. Keep
JAX and jaxlib on the supported 0.11.x line rather than independently upgrading
them. For a custom installation prefix, pass `--prefix /your/path` and set
`CMAKE_PREFIX_PATH` to that path when building the plugin.

## Install Palladium and compile the modified plugin

Install everything into Palladium's environment:

```sh
cd ~/Workspaces/palladium-demo/palladium
uv sync --python 3.14

uv pip install --python .venv/bin/python \
    --editable ../jax-mps --reinstall-package jax-mps
```

`uv sync` builds Palladium and its sibling `metal-runtime` dependency. The
second command compiles the modified PJRT plugin and installs it into that
same environment. Repeat that command after changes to jax-mps C++ sources;
an editable installation does not automatically rebuild C++ code.

Use `.venv/bin/python` for the commands below. A later `uv sync`, or `uv run`
that synchronizes automatically, can remove a manually installed plugin if
jax-mps is not selected in the project's dependency groups. `uv run --no-sync`
is another way to use this already prepared environment.

Restart any Python process or notebook kernel after reinstalling the plugin:
a process that already loaded the old native library will keep using it.

## Verify the custom-call path

Save this as `smoke_mps.py` in the Palladium checkout:

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import numpy as np
import palladium  # Registers the pallas_call lowering.


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
)

with jax.default_device(jax.devices("mps")[0]):
    x = jnp.arange(n, dtype=jnp.float32)
    y = jnp.ones(n, dtype=jnp.float32)
    compiled = jax.jit(call)
    ir = str(compiled.lower(x, y).compiler_ir(dialect="stablehlo"))
    assert "palladium.dispatch" in ir, "Palladium lowering was not selected"
    result = jax.block_until_ready(compiled(x, y))
    assert result.devices() == {jax.devices("mps")[0]}
    np.testing.assert_allclose(np.asarray(result), 2 * np.arange(n) + 1)

print("Palladium custom call executed successfully on MPS")
```

Run it with both backends available:

```sh
JAX_PLATFORMS=mps,cpu .venv/bin/python smoke_mps.py
```

Listing an MPS device alone does not prove the custom handler is present. This
check verifies both the emitted `palladium.dispatch` operation and its execution
on the GPU.

## Run the CNF example and benchmark

From the Palladium checkout:

```sh
JAX_PLATFORMS=mps,cpu .venv/bin/python examples/cnf_density.py \
    --device mps --variant reverse --n 16384 --iterations 500

JAX_PLATFORMS=mps,cpu .venv/bin/mew run benchmarks/bench_cnf_training.py \
    --repetitions 5 --random-interleaving --min-time 1s
```

The bundled benchmark uses 256 points; `--n 16384` above applies only to the
training example. The example also evaluates the trained weights using CPU
JAX, which is why `cpu` remains in `JAX_PLATFORMS`.

## If something fails

- If JAX cannot find `mps`, check that jax-mps was installed into the interpreter
  you are running. Inspect it with `.venv/bin/python -c
  'import jax_plugins.mps; print(jax_plugins.mps.__file__)'`.
- If lowering succeeds but execution rejects `palladium.dispatch`, rebuild the
  patched jax-mps checkout into that environment and restart Python.
- If CMake cannot find MLIR, StableHLO, Protobuf, or MLX, finish
  `scripts/setup_deps.sh` and check the dependency prefix.
- If StableHLO deserialization fails, check the JAX/jaxlib versions against the
  revisions pinned by jax-mps's dependency scripts.
- If Metal reports no device, run in a session with access to the GPU. A
  restrictive sandbox may allow device discovery but block actual execution.

For eager `palladium.metal_call` or the CPU-FFI `metal_call_jit` path, the
modified jax-mps plugin is unnecessary. Those paths use `metal-runtime`
directly; see [Getting started](docs/getting-started.md).
