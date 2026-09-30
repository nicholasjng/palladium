# Getting started

Palladium accepts a function written for jax.experimental.pallas, traces its
single pallas_call, and emits a Metal kernel. Two entry points run it:

| Entry point | Execution |
|---|---|
| pl.pallas_call on mps | palladium is the registered Pallas backend for jax-mps; composable with jax.jit |
| metal_call | Metal dispatch through a CPU jax.ffi target backed by metal-runtime; composable with jax.jit and jax.vmap, no plugin |

Both accept the same kernels. See the [README](../README.md) for the install
requirements.

## First kernel

~~~python
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
import palladium

def saxpy(x_ref, y_ref, out_ref):
    out_ref[...] = 2.0 * x_ref[...] + y_ref[...]

n = 4096
block = pl.BlockSpec((1,), lambda i: (i,))
call = palladium.metal_call(
    saxpy,
    grid=(n,),
    in_specs=(block, block),
    out_specs=block,
    out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
)
x = np.arange(n, dtype=np.float32)
y = np.ones(n, dtype=np.float32)
result = jax.jit(call)(x, y)
~~~

Each program instance handles one element here. Keep blocks small: Pallas
blocks and intermediate arrays are stored per thread, and oversized kernels
can exceed Metal's per-thread stack limit.

## Check the result

Every call exposes .interpret, which runs the same Pallas kernel through the
CPU interpreter. Use it as the reference for independent-thread kernels:

~~~python
np.testing.assert_allclose(call(x, y), call.interpret(x, y), rtol=1e-5)
~~~

FAST math is the default, so transcendental results and reduction order can differ from
the reference.

.explain(*args) reports the grid, threadgroup, declared storage, and emitted
MSL size. It traces and emits source but does not compile or dispatch.
palladium.debug_msl returns the generated MSL directly.

## Plain Pallas on jax-mps

Importing palladium registers it as the Pallas backend for the mps platform:
a plain `pl.pallas_call` inside `jax.jit` on a jax-mps device lowers to the
same Metal kernel `metal_call` would build, with no wrapper call. Metal-side
options travel as `compiler_params` on both paths:

~~~python
call = pl.pallas_call(
    kernel,
    out_shape=...,
    compiler_params=palladium.CompilerParams(dot_general="tensorops", threadgroup=128),
)
jax.jit(call)(x)  # palladium.dispatch on mps; JAX's own lowering elsewhere
~~~

Other platforms keep JAX's behavior (interpret=True, or an error), and
`interpret=True` is honored everywhere. Gradients pair a forward call with a
backward implementation: `palladium.with_vjp(forward, backward)`. Pass
`residuals=k` to keep the forward call's last `k` outputs (checkpoints, for
example) for the backward pass instead of returning them.

## JAX transformations

metal_call supports jax.jit and jax.vmap. A vmapped call handles the batch
in one FFI call; nested batch levels dispatch one element at a time. Put a
batch axis in the Pallas grid when possible. A kernel that advances a state
over many dispatches is an ordinary `lax.fori_loop` or `lax.scan` around the
call under `jax.jit`.

On mps, jax.vmap over a pallas_call is JAX's own batching. No path derives
gradients from emitted MSL. Pair forward and backward calls with
jax.custom_vjp, or provide a pure-JAX reference VJP where supported. See the
[supported functionality](supported-jax.md) for details.

Unsupported kernel structure raises TraceError; unsupported lowering raises
EmitError or UnsupportedPrimitiveError. Unsupported input dtypes raise
DispatchError. These errors derive from PalladiumError.
