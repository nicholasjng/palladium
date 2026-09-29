# Getting started

Palladium accepts a function written for jax.experimental.pallas, traces its
single pallas_call, and emits a Metal kernel. Choose an entry point based on
where the call should run:

| Entry point | Execution |
|---|---|
| pl.pallas_call on mps | palladium is the registered Pallas backend for jax-mps; composable with jax.jit |
| metal_call | Eager Metal dispatch; NumPy inputs and outputs |
| metal_call_jit | Metal dispatch through a CPU jax.ffi target; composable with jax.jit |

The latter two use metal-runtime. The mps path requires jax-mps to be
installed and selected. See the [README](../README.md) for the current install
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
result = call(x, y)
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

Cooperative kernels use multiple threads in a threadgroup; the interpreter
models only one thread per instance and is not a valid reference for them.
Supply an independent reference= to .verify for those kernels. FAST math
is the default, so transcendental results and reduction order can differ from
the reference.

.explain(*args) reports the grid, threadgroup, declared storage, emitted MSL
size, and expected execution path. It traces and emits source but does not
compile or dispatch. palladium.debug_msl returns the generated MSL directly.

## Plain Pallas on jax-mps

Importing palladium registers it as the Pallas backend for the mps platform:
a plain `pl.pallas_call` inside `jax.jit` on a jax-mps device lowers to the
Metal kernel `metal_call_jit` would build, with no wrapper call. Metal-side
options travel as `compiler_params`:

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
backward implementation: `palladium.with_vjp(forward, backward)`,
`with_auxiliary_vjp` for checkpointed adjoints, or `with_reference_vjp` to
borrow JAX's pullback of a pure-JAX reference.

## Iterating on the device

A kernel that advances a state needs one dispatch per step, because a
threadgroup barrier cannot synchronize the whole grid. `call.iterate(*arrays,
steps=n)` encodes all `n` dispatches into one command buffer and feeds each
step's outputs back into the next step's inputs without leaving the GPU
(output j into input j by default, or explicit `feedback=[(output, input), ...]`
pairs; inputs never fed back stay fixed). It returns the last step's outputs.

## JAX transformations

metal_call_jit supports jax.jit and jax.vmap. Its default
vmap_method="pipelined" handles the batch in one FFI call; nested batch
levels and the sequential methods dispatch one element at a time. Put a
batch axis in the Pallas grid when possible.

On mps, jax.vmap over a pallas_call is JAX's own batching. No path derives
gradients from emitted MSL. Pair
forward and backward calls with jax.custom_vjp, or provide a pure-JAX
reference VJP where supported. See the [supported functionality](supported-jax.md)
for details.

Unsupported kernel structure raises TraceError; unsupported lowering raises
EmitError or UnsupportedPrimitiveError. Invalid runtime arguments raise
DispatchError. These errors derive from PalladiumError.
