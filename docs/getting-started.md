# Getting started

Palladium accepts a function written for jax.experimental.pallas, traces its
single pallas_call, and emits a Metal kernel. Choose an entry point based on
where the call should run:

| Entry point | Execution |
|---|---|
| mps_call_jit | jax-mps custom call on the MPS platform; composable with jax.jit |
| metal_call | Eager Metal dispatch; NumPy inputs and outputs |
| metal_call_jit | Metal dispatch through a CPU jax.ffi target; composable with jax.jit |

The latter two use metal-runtime. mps_call_jit requires jax-mps to be
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

## JAX transformations

metal_call_jit supports jax.jit and jax.vmap. Its default
vmap_method="pipelined" handles the batch in one FFI call; nested batch
levels and the sequential methods dispatch one element at a time. Put a
batch axis in the Pallas grid when possible.

mps_call_jit supports jax.jit; jax.vmap over its custom call is not
supported. Neither custom-call path derives gradients from emitted MSL. Pair
forward and backward calls with jax.custom_vjp, or provide a pure-JAX
reference VJP where supported. See the [supported functionality](supported-jax.md)
for details.

Unsupported kernel structure raises TraceError; unsupported lowering raises
EmitError or UnsupportedPrimitiveError. Invalid runtime arguments raise
DispatchError. These errors derive from PalladiumError.
