# Getting started

palladium runs [Pallas](https://docs.jax.dev/en/latest/pallas/index.html)
kernels on Apple GPUs. You write an ordinary Pallas kernel; palladium
traces it, emits Metal Shading Language, compiles it through
[metal-runtime](https://github.com/nicholasjng/metal-runtime), and gives
you back a callable.

## Install

Requires macOS on Apple silicon, Python 3.12+, and CMake + Ninja on
PATH (`brew install cmake ninja`).

```sh
uv sync          # builds the native FFI handler as part of the install
uv run pytest -q # sanity check: must be green on a Metal-capable Mac
```

## First kernel

```python
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
import palladium

def saxpy(x_ref, y_ref, o_ref):
    o_ref[...] = 2.0 * x_ref[...] + y_ref[...]

spec = pl.BlockSpec((256,), lambda i: (i,))
call = palladium.metal_call(
    saxpy,
    grid=(16,),            # 16 program instances...
    in_specs=[spec, spec],  # ...each seeing one 256-element block
    out_specs=spec,
    out_shape=jax.ShapeDtypeStruct((4096,), jnp.float32),
)

x = np.arange(4096, dtype=np.float32)
y = np.ones(4096, dtype=np.float32)
out = call(x, y)   # NumPy in, NumPy out; compiled on first call
```

`metal_call` takes the usual `pl.pallas_call` keywords (`out_shape`,
`grid`, `in_specs`, `out_specs`, ...) plus two Metal extras:
`math_mode` (`metal_runtime.MathMode`, FAST by default) and
`threadgroup` (explicit threadgroup size; leave unset normally).

The grid matters more than on other backends: each program instance's
blocks are copied into thread-local memory, so per-instance blocks must
stay small (a gridless call over the full 4096-element array would
overflow the per-thread stack, and says so). Parallelism comes from the
grid, not from threads inside an instance.

## The oracle workflow

Every `metal_call` result carries `.interpret`, the same kernel run by
Pallas's CPU interpreter. Diff against it while developing; it is the
ground truth for what the kernel should compute:

```python
np.testing.assert_allclose(call(a, x, y), call.interpret(a, x, y), rtol=1e-5)
```

FAST math means transcendentals and reduction orders are not bit-equal
to the oracle; expect roughly 1e-6 relative deviation for f32
elementwise work and up to 1e-4 through exp/log-heavy kernels. Pass
`math_mode=metal_runtime.MathMode.SAFE` for IEEE ordering.

## Composing with jax.jit

`metal_call` is eager and NumPy-based; its result cannot sit inside a
jitted computation. `metal_call_jit` registers the kernel as a jax.ffi
target instead, so the call is traceable and composes with surrounding
`jax.numpy` code:

```python
attn = palladium.metal_call_jit(kernel, out_shape=...)
fast = jax.jit(lambda q, k, v: jnp.tanh(attn(q, k, v)))
```

Two transformation notes:

- **`jax.grad`**: not differentiable by itself (palladium cannot derive
  a backward kernel from forward MSL). Author the backward pass as a
  second Pallas kernel and pair the two with `jax.custom_vjp`;
  `examples/07_custom_vjp.py` is the worked recipe, gradient-checked
  against `jax.grad` of the plain jnp expression.
- **`jax.vmap`**: pass `vmap_method="sequential"` (or
  `"sequential_unrolled"`) to `metal_call_jit`. Each batch element is a
  separate dispatch paying the fixed dispatch floor, so this is the
  convenience path; putting the batch dimension in the Pallas grid is
  the fast one. Whole-batch methods are rejected (the launch grid is
  baked per unbatched shape).

## Seeing what you got

- `call.explain(*args)` reports which execution model the kernel gets
  (`cooperative`, one 32-thread SIMD-group per program instance, or
  `classic`, one thread per instance), the launch geometry, and, on the
  classic model, the first reason the cooperative one was rejected.
  Takes arrays or `jax.ShapeDtypeStruct`s; compiles and runs nothing.
- `palladium.debug_msl(kernel, *example_args, **pallas_kwargs)` returns
  the emitted MSL text.
- `PALLADIUM_EXPLAIN=1` prints one stderr line per newly compiled
  kernel; `PALLADIUM_DUMP_MSL=1` prints each kernel's source (a
  directory path writes `.metal` files instead).

## When something is rejected

Everything palladium raises derives from `palladium.PalladiumError`:

- `TraceError`: the pallas_call itself is out of scope (multiple
  pallas_calls, scalar prefetch grids).
- `EmitError`: the kernel uses an unsupported case of a supported
  primitive (non-contiguous blocks, batched dots, strided slices);
  the message names the construct.
- `UnsupportedPrimitiveError`: no lowering rule exists for a staged
  primitive; the message names it, and `docs/extending.md` shows how to
  add one.
- `DispatchError`: a compiled kernel was called with the wrong argument
  count, shape, or dtype. Dtypes are checked strictly and never cast.

See `docs/supported-subset.md` for the full contract.
