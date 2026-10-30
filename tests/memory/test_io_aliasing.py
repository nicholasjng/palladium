"""input_output_aliases: in-place kernels, one buffer behind the aliased
input/output pair."""

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
import pytest

import palladium
from palladium.errors import TraceError

F32 = jnp.float32


def _double_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0


def test_aliased_call_matches_interpret(rng):
    """Compare an aliased output with the interpreter and verify that the caller's NumPy input
    stays unchanged."""
    f = palladium.metal_call(
        _double_kernel,
        out_shape=jax.ShapeDtypeStruct((64,), F32),
        input_output_aliases={0: 0},
    )
    x = rng.standard_normal(64).astype(np.float32)
    before = x.copy()
    got = f(x)
    np.testing.assert_allclose(got, np.asarray(f.interpret(before)), rtol=1e-6)
    # The caller's array is copied at upload, so it must stay untouched.
    np.testing.assert_array_equal(x, before)


def test_aliased_kernel_grid_and_second_input(rng):
    """Compare a blocked two-input kernel with the interpreter when one input aliases the output."""

    def kernel(x_ref, w_ref, o_ref):
        o_ref[...] = x_ref[...] * w_ref[...]

    f = palladium.metal_call(
        kernel,
        grid=(4,),
        in_specs=[
            pl.BlockSpec((16,), lambda i: i),
            pl.BlockSpec((16,), lambda i: i),
        ],
        out_specs=pl.BlockSpec((16,), lambda i: i),
        out_shape=jax.ShapeDtypeStruct((64,), F32),
        input_output_aliases={0: 0},
    )
    x = rng.standard_normal(64).astype(np.float32)
    w = rng.standard_normal(64).astype(np.float32)
    np.testing.assert_allclose(f(x, w), np.asarray(f.interpret(x, w)), rtol=1e-6)


def test_ffi_aliased_vmap(rng):
    """Check that vmapped aliased FFI calls double each batch element correctly."""
    f = palladium.metal_call(
        _double_kernel,
        out_shape=jax.ShapeDtypeStruct((16,), F32),
        input_output_aliases={0: 0},
    )
    xs = rng.standard_normal((11, 16)).astype(np.float32)
    got = np.asarray(jax.vmap(f)(xs))
    np.testing.assert_allclose(got, xs * 2.0, rtol=1e-6)


def test_read_after_aliased_write_is_rejected(rng):
    """Verify that reading an input after an aliased output write raises TraceError."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0
        o_ref[0] = x_ref[1]  # reads the input after the aliased write

    f = palladium.metal_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8,), F32),
        input_output_aliases={0: 0},
    )
    with pytest.raises(TraceError, match="read after"):
        f(rng.standard_normal(8).astype(np.float32))


def test_alias_pair_with_different_blocks_is_rejected():
    """Verify that an alias whose input and output blocks map to different programs is rejected:
    sharing one buffer would let a program overwrite another's input."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    call = pl.pallas_call(
        kernel,
        grid=(2,),
        in_specs=[pl.BlockSpec((8,), lambda i: (i,))],
        out_specs=pl.BlockSpec((8,), lambda i: (1 - i,)),
        out_shape=jax.ShapeDtypeStruct((16,), F32),
        input_output_aliases={0: 0},
    )
    with pytest.raises(TraceError, match="identical array shape"):
        palladium.trace(call, jax.ShapeDtypeStruct((16,), F32))
