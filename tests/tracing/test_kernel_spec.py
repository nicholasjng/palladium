"""Tracing Pallas into KernelSpecs."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium
from palladium import trace


def _mad_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0 + y_ref[...]


def test_gridless_spec(rng):
    """Check that a gridless call normalizes to one program with whole-array blocks."""
    f = pl.pallas_call(_mad_kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32))
    x = rng.standard_normal((8, 128), dtype=np.float32)
    spec = trace(f, x, x)

    assert spec.grid == (1,)
    assert [i.block_shape for i in spec.inputs] == [(8, 128), (8, 128)]
    assert spec.outputs[0].array_shape == (8, 128)
    assert spec.outputs[0].dtype == np.float32


def test_gridded_spec(rng):
    """Check the grid and block layout of a blocked call."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 1.1

    f = pl.pallas_call(
        kernel,
        grid=(16,),
        in_specs=[pl.BlockSpec((8,), lambda i: (i,))],
        out_specs=pl.BlockSpec((8,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
    )
    x = rng.standard_normal(128, dtype=np.float32)
    spec = trace(f, x)

    assert spec.grid == (16,)
    assert spec.inputs[0].block_shape == (8,)
    assert spec.inputs[0].array_shape == (128,)


def test_trace_rejects_multiple_pallas_calls(rng):
    """Verify that tracing a wrapper with two Pallas calls reports the call count and rejects it."""
    f = pl.pallas_call(_mad_kernel, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32))

    def two_calls(x, y):
        return f(x, y) + f(y, x)

    x = rng.standard_normal(4, dtype=np.float32)
    with pytest.raises(ValueError, match="2 pallas_call"):
        trace(two_calls, x, x)


def test_non_blocked_dims_rejected():
    """Verify that an Element block dimension is rejected despite exposing a block_size attribute."""
    from palladium.errors import TraceError

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    call = pl.pallas_call(
        kernel,
        grid=(2,),
        in_specs=[pl.BlockSpec((pl.Element(8),), lambda i: (i,))],
        out_specs=pl.BlockSpec((8,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((16,), jnp.float32),
    )
    with pytest.raises(TraceError, match="Element"):
        trace(call, jax.ShapeDtypeStruct((16,), jnp.float32))


def test_trace_looks_through_a_jit_wrapper(rng):
    """Check that tracing a JIT-wrapped Pallas call preserves direct tracing's grid and operand
    layouts."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    call = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))
    x = jax.ShapeDtypeStruct((8,), jnp.float32)
    direct = trace(call, x)
    wrapped = trace(jax.jit(call), x)
    assert wrapped.grid == direct.grid
    assert wrapped.inputs == direct.inputs
    assert wrapped.outputs == direct.outputs


def test_compiler_params_validate_and_normalize_their_options():
    """Options fail when constructed, not at the first dispatch: math modes convert from
    their values, and threadgroup dimensions are integers other than bool."""
    safe: Any = "safe"  # typed as MathMode; its value converts
    assert palladium.CompilerParams(math_mode=safe).math_mode is palladium.MathMode.SAFE
    bogus: Any = "bogus"
    with pytest.raises(ValueError, match="bogus"):
        palladium.CompilerParams(math_mode=bogus)
    assert palladium.CompilerParams(threadgroup=np.int64(4)).threadgroup_dims == (4,)
    assert palladium.CompilerParams(threadgroup=[np.int32(4), 2]).threadgroup_dims == (4, 2)
    invalid: tuple[Any, ...] = (True, (4, False), 2.0, "4", (0,), (1, 1, 1, 1))
    for bad in invalid:
        with pytest.raises(ValueError, match="threadgroup"):
            palladium.CompilerParams(threadgroup=bad)


@pytest.mark.parametrize("name", ["my kernel-1", "1x", "é"])
def test_kernel_name_must_be_an_identifier(name):
    """The name becomes the Metal function's, so tracing rejects one Metal cannot compile."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    x = jax.ShapeDtypeStruct((8,), jnp.float32)
    with pytest.raises(palladium.TraceError, match="not a valid identifier"):
        palladium.trace(pl.pallas_call(kernel, out_shape=x, name=name), x)
    assert palladium.trace(pl.pallas_call(kernel, out_shape=x, name="scale_2"), x).name == "scale_2"
