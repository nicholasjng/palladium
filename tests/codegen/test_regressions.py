"""Emitter regressions that remain testable without a Metal device."""

import re

import jax
import jax.numpy as jnp
import metal_runtime as mr
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium


def _case(kind):
    dtype = jnp.float32
    input_shape = output_shape = (2,)
    kwargs = {}
    if kind == "bfloat_literal":
        dtype = jnp.bfloat16

        def kernel(x, o):
            o[...] = x[...] + jnp.bfloat16(0.5)

    elif kind == "scalar_to_array":
        input_shape = output_shape = (1,)

        def kernel(x, o):
            o[...] = jnp.reshape(x[0], (1,))

    elif kind == "array_to_scalar":
        input_shape, output_shape = (1,), ()

        def kernel(x, o):
            o[...] = jnp.reshape(x[...], ())

    elif kind == "zero_power":

        def kernel(x, o):
            o[...] = jax.lax.integer_pow(x[...], 0)

    elif kind == "swap":
        kwargs["scratch_shapes"] = [pl.MemorySpace.ANY((2,), dtype)]

        def kernel(x, o, scratch):
            scratch[...] = x[...]
            old = jax.ref.swap(scratch, Ellipsis, x[...] + 1)
            scratch[...] = x[...] + 2
            o[...] = old

    else:
        raise AssertionError(kind)
    kwargs["out_shape"] = jax.ShapeDtypeStruct(output_shape, dtype)
    arg = jax.ShapeDtypeStruct(input_shape, dtype)
    return kernel, arg, kwargs


def _source(kind):
    kernel, arg, kwargs = _case(kind)
    return palladium.debug_msl(kernel, arg, **kwargs)


def test_fractional_bfloat_literal():
    assert " + bfloat(0.5f);" in _source("bfloat_literal")


@pytest.mark.parametrize("kind", ["scalar_to_array", "array_to_scalar"])
def test_reshape_between_scalar_and_one_element_array_stays_in_registers(kind):
    """Rank-0 values and one-element arrays are both plain scalar variables;
    neither side of the reshape may index the other."""
    source = _source(kind)
    scalars = re.findall(r"^    float (t\d+);$", source, re.MULTILINE)
    assert scalars
    assert not re.search(r"^    float t\d+\[1\];$", source, re.MULTILINE)
    for scalar in scalars:
        assert not re.search(rf"\b{scalar}\[", source)


def test_swap_snapshots_before_store_and_later_mutation():
    source = _source("swap")
    snapshot = re.search(r"(t\d+)\[_i\d+\] = scratch0\[_i\d+\];", source)
    assert snapshot is not None
    assert re.search(rf"arg1\[_i\d+\] = {snapshot[1]}\[_i\d+\];", source)
    # Initialization precedes the snapshot; both subsequent writes follow it.
    writes = list(re.finditer(r"scratch0\[_i\d+\] =", source))
    assert len(writes) == 3
    assert writes[0].start() < snapshot.start() < writes[1].start()


@pytest.mark.parametrize(
    "kind",
    ["bfloat_literal", "scalar_to_array", "array_to_scalar", "zero_power", "swap"],
)
def test_regressions_against_interpret(kind):
    try:
        mr.device_name()
    except mr.DeviceError as exc:
        pytest.skip(str(exc))
    kernel, arg, kwargs = _case(kind)
    call = palladium.metal_call(
        kernel, **kwargs, compiler_params=palladium.CompilerParams(math_mode=mr.MathMode.SAFE)
    )
    x = np.arange(np.prod(arg.shape), dtype=np.float32).reshape(arg.shape)
    x = x.astype(arg.dtype)
    np.testing.assert_array_equal(
        np.asarray(call(x), dtype=np.float32),
        np.asarray(call.interpret(x), dtype=np.float32),
    )


def test_a_jitted_helper_fuses_on_every_call():
    # jit bodies are cached and shared by every call, Vars included, so the
    # second walk once saw the first walk's consumers and fused nothing.
    @jax.jit
    def act(v):
        return jnp.tanh(v * 2.0 + 1.0) * 3.0 - v

    def kernel(x_ref, o_ref):
        o_ref[...] = act(act(x_ref[...]))

    shape = jax.ShapeDtypeStruct((64,), jnp.float32)
    once = palladium.debug_msl(
        lambda x_ref, o_ref: o_ref.__setitem__(..., act(x_ref[...])), shape, out_shape=shape
    )
    twice = palladium.debug_msl(kernel, shape, out_shape=shape)
    assert twice.count("float t") <= once.count("float t") + 1


def test_scalar_fill_of_where_is_not_materialized():
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.where(x_ref[...] > 0.0, x_ref[...], 0.0)

    shape = jax.ShapeDtypeStruct((64,), jnp.float32)
    msl = palladium.debug_msl(kernel, shape, out_shape=shape)
    # A materialized broadcast is the only loop named _b.
    assert "for (uint _b" not in msl


def test_sum_of_squares_needs_no_block_array():
    def kernel(x_ref, o_ref):
        o_ref[0] = jnp.sum(x_ref[...] * x_ref[...])

    msl = palladium.debug_msl(
        kernel,
        jax.ShapeDtypeStruct((256,), jnp.float32),
        out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
    )
    assert "[256]" not in msl
