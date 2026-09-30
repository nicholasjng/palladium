"""Plain pl.pallas_call lowered for the mps platform goes through Palladium.

These lower for ``mps`` without the plugin (JAX allows lowering for an
absent platform) and inspect the StableHLO; execution is covered by the
device tests in test_mps.py."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from metal_runtime import MathMode

import palladium


def _add(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def _lower_for(platforms, fn, *args):
    return jax.jit(fn).trace(*args).lower(lowering_platforms=platforms).as_text()


def _backend_config(text: str) -> dict:
    """The dispatch descriptor, decoded from MLIR's string escaping:
    `\\XX` is a hex byte and `\\\\` a backslash."""
    start = text.find("@palladium.dispatch(")
    assert start >= 0, "no palladium.dispatch custom call in the module"
    start = text.index('backend_config = "', start) + len('backend_config = "')
    out, i = [], start
    while text[i] != '"':
        if text[i] == "\\":
            if text[i + 1] == "\\":
                out.append("\\")
                i += 2
            else:
                out.append(chr(int(text[i + 1 : i + 3], 16)))
                i += 3
        else:
            out.append(text[i])
            i += 1
    return json.loads("".join(out))


def test_plain_pallas_call_lowers_to_the_dispatch_custom_call_on_mps():
    call = pl.pallas_call(_add, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))
    x = jnp.ones(8, jnp.float32)
    text = _lower_for(("mps",), lambda a, b: call(a, b) * 2.0, x, x)
    assert "stablehlo.custom_call @palladium.dispatch" in text
    config = _backend_config(text)
    assert config["version"] == 2
    assert "kernel void" not in config["body"]
    assert "arg0_base" in config["prologue"]
    assert config["grid"] == [1, 1, 1]


def test_compiler_params_select_tensorops_and_scale_the_launch():
    def dot(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...])

    call = pl.pallas_call(
        dot,
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((16, 16), lambda i, j: (i, 0)),
            pl.BlockSpec((16, 32), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((16, 32), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((32, 64), jnp.float32),
        compiler_params=palladium.CompilerParams(dot_general="tensorops"),
    )
    a = jnp.ones((32, 16), jnp.float32)
    b = jnp.ones((16, 64), jnp.float32)
    config = _backend_config(_lower_for(("mps",), call, a, b))
    width = palladium.launch.simdgroup_width()
    assert "matmul2d" in config["body"]
    assert config["threadgroup"] == [4 * width, 1, 1]
    assert config["grid"] == [2 * 4 * width, 2, 1]


def test_other_platforms_keep_jax_own_lowering():
    call = pl.pallas_call(_add, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))
    x = jnp.ones(8, jnp.float32)
    with pytest.raises(ValueError, match="interpret mode"):
        _lower_for(("cpu",), call, x, x)
    interpreted = pl.pallas_call(
        _add, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32), interpret=True
    )
    np.testing.assert_array_equal(np.asarray(jax.jit(interpreted)(x, x)), 2.0 * np.ones(8))
    assert "palladium.dispatch" not in _lower_for(("cpu",), interpreted, x, x)


def test_mps_lowering_rejects_non_fast_math_and_aliases():
    call = pl.pallas_call(
        _add,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        compiler_params=palladium.CompilerParams(math_mode=MathMode.SAFE),
    )
    x = jnp.ones(8, jnp.float32)
    with pytest.raises(ValueError, match="FAST only"):
        _lower_for(("mps",), call, x, x)
    aliased = pl.pallas_call(
        _add,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        input_output_aliases={0: 0},
    )
    with pytest.raises(ValueError, match="input_output_aliases"):
        _lower_for(("mps",), aliased, x, x)
    with pytest.raises(ValueError, match="dot_general"):
        palladium.CompilerParams(dot_general="simdgroup")
