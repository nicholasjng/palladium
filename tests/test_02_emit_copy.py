"""Exercise 1: `_rule_get` and `_rule_swap` in emit.py.

The smallest kernel that exists: copy in, copy out. Green means your
emitted MSL compiles, dispatches, and the block load/store loops are right.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import palladium

pytestmark = pytest.mark.exercise


def _copy_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]


def test_emitted_source_is_plausible_msl(rng):
    from jax.experimental import pallas as pl

    f = pl.pallas_call(_copy_kernel, out_shape=jax.ShapeDtypeStruct((64,), jnp.float32))
    spec = palladium.trace(f, rng.standard_normal(64, dtype=np.float32))
    msl = palladium.emit_msl(spec)
    assert "kernel void" in msl
    assert "[[buffer(0)]]" in msl and "[[buffer(1)]]" in msl
    # It must actually compile: Metal's frontend is the real judge.
    import metal_runtime as mr

    mr.Kernel(msl, spec.name)


def test_copy_roundtrip_1d(rng):
    f = palladium.metal_call(
        _copy_kernel, out_shape=jax.ShapeDtypeStruct((256,), jnp.float32)
    )
    x = rng.standard_normal(256, dtype=np.float32)
    np.testing.assert_array_equal(f(x), x)


def test_copy_roundtrip_2d(rng):
    f = palladium.metal_call(
        _copy_kernel, out_shape=jax.ShapeDtypeStruct((16, 32), jnp.float32)
    )
    x = rng.standard_normal((16, 32), dtype=np.float32)
    np.testing.assert_array_equal(f(x), x)


def test_copy_matches_interpret_oracle(rng):
    f = palladium.metal_call(
        _copy_kernel, out_shape=jax.ShapeDtypeStruct((128,), jnp.float32)
    )
    x = rng.standard_normal(128, dtype=np.float32)
    np.testing.assert_array_equal(f(x), np.asarray(f.interpret(x)))


def _sum_and_diff_kernel(x_ref, y_ref, sum_ref, diff_ref):
    sum_ref[...] = x_ref[...] + y_ref[...]
    diff_ref[...] = x_ref[...] - y_ref[...]


def test_multi_output_kernel(rng):
    """Two output refs: `_rule_swap` is generic over which ref it targets,
    and dispatch.py's BoundKernel.launch already loops over spec.outputs,
    so this needs no new emitter or dispatch code, only a kernel that
    exercises it."""
    f = palladium.metal_call(
        _sum_and_diff_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((32,), jnp.float32),
            jax.ShapeDtypeStruct((32,), jnp.float32),
        ),
    )
    x = rng.standard_normal(32, dtype=np.float32)
    y = rng.standard_normal(32, dtype=np.float32)
    s, d = f(x, y)
    np.testing.assert_array_equal(s, x + y)
    np.testing.assert_array_equal(d, x - y)
    s_ref, d_ref = f.interpret(x, y)
    np.testing.assert_array_equal(s, np.asarray(s_ref))
    np.testing.assert_array_equal(d, np.asarray(d_ref))
