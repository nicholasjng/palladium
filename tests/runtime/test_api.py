"""The conveniences layered over the core pipeline: storage accounting in
`explain`, the device thread limit, and structured error fields.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import palladium
from palladium.device import device_limits
from palladium.errors import (
    EmitError,
    UnsupportedPrimitiveError,
)


def _tanh_kernel(x_ref, o_ref):
    o_ref[...] = jnp.tanh(x_ref[...]) * 2.0


def _tanh_call(**kwargs):
    return palladium.metal_call(
        _tanh_kernel, out_shape=jax.ShapeDtypeStruct((64,), jnp.float32), **kwargs
    )


def _reused_kernel(x_ref, o_ref):
    t = jnp.tanh(x_ref[...])  # two consumers, so it stays a thread-local array
    o_ref[...] = t * t + t


def _reused_call(n):
    return palladium.metal_call(_reused_kernel, out_shape=jax.ShapeDtypeStruct((n,), jnp.float32))


def test_explain_reports_per_thread_storage():
    """Check that explain reports declared per-thread storage, scaling with the block size."""
    small = _reused_call(64).explain(jax.ShapeDtypeStruct((64,), jnp.float32))
    big = _reused_call(512).explain(jax.ShapeDtypeStruct((512,), jnp.float32))
    # 64 f32 elements per live array, several arrays.
    assert small.thread_bytes >= 64 * 4
    assert small.threadgroup_bytes == 0
    assert big.thread_bytes == small.thread_bytes * 8


def test_threadgroup_over_device_thread_limit_is_rejected():
    """Verify that an explicit group exceeding the device thread limit is rejected before
    compilation."""
    limit = device_limits()["max_threads_per_threadgroup"]
    f = _tanh_call(compiler_params=palladium.CompilerParams(threadgroup=limit * 2))
    with pytest.raises(EmitError, match="max_threads_per_threadgroup"):
        f(np.zeros(64, dtype=np.float32))


# --- structured errors ---------------------------------------------------


def test_unsupported_primitive_names_the_primitive_as_a_field():
    """Check that a missing primitive raises UnsupportedPrimitiveError with a machine-readable
    primitive name."""

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sort(x_ref[...])

    with pytest.raises(UnsupportedPrimitiveError) as excinfo:
        palladium.debug_msl(
            kernel,
            jax.ShapeDtypeStruct((8,), jnp.float32),
            out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        )
    assert excinfo.value.primitive == "sort"
    assert excinfo.value.primitive in str(excinfo.value)
