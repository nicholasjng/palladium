"""The conveniences layered over the core pipeline: storage accounting in
`explain`, the device thread limit, structured error fields, and the
bounded per-shape cache.
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


def test_explain_reports_per_thread_stack():
    """The per-thread stack has no published ceiling, so explain reports the estimate before compiling."""
    d = _reused_call(64).explain(jax.ShapeDtypeStruct((64,), jnp.float32))
    # 64 f32 elements per live array, several arrays.
    assert d.thread_bytes >= 64 * 4
    assert d.threadgroup_bytes == 0
    assert "stack~" in str(d)


def test_stack_estimate_scales_with_block_size():
    small = _reused_call(64).explain(jax.ShapeDtypeStruct((64,), jnp.float32))
    big = _reused_call(512).explain(jax.ShapeDtypeStruct((512,), jnp.float32))
    assert big.thread_bytes == small.thread_bytes * 8


def test_threadgroup_over_device_thread_limit_is_rejected():
    limit = device_limits()["max_threads_per_threadgroup"]
    f = _tanh_call(compiler_params=palladium.CompilerParams(threadgroup=limit * 2))
    with pytest.raises(EmitError, match="max_threads_per_threadgroup"):
        f(np.zeros(64, dtype=np.float32))


# --- structured errors ---------------------------------------------------


def test_unsupported_primitive_names_the_primitive_as_a_field():
    """The primitive name is a field a caller can branch on instead of matching message text."""

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


# --- bounded caches ------------------------------------------------------


def _sum_kernel(x_ref, o_ref):
    o_ref[...] = jnp.sum(x_ref[...], keepdims=True)


def _key(n):
    """The cache key for one float32 input of length n."""
    return (((n,), np.dtype(np.float32).str),)


def test_cache_is_bounded(rng, monkeypatch):
    """Under JAX's trace caching a hit need not re-enter Python, so the
    bound is on entries, not on recency."""
    monkeypatch.setattr(palladium.ffi, "_CACHE_SIZE", 2)
    # out_shape is fixed while the input length is free, so every n is a new key.
    f = palladium.metal_call(_sum_kernel, out_shape=jax.ShapeDtypeStruct((1,), jnp.float32))
    for n in (8, 16, 24):
        f(rng.standard_normal(n, dtype=np.float32))
    assert list(f._cache) == [_key(16), _key(24)]
    f(rng.standard_normal(32, dtype=np.float32))
    assert list(f._cache) == [_key(24), _key(32)]
