"""The conveniences layered over the core pipeline: storage accounting in
`explain`, device-derived threadgroup sizing, structured
error fields, and the bounded per-shape cache.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium
from palladium.diagnostics import device_limits, normalize_threadgroup, simdgroup_width
from palladium.errors import (
    EmitError,
    UnsupportedPrimitiveError,
)
from palladium.threadgroup import (
    barrier,
    thread_index,
    threadgroup_memory,
    threads_per_threadgroup,
)

TG = 32


def _tanh_kernel(x_ref, o_ref):
    o_ref[...] = jnp.tanh(x_ref[...]) * 2.0


def _tanh_call(**kwargs):
    return palladium.metal_call(
        _tanh_kernel, out_shape=jax.ShapeDtypeStruct((64,), jnp.float32), **kwargs
    )


def _block_sum_kernel(x_ref, o_ref, s_ref):
    t = thread_index()
    s_ref[t] = x_ref[0]
    barrier()
    n = threads_per_threadgroup()
    o_ref[0] = jax.lax.fori_loop(0, n, lambda i, a: a + s_ref[i], jnp.float32(0.0))


def _block_sum_call(n, threadgroup=TG, extent=TG):
    return palladium.metal_call(
        _block_sum_kernel,
        grid=(n,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
        scratch_shapes=[threadgroup_memory((extent,), jnp.float32)],
        threadgroup=threadgroup,
    )


# --- cooperative execution ------------------------------------------


def test_cooperative_kernel_matches_an_explicit_reference():
    n = 100
    f = _block_sum_call(n)
    x = np.arange(n, dtype=np.float32)

    def reference(a):
        out = np.empty_like(a)
        for start in range(0, len(a), TG):
            out[start : start + TG] = a[start : start + TG].sum()
        return out

    np.testing.assert_allclose(f(x), reference(x))


# --- explain / accounting ------------------------------------------------


def test_explain_reports_per_thread_stack():
    """The per-thread stack has no published ceiling, so explain reports the estimate before compiling."""
    f = _tanh_call()
    d = f.explain(jax.ShapeDtypeStruct((64,), jnp.float32))
    # 64 f32 elements per live array, several arrays.
    assert d.thread_bytes >= 64 * 4
    assert d.threadgroup_bytes == 0
    assert "stack~" in str(d)


def test_stack_estimate_scales_with_block_size():
    small = _tanh_call().explain(jax.ShapeDtypeStruct((64,), jnp.float32))
    big = palladium.metal_call(
        _tanh_kernel, out_shape=jax.ShapeDtypeStruct((512,), jnp.float32)
    ).explain(jax.ShapeDtypeStruct((512,), jnp.float32))
    assert big.thread_bytes == small.thread_bytes * 8


def test_explain_reports_threadgroup_memory_against_the_device_budget():
    d = _block_sum_call(64).explain(jax.ShapeDtypeStruct((64,), jnp.float32))
    assert d.threadgroup_bytes == TG * 4
    assert d.threadgroup_limit == device_limits()["max_threadgroup_memory_length"]
    assert "shared=" in str(d)


# --- threadgroup sizing --------------------------------------------------


def test_simdgroup_sentinel_resolves_to_the_simd_width():
    assert normalize_threadgroup("simdgroup") == (simdgroup_width(),)
    assert simdgroup_width() == 32  # every Apple GPU family to date


def test_simdgroup_sentinel_runs_a_cooperative_kernel():
    n = 96
    f = _block_sum_call(n, threadgroup="simdgroup")
    x = np.arange(n, dtype=np.float32)
    expected = np.empty_like(x)
    for start in range(0, n, simdgroup_width()):
        expected[start : start + simdgroup_width()] = x[start : start + simdgroup_width()].sum()
    np.testing.assert_allclose(f(x), expected, rtol=1e-6)


def test_threadgroup_memory_over_device_budget_is_rejected():
    """The device budget is checked before Metal rejects the pipeline with a vaguer message."""
    limit = device_limits()["max_threadgroup_memory_length"]
    too_many = limit // 4 + 1024  # in f32 elements
    f = _block_sum_call(64, threadgroup=TG, extent=too_many)
    with pytest.raises(EmitError, match="max_threadgroup_memory_length"):
        f(np.zeros(64, dtype=np.float32))


def test_threadgroup_over_device_thread_limit_is_rejected():
    limit = device_limits()["max_threads_per_threadgroup"]
    f = _block_sum_call(64, threadgroup=limit * 2)
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


def test_cache_is_bounded(rng):
    """Under JAX's trace caching a hit need not re-enter Python, so the
    bound is on entries, not on recency."""
    # out_shape is fixed while the input length is free, so every n is a new key.
    f = palladium.metal_call(
        _sum_kernel, out_shape=jax.ShapeDtypeStruct((1,), jnp.float32), cache_size=2
    )
    for n in (8, 16, 24):
        f(rng.standard_normal(n, dtype=np.float32))
    assert list(f._cache) == [_key(16), _key(24)]
    f(rng.standard_normal(32, dtype=np.float32))
    assert list(f._cache) == [_key(24), _key(32)]


def test_cache_size_zero_disables_eviction(rng):
    f = palladium.metal_call(
        _sum_kernel, out_shape=jax.ShapeDtypeStruct((1,), jnp.float32), cache_size=0
    )
    for n in (8, 16, 24, 32):
        f(rng.standard_normal(n, dtype=np.float32))
    assert list(f._cache) == [_key(n) for n in (8, 16, 24, 32)]
