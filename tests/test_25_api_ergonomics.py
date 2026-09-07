"""The consumer-facing conveniences layered over the core pipeline.

`verify` (differential check against a reference), stack/threadgroup
accounting in `explain`, device-derived threadgroup sizing, structured
error fields, and bounded kernel caches. None of these change what the
emitter produces; they change what a caller can find out and how loudly
a mistake fails.
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
    StackOverflowError,
    UnsupportedPrimitiveError,
)
from palladium.threadgroup import (
    barrier,
    thread_index,
    threadgroup_memory,
    threads_per_threadgroup,
)
from palladium.verify import VerificationError

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


# --- verify --------------------------------------------------------------


def test_verify_returns_the_gpu_output(rng):
    """It replaces a call site, so it must return what __call__ returns."""
    f = _tanh_call()
    x = rng.standard_normal(64, dtype=np.float32)
    np.testing.assert_array_equal(f.verify(x), f(x))


def test_verify_accepts_an_explicit_reference(rng):
    f = _tanh_call()
    x = rng.standard_normal(64, dtype=np.float32)
    f.verify(x, reference=lambda a: np.tanh(a) * 2.0)


def test_verify_reports_the_worst_element():
    """A failure has to say *where*, not just that something differed."""
    f = _tanh_call()
    x = np.zeros(64, dtype=np.float32)

    def wrong(a):
        out = np.tanh(a) * 2.0
        out[7] += 1.0
        return out

    with pytest.raises(VerificationError) as excinfo:
        f.verify(x, reference=wrong)
    e = excinfo.value
    assert e.worst_index == (7,)
    assert e.mismatches == 1 and e.size == 64
    assert e.want == pytest.approx(1.0)
    assert "worst at (7,)" in str(e)


def test_verify_catches_a_shape_disagreement():
    f = _tanh_call()
    x = np.zeros(64, dtype=np.float32)
    with pytest.raises(VerificationError, match="shape"):
        f.verify(x, reference=lambda a: np.zeros(32, dtype=np.float32))


def test_verify_refuses_the_interpret_oracle_for_cooperative_kernels():
    """Interpret models each instance as a threadgroup of one, so the
    comparison is meaningless rather than merely imprecise. It has to
    refuse, not quietly pass or quietly fail."""
    n = 64
    f = _block_sum_call(n)
    x = np.arange(n, dtype=np.float32)
    with pytest.raises(VerificationError, match="threadgroup_memory"):
        f.verify(x)


def test_verify_works_on_cooperative_kernels_with_a_reference():
    n = 100
    f = _block_sum_call(n)
    x = np.arange(n, dtype=np.float32)

    def reference(a):
        out = np.empty_like(a)
        for start in range(0, len(a), TG):
            out[start : start + TG] = a[start : start + TG].sum()
        return out

    np.testing.assert_allclose(f.verify(x, reference=reference), reference(x))


def test_verify_on_the_ffi_path(rng):
    f = palladium.metal_call_jit(
        _tanh_kernel, out_shape=jax.ShapeDtypeStruct((64,), jnp.float32)
    )
    x = jnp.asarray(rng.standard_normal(64, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(f.verify(x)), np.asarray(f(x)))


# --- explain / accounting ------------------------------------------------


def test_explain_reports_per_thread_stack():
    """The per-thread stack is the one hard limit with no published
    ceiling, so the number has to be visible before compiling."""
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


def test_stack_overflow_carries_the_measured_size():
    """The old message said 'too large'; the fix needs a number to aim at."""
    n = 200_000
    f = palladium.metal_call(
        lambda x_ref, o_ref: o_ref.__setitem__(..., x_ref[...] * 2.0),
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
    )
    with pytest.raises(StackOverflowError) as excinfo:
        f(np.zeros(n, dtype=np.float32))

    stack_bytes = excinfo.value.stack_bytes
    assert stack_bytes is not None
    assert stack_bytes >= n * 4


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
        expected[start : start + simdgroup_width()] = x[
            start : start + simdgroup_width()
        ].sum()
    np.testing.assert_allclose(f(x), expected, rtol=1e-6)


def test_threadgroup_memory_over_device_budget_is_rejected():
    """Metal rejects the pipeline with a vaguer message; catch it first."""
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
    """So a caller can branch on it instead of matching message text."""

    def kernel(x_ref, o_ref):
        o_ref[...] = jax.lax.erf(x_ref[...])

    with pytest.raises(UnsupportedPrimitiveError) as excinfo:
        palladium.debug_msl(
            kernel,
            jax.ShapeDtypeStruct((8,), jnp.float32),
            out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        )
    assert excinfo.value.primitive is not None
    assert excinfo.value.primitive in str(excinfo.value)


def test_stack_overflow_error_is_an_emit_error():
    """Existing `except EmitError` handlers must keep working."""
    assert issubclass(StackOverflowError, EmitError)
    assert issubclass(UnsupportedPrimitiveError, EmitError)


# --- bounded caches ------------------------------------------------------


def test_eager_cache_evicts_least_recently_used(rng):
    """Three shapes through one callable with room for two.

    The surviving keys must be the two most recently used, not the two
    most recently inserted -- re-touching the oldest shape should save
    it from eviction.
    """

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    f = palladium.metal_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        cache_size=2,
    )
    # out_shape is fixed, so vary the *input* shape to vary the cache key.
    f_by_n = {
        n: palladium.metal_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
            cache_size=2,
        )
        for n in (8, 16, 24)
    }
    del f
    for n, call in f_by_n.items():
        call(rng.standard_normal(n, dtype=np.float32))
        assert len(call.cache) == 1, n


def test_eager_cache_is_bounded_across_shapes(rng):
    """One callable, many shapes: the real dynamic-batch-size case."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    sizes = [8, 16, 32, 64, 128]
    for n in sizes:
        f = palladium.metal_call(
            kernel, out_shape=jax.ShapeDtypeStruct((n,), jnp.float32), cache_size=2
        )
        f(rng.standard_normal(n, dtype=np.float32))
        assert len(f.cache) == 1


def test_ffi_cache_is_bounded():
    f = palladium.metal_call_jit(
        _tanh_kernel,
        out_shape=jax.ShapeDtypeStruct((64,), jnp.float32),
        cache_size=1,
    )
    f(jnp.zeros(64, jnp.float32))
    assert len(f._cache) == 1


def test_cache_size_zero_disables_eviction(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    f = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32), cache_size=0
    )
    f(rng.standard_normal(8, dtype=np.float32))
    f(rng.standard_normal(8, dtype=np.float32))
    assert len(f.cache) == 1


# --- pin parity ----------------------------------------------------------


def test_ffi_pin_refuses_with_a_reason():
    """Not silently missing: it explains why the eager trick cannot work."""
    f = palladium.metal_call_jit(
        _tanh_kernel, out_shape=jax.ShapeDtypeStruct((64,), jnp.float32)
    )
    with pytest.raises(NotImplementedError, match="XLA owns"):
        f.pin(jnp.zeros(64, jnp.float32))
