"""Robustness: thread safety, and sums in narrow and integer dtypes."""

import threading
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import palladium

# --- thread safety -----------------------------------------------------------


def test_concurrent_first_calls_emit_once_per_shape(monkeypatch, rng):
    """Check correct concurrent first-call results and exactly one emission for each of two shape
    specializations."""
    import palladium.ffi as callable_module

    emits = []
    real_emit = callable_module.emit_msl

    def counting_emit(*args, **kwargs):
        emits.append(1)
        return real_emit(*args, **kwargs)

    monkeypatch.setattr(callable_module, "emit_msl", counting_emit)

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0 + 1.0

    n_threads = 16
    call = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))
    # out_shape is fixed per callable, so two callables exercise two cache
    # entries under one barrier.
    call16 = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((16,), jnp.float32))
    inputs = [
        rng.standard_normal(8 if i % 2 == 0 else 16, dtype=np.float32) for i in range(n_threads)
    ]
    barrier = threading.Barrier(n_threads)

    def work(i):
        barrier.wait()  # maximize first-call contention
        target = call if i % 2 == 0 else call16
        return np.asarray(target(inputs[i]))

    with ThreadPoolExecutor(n_threads) as pool:
        results = list(pool.map(work, range(n_threads)))
    for i, out in enumerate(results):
        np.testing.assert_allclose(out, inputs[i] * 2.0 + 1.0, rtol=1e-6)
    assert len(emits) == 2, f"expected 2 emits, emit_msl ran {len(emits)} times"
    assert len(call._cache) == 1 and len(call16._cache) == 1


def test_concurrent_calls_on_one_compiled_kernel(rng):
    """Check that concurrent reuse of one compiled kernel keeps each call's input and output
    buffers distinct."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1.0

    call = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((64,), jnp.float32))
    call(rng.standard_normal(64, dtype=np.float32))  # compile once
    inputs = [rng.standard_normal(64, dtype=np.float32) for _ in range(32)]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda a: np.asarray(call(a)), inputs))
    for a, out in zip(inputs, results, strict=True):
        np.testing.assert_allclose(out, a + 1.0, rtol=1e-6)


# --- dtype coverage ----------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float16, np.int32])
def test_sum_accumulates_in_its_own_dtype(dtype):
    """Compare sums of half-precision and int32 blocks with NumPy exactly."""

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], keepdims=True)

    x = (np.arange(16) - 11).astype(dtype)
    call = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((1,), dtype))
    np.testing.assert_array_equal(np.asarray(call(x)), np.sum(x, keepdims=True))
