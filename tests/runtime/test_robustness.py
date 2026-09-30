"""Robustness: thread safety and the dtype coverage matrix."""

import threading
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np

import palladium

# --- thread safety -----------------------------------------------------------


def test_concurrent_first_calls_emit_once_per_shape(monkeypatch, rng):
    import palladium._callable as callable_module

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


def _roundtrip_and_op(dtype, op, expect, x):
    def kernel(x_ref, o_ref):
        o_ref[...] = op(x_ref[...])

    call = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, dtype))
    got = np.asarray(call(x))
    want = np.asarray(expect(x))
    if np.issubdtype(want.dtype, np.floating):
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
    else:
        np.testing.assert_array_equal(got, want)


def test_dtype_float16(rng):
    x = rng.standard_normal(16).astype(np.float16)
    _roundtrip_and_op(jnp.float16, lambda v: v * 2.0 + 1.0, lambda v: v * 2 + 1, x)


def test_dtype_int32():
    x = np.arange(-8, 8, dtype=np.int32)
    _roundtrip_and_op(jnp.int32, lambda v: v * 3 + 1, lambda v: v * 3 + 1, x)


def test_dtype_uint32():
    x = np.arange(16, dtype=np.uint32)
    _roundtrip_and_op(jnp.uint32, lambda v: v ^ np.uint32(0xFF), lambda v: v ^ 0xFF, x)


def test_dtype_bool():
    x = np.array([True, False] * 8)
    _roundtrip_and_op(jnp.bool_, jnp.logical_not, np.logical_not, x)


def test_dtype_float16_reduction(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], keepdims=True)

    x = rng.standard_normal(16).astype(np.float16)
    call = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((1,), jnp.float16))
    got = np.asarray(call(x))
    np.testing.assert_allclose(got, np.sum(x, keepdims=True), rtol=1e-2, atol=1e-2)


def test_dtype_int32_reduction():
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], keepdims=True)

    x = np.arange(16, dtype=np.int32)
    call = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((1,), jnp.int32))
    np.testing.assert_array_equal(np.asarray(call(x)), [x.sum()])


def test_bfloat16_round_trips_through_the_handler():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + jnp.bfloat16(1.0)

    call = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.bfloat16))
    x = jnp.arange(8, dtype=jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(call(x)).astype(np.float32), np.arange(1, 9, dtype=np.float32)
    )
