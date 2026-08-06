"""Robustness: thread safety, the dtype coverage matrix, and resource
limits.

Sabotage checklist (each broken deliberately, matching test failed):
MetalCallable cache lock removed (bind ran more than twice under the
barrier), BoundKernel launch lock removed (cross-thread results
corrupted), tg_bytes accounting removed (over-budget kernel reached
Metal instead of raising EmitError).
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import palladium
from palladium import EmitError

# --- thread safety -----------------------------------------------------------


def test_concurrent_first_calls_compile_once_per_shape(monkeypatch, rng):
    calls = []
    real_bind = palladium.bind

    def counting_bind(*args, **kwargs):
        calls.append(kwargs.get("cooperative"))
        return real_bind(*args, **kwargs)

    monkeypatch.setattr(palladium, "bind", counting_bind)

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0 + 1.0

    n_threads = 16
    call = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32)
    )
    # Every thread gets its own data; half use shape (8,), half (16,).
    # jax.ShapeDtypeStruct out_shape is fixed, so use two separate
    # MetalCallables to exercise two cache entries under one barrier.
    call16 = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((16,), jnp.float32)
    )
    inputs = [
        rng.standard_normal(8 if i % 2 == 0 else 16, dtype=np.float32)
        for i in range(n_threads)
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
    assert len(calls) == 2, f"expected 2 compiles, bind ran {len(calls)} times"
    assert len(call.cache) == 1 and len(call16.cache) == 1


def test_concurrent_calls_on_one_bound_kernel(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1.0

    call = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((64,), jnp.float32)
    )
    warm = rng.standard_normal(64, dtype=np.float32)
    call(warm)  # compile once; the threads below share one BoundKernel
    (bound,) = call.cache.values()
    inputs = [rng.standard_normal(64, dtype=np.float32) for _ in range(32)]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda a: np.asarray(bound(a)), inputs))
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


def test_dtype_float32(rng):
    x = rng.standard_normal(16, dtype=np.float32)
    _roundtrip_and_op(jnp.float32, lambda v: v * 2.0 + 1.0, lambda v: v * 2 + 1, x)


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
    call = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((1,), jnp.float16)
    )
    got = np.asarray(call(x))
    np.testing.assert_allclose(got, np.sum(x, keepdims=True), rtol=1e-2, atol=1e-2)


def test_dtype_int32_reduction():
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], keepdims=True)

    x = np.arange(16, dtype=np.int32)
    call = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((1,), jnp.int32))
    np.testing.assert_array_equal(np.asarray(call(x)), [x.sum()])


def test_bfloat16_eager_roundtrip_is_bit_lossless():
    # NumPy cannot pass ml_dtypes arrays over DLPack, so the eager path
    # ships bf16 bytes as uint16 and relabels (dispatch._to_native). A
    # copy kernel must round-trip every bit pattern, including the NaN
    # payload a numeric conversion would normalize.
    import ml_dtypes

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    call = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.bfloat16)
    )
    bits = np.array(
        [0x0000, 0x8000, 0x3F80, 0x0001, 0x7F80, 0xFF80, 0x7FC1, 0x4049],
        dtype=np.uint16,
    )  # +0, -0, 1.0, subnormal, +inf, -inf, NaN with payload, pi-ish
    x = bits.view(ml_dtypes.bfloat16)
    got = call(x)
    assert got.dtype == x.dtype
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16), bits)


def test_bfloat16_eager_compute_and_pin(rng):
    import ml_dtypes

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0 + 1.0

    call = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((16,), jnp.bfloat16)
    )
    x = rng.standard_normal(16).astype(ml_dtypes.bfloat16)
    want = np.asarray(call.interpret(x)).astype(np.float32)
    got = np.asarray(call(x)).astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2)
    pinned = call.pin(x)
    np.testing.assert_allclose(
        np.asarray(pinned()).astype(np.float32), want, rtol=2e-2, atol=2e-2
    )


def test_bfloat16_dispatches_through_ffi():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + jnp.bfloat16(1.0)

    call = palladium.metal_call_jit(
        kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.bfloat16)
    )
    x = jnp.arange(8, dtype=jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(call(x)).astype(np.float32), np.arange(1, 9, dtype=np.float32)
    )


# --- resource limits ---------------------------------------------------------


def test_stack_overflow_names_the_fix():
    # A gridless kernel copies its whole 4096-element block into a
    # thread-local array, overflowing the per-thread stack. The opaque
    # Metal pipeline error must arrive wrapped with the actual fix.
    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = 2.0 * x_ref[...] + y_ref[...]

    call = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((4096,), jnp.float32)
    )
    with pytest.raises(EmitError, match="grid and BlockSpecs"):
        call(np.zeros(4096, dtype=np.float32), np.zeros(4096, dtype=np.float32))


def test_threadgroup_memory_over_budget_rejected():
    # 17 MMA dots at 8x64 f32 tiles = 17 * 2048 = 34816 bytes > 32768.
    n_ks = 17

    def kernel(*refs):
        q_ref, ks, o_ref = refs[0], refs[1:-1], refs[-1]
        total = None
        for k_ref in ks:
            s = jnp.dot(q_ref[...], k_ref[...].T)
            total = s if total is None else total + s
        o_ref[...] = total

    with pytest.raises(EmitError, match="threadgroup memory"):
        palladium.debug_msl(
            kernel,
            jax.ShapeDtypeStruct((8, 64), jnp.float32),
            *[jax.ShapeDtypeStruct((64, 64), jnp.float32) for _ in range(n_ks)],
            out_shape=jax.ShapeDtypeStruct((8, 64), jnp.float32),
        )
