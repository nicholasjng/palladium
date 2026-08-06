"""Stretch 10: jax.ffi integration (`native/ffi/palladium_ffi.cpp`).

`metal_call_jit` emits MSL the same way `metal_call` does, but dispatches
through a registered jax.ffi target, so the result composes inside
`jax.jit` next to ordinary `jnp` ops. The native handler ships inside the
`palladium` package (root `CMakeLists.txt`, scikit-build-core) and is
built automatically by `uv sync`/`pip install`; this skips cleanly only
if that didn't happen (e.g. an unbuilt from-source checkout).
"""

import jax
import jax.numpy as jnp
import metal_runtime as mr
import numpy as np
import pytest

import palladium
from palladium.ffi import _library_path

try:
    _library_path()
    _ffi_available = True
except FileNotFoundError:
    _ffi_available = False

pytestmark = pytest.mark.skipif(
    not _ffi_available, reason="palladium's native jax.ffi handler isn't built"
)


def _add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def test_eager_call_matches_interpret(rng):
    x = rng.standard_normal((8, 8), dtype=np.float32)
    y = rng.standard_normal((8, 8), dtype=np.float32)
    call = palladium.metal_call_jit(
        _add_kernel, out_shape=jax.ShapeDtypeStruct((8, 8), jnp.float32)
    )
    got = np.asarray(call(x, y))
    want = np.asarray(call.interpret(x, y))
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_composes_inside_jax_jit_with_ordinary_jnp_ops(rng):
    """The actual point: a bare palladium kernel result feeding into
    jnp.sum under jax.jit, no np.asarray/tracer conflict."""
    x = rng.standard_normal((8, 8), dtype=np.float32)
    y = rng.standard_normal((8, 8), dtype=np.float32)
    call = palladium.metal_call_jit(
        _add_kernel, out_shape=jax.ShapeDtypeStruct((8, 8), jnp.float32)
    )

    @jax.jit
    def composed(a, b):
        return jnp.sum(call(a, b) ** 2)

    got = float(composed(x, y))
    want = float(np.sum((x + y) ** 2))
    assert got == pytest.approx(want, rel=1e-4)


def test_repeated_calls_reuse_the_cached_kernel(rng):
    """Second call with the same shape/dtype must not retrace or
    recompile; correctness across two distinct inputs is what proves the
    cached MSL/spec weren't stale for the second one."""
    call = palladium.metal_call_jit(
        _add_kernel, out_shape=jax.ShapeDtypeStruct((4, 4), jnp.float32)
    )

    @jax.jit
    def composed(a, b):
        return jnp.sum(call(a, b) ** 2)

    x1 = rng.standard_normal((4, 4), dtype=np.float32)
    x2 = rng.standard_normal((4, 4), dtype=np.float32)
    y = np.ones((4, 4), dtype=np.float32)

    assert len(call._cache) == 0
    r1 = float(composed(x1, y))
    assert len(call._cache) == 1
    r2 = float(composed(x2, y))
    assert len(call._cache) == 1  # same shape/dtype: cache hit, not a second entry

    assert r1 == pytest.approx(float(np.sum((x1 + y) ** 2)), rel=1e-4)
    assert r2 == pytest.approx(float(np.sum((x2 + y) ** 2)), rel=1e-4)


def test_multi_output_kernel(rng):
    """RemainingRets with more than one buffer: a genuinely different code
    path from the single-output case (list vs bare ShapeDtypeStruct on
    the Python side, n_inputs offset into `wrapped` on the C++ side)."""

    def kernel(x_ref, y_ref, sum_ref, diff_ref):
        sum_ref[...] = x_ref[...] + y_ref[...]
        diff_ref[...] = x_ref[...] - y_ref[...]

    x = rng.standard_normal((6, 6), dtype=np.float32)
    y = rng.standard_normal((6, 6), dtype=np.float32)
    call = palladium.metal_call_jit(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((6, 6), jnp.float32),
            jax.ShapeDtypeStruct((6, 6), jnp.float32),
        ),
    )

    s, d = call(x, y)
    np.testing.assert_allclose(s, x + y, atol=1e-6)
    np.testing.assert_allclose(d, x - y, atol=1e-6)

    @jax.jit
    def composed(a, b):
        s, d = call(a, b)
        return jnp.sum(s) + jnp.sum(d**2)

    got = float(composed(x, y))
    want = float(np.sum(x + y) + np.sum((x - y) ** 2))
    assert got == pytest.approx(want, rel=1e-4)


def test_math_mode_safe_is_actually_requested(rng):
    """`x + nan` only reliably produces NaN under SAFE: FAST permits
    Metal to assume NaN never occurs (test_07_preflight.py's
    `test_nan_literal_propagates` establishes this for the base
    metal_call path). Confirmed load-bearing by sabotage: swapping the
    SAFE/FAST ordinals in ffi.py's _MATH_MODE_ORDINALS makes this fail,
    so a passing run means math_mode is actually reaching
    mr_compile_library, not silently ignored."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.where(x > 0.0, x + jnp.nan, x)

    f = palladium.metal_call_jit(
        kernel,
        math_mode=mr.MathMode.SAFE,
        out_shape=jax.ShapeDtypeStruct((64,), jnp.float32),
    )
    x = rng.standard_normal(64, dtype=np.float32)
    got = f(x)
    want = np.asarray(f.interpret(x))
    assert np.array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)])


def test_cooperative_kernel_dispatches_through_ffi(rng):
    """A query-blocked attention kernel takes the cooperative model
    through the FFI path: the launch must carry the (32, 1, 1)
    threadgroup geometry or every SIMD-group would compute instance 0."""
    from jax.experimental import pallas as pl

    seq, hd, bq, bkv = 256, 64, 8, 32
    scale = 1.0 / (hd**0.5)

    def kernel(q_ref, k_ref, v_ref, o_ref):
        q = q_ref[...]

        def step(j, carry):
            m, l, o = carry
            kj = k_ref[pl.dslice(j * bkv, bkv), :]
            vj = v_ref[pl.dslice(j * bkv, bkv), :]
            s = jnp.dot(q, kj.T) * scale
            m_new = jnp.maximum(m, jnp.max(s, axis=1, keepdims=True))
            corr = jnp.exp(m - m_new)
            p = jnp.exp(s - m_new)
            l_new = l * corr + jnp.sum(p, axis=1, keepdims=True)
            return (m_new, l_new, o * corr + jnp.dot(p, vj))

        m0 = jnp.full((bq, 1), -jnp.inf, jnp.float32)
        l0 = jnp.zeros((bq, 1), jnp.float32)
        o0 = jnp.zeros((bq, hd), jnp.float32)
        _, l, o = jax.lax.fori_loop(0, seq // bkv, step, (m0, l0, o0))
        o_ref[...] = o / l

    f = palladium.metal_call_jit(
        kernel,
        grid=(seq // bq,),
        in_specs=[
            pl.BlockSpec((bq, hd), lambda i: (i, 0)),
            pl.BlockSpec((seq, hd), lambda i: (0, 0)),
            pl.BlockSpec((seq, hd), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((bq, hd), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((seq, hd), jnp.float32),
    )
    q = rng.standard_normal((seq, hd)).astype(np.float32)
    k = rng.standard_normal((seq, hd)).astype(np.float32)
    v = rng.standard_normal((seq, hd)).astype(np.float32)

    got = np.asarray(f(q, k, v))
    want = np.asarray(f.interpret(q, k, v))
    np.testing.assert_allclose(got, want, atol=1e-4)

    # the cached entry must actually be the cooperative lowering
    (entry,) = f._cache.values()
    assert entry[2] is True and "simdgroup" in entry[1]

    # and it composes inside jax.jit next to jnp ops
    jitted = jax.jit(lambda q, k, v: jnp.tanh(f(q, k, v)).sum())
    np.testing.assert_allclose(
        float(jitted(q, k, v)), float(np.tanh(want).sum()), rtol=1e-4
    )
