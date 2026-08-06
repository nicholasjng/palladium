"""Diagnostics and contract tests: explain(), the error taxonomy, call
boundary validation, and the dot-rule contract fixes (rank-1
canonicalization, preferred_element_type).

Sabotage checklist (each mechanism was broken deliberately and the
matching test failed): dtype check removed from BoundKernel.launch
(silent cast came back), rank-1 canonicalization disabled (contraction
EmitError). Exception: removing the promote cast in the scalar dot does
NOT currently fail the accumulation test, because the Metal compiler
contracts `float_acc += half * half` into a float fma anyway (verified
against the sabotaged MSL); the test pins the required behavior against
future compilers, and the cast makes the semantics explicit in the source.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import palladium
from palladium import (
    DispatchError,
    EmitError,
    TraceError,
    UnsupportedPrimitiveError,
)

F32 = jnp.float32


def _shaped(*shape):
    return jax.ShapeDtypeStruct(shape, F32)


def _classic_call():
    def kernel(q_ref, k_ref, o_ref):
        o_ref[...] = jnp.dot(q_ref[...], k_ref[...] * 2.0)

    return palladium.metal_call(kernel, out_shape=_shaped(8, 8))


# --- explain() -------------------------------------------------------------


def test_explain_reports_geometry():
    diag = _classic_call().explain(_shaped(8, 8), _shaped(8, 8))
    assert diag.grid == (1,)
    assert diag.threadgroup is None
    assert diag.msl_lines > 0
    assert f"palladium kernel {diag.name}" in str(diag)


def test_explain_explicit_threadgroup():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    call = palladium.metal_call(kernel, out_shape=_shaped(64), threadgroup=64)
    diag = call.explain(_shaped(64))
    assert diag.threadgroup == (64,)
    assert "threadgroup=(64,)" in str(diag)


def test_explain_ffi_matches_metal_call():
    def kernel(q_ref, k_ref, o_ref):
        o_ref[...] = jnp.dot(q_ref[...], k_ref[...] * 2.0)

    call = palladium.metal_call_jit(kernel, out_shape=_shaped(8, 8))
    diag = call.explain(_shaped(8, 8), _shaped(8, 8))
    assert diag.grid == (1,)


def test_palladium_explain_env_logs_once_per_compile(monkeypatch, capsys, rng):
    monkeypatch.setenv("PALLADIUM_EXPLAIN", "1")
    call = _classic_call()
    x = rng.standard_normal((8, 8), dtype=np.float32)
    y = rng.standard_normal((8, 8), dtype=np.float32)
    call(x, y)
    call(x, y)  # cached shape: no second line
    err = capsys.readouterr().err
    assert err.count("palladium kernel") == 1


# --- error taxonomy --------------------------------------------------------


def test_all_errors_are_palladium_errors():
    for exc in (TraceError, EmitError, UnsupportedPrimitiveError, DispatchError):
        assert issubclass(exc, palladium.PalladiumError)
    # Pre-hierarchy behavior kept: TraceError catches as ValueError,
    # DispatchError as TypeError, UnsupportedPrimitiveError as
    # NotImplementedError.
    assert issubclass(TraceError, ValueError)
    assert issubclass(DispatchError, TypeError)
    assert issubclass(UnsupportedPrimitiveError, NotImplementedError)


def test_unsupported_primitive_names_it():
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sort(x_ref[...])

    with pytest.raises(UnsupportedPrimitiveError, match="sort"):
        palladium.debug_msl(kernel, _shaped(8), out_shape=_shaped(8))


def test_batched_dot_general_rejected():
    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.einsum("bij,bjk->bik", a_ref[...], b_ref[...])

    with pytest.raises(EmitError, match="batch dims"):
        palladium.debug_msl(
            kernel, _shaped(2, 4, 4), _shaped(2, 4, 4), out_shape=_shaped(2, 4, 4)
        )


def test_scan_with_stacked_ys_rejected():
    def kernel(x_ref, o_ref):
        def body(c, x):
            return c + x, c

        _c, ys = jax.lax.scan(body, x_ref[0], x_ref[...])
        o_ref[...] = ys

    with pytest.raises(EmitError, match="stacked ys"):
        palladium.debug_msl(kernel, _shaped(8), out_shape=_shaped(8))


def test_strided_ref_access_rejected():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[::2]

    with pytest.raises(EmitError, match="stride"):
        palladium.debug_msl(kernel, _shaped(8), out_shape=_shaped(4))


def test_multiple_pallas_calls_rejected():
    import jax.experimental.pallas as pl

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    def two_calls(x):
        y = pl.pallas_call(kernel, out_shape=_shaped(8))(x)
        return pl.pallas_call(kernel, out_shape=_shaped(8))(y)

    with pytest.raises(TraceError, match="2 pallas_call"):
        palladium.trace(two_calls, _shaped(8))


# --- call boundary ---------------------------------------------------------


def test_float64_input_rejected_with_hint():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    call = palladium.metal_call(kernel, out_shape=_shaped(8))
    with pytest.raises(DispatchError, match="float64.*jax_enable_x64"):
        call(np.zeros(8, dtype=np.float64))


def test_bound_kernel_rejects_dtype_and_shape_mismatch(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1.0

    call = palladium.metal_call(kernel, out_shape=_shaped(8))
    x = rng.standard_normal(8, dtype=np.float32)
    np.testing.assert_allclose(call(x), x + 1.0, rtol=1e-6)
    (bound,) = call.cache.values()
    with pytest.raises(DispatchError, match="dtype float64 does not match"):
        bound(x.astype(np.float64))
    with pytest.raises(DispatchError, match="expected shape"):
        bound(x[:4])
    with pytest.raises(DispatchError, match="takes 1 arrays"):
        bound(x, x)


# --- dot-rule contract -----------------------------------------------------


def _run_and_compare(kernel, out_shape, *arrays, tol=1e-5):
    call = palladium.metal_call(kernel, out_shape=out_shape)
    got = call(*arrays)
    want = call.interpret(*arrays)
    np.testing.assert_allclose(got, np.asarray(want), rtol=tol, atol=tol)


def test_rank1_matvec(rng):
    def kernel(m_ref, v_ref, o_ref):
        o_ref[...] = jnp.dot(m_ref[...], v_ref[...])

    m = rng.standard_normal((8, 16), dtype=np.float32)
    v = rng.standard_normal(16, dtype=np.float32)
    _run_and_compare(kernel, _shaped(8), m, v)


def test_rank1_vecmat(rng):
    def kernel(v_ref, m_ref, o_ref):
        o_ref[...] = jnp.dot(v_ref[...], m_ref[...])

    v = rng.standard_normal(16, dtype=np.float32)
    m = rng.standard_normal((16, 8), dtype=np.float32)
    _run_and_compare(kernel, _shaped(8), v, m)


def test_rank1_vecvec(rng):
    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...])

    a = rng.standard_normal(16, dtype=np.float32)
    b = rng.standard_normal(16, dtype=np.float32)
    _run_and_compare(kernel, jax.ShapeDtypeStruct((), F32), a, b)


def test_preferred_element_type_accumulates_in_f32():
    # a_i = b_i = 1 + 2^-10, exact in f16. The product 1 + 2^-9 + 2^-20
    # is exact in f32 but rounds to 1 + 2^-9 in f16, so a kernel that
    # multiplies in half before accumulating is off by k * 2^-20 = 2^-16,
    # past the 2^-17 tolerance below. Catches a dropped promote cast.
    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jax.lax.dot_general(
            a_ref[...],
            b_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    k = 16
    a = np.full((7, k), 1 + 2**-10, dtype=np.float16)
    b = np.full((k, 8), 1 + 2**-10, dtype=np.float16)
    call = palladium.metal_call(kernel, out_shape=_shaped(7, 8))
    got = call(a, b)
    want = np.full((7, 8), k * (1 + 2**-9 + 2**-20), dtype=np.float32)
    np.testing.assert_allclose(got, want, rtol=0, atol=2**-17)


def test_preferred_element_type_f16_to_f32(rng):
    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jax.lax.dot_general(
            a_ref[...],
            b_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    a = rng.standard_normal((8, 16)).astype(np.float16)
    b = rng.standard_normal((16, 8)).astype(np.float16)
    call = palladium.metal_call(kernel, out_shape=_shaped(8, 8))
    got = call(a, b)
    assert isinstance(got, np.ndarray)  # single out_shape: never a tuple
    assert got.dtype == np.float32
    want = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)
