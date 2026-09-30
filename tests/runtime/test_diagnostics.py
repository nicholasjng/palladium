"""Public diagnostics and dispatch-boundary contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import palladium
from palladium import DispatchError, EmitError, TraceError, UnsupportedPrimitiveError

F32 = jnp.float32


def _shaped(*shape):
    return jax.ShapeDtypeStruct(shape, F32)


def _classic_call():
    def kernel(q_ref, k_ref, o_ref):
        o_ref[...] = jnp.dot(q_ref[...], k_ref[...] * 2.0)

    return palladium.metal_call(kernel, out_shape=_shaped(8, 8))


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

    call = palladium.metal_call(kernel, out_shape=_shaped(8, 8))
    assert call.explain(_shaped(8, 8), _shaped(8, 8)).grid == (1,)


def test_explain_env_logs_once_per_compile(monkeypatch, capsys, rng):
    monkeypatch.setenv("PALLADIUM_EXPLAIN", "1")
    call = _classic_call()
    x = rng.standard_normal((8, 8), dtype=np.float32)
    y = rng.standard_normal((8, 8), dtype=np.float32)
    call(x, y)
    call(x, y)
    assert capsys.readouterr().err.count("palladium kernel") == 1


def test_all_errors_are_palladium_errors():
    for exc in (TraceError, EmitError, UnsupportedPrimitiveError, DispatchError):
        assert issubclass(exc, palladium.PalladiumError)
    assert issubclass(TraceError, ValueError)
    assert issubclass(DispatchError, TypeError)
    assert issubclass(UnsupportedPrimitiveError, NotImplementedError)
    assert issubclass(UnsupportedPrimitiveError, EmitError)


def test_multiple_pallas_calls_rejected():
    from jax.experimental import pallas as pl

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    def two_calls(x):
        y = pl.pallas_call(kernel, out_shape=_shaped(8))(x)
        return pl.pallas_call(kernel, out_shape=_shaped(8))(y)

    with pytest.raises(TraceError, match="2 pallas_call"):
        palladium.trace(two_calls, _shaped(8))
