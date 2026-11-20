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
    """Check default grid, automatic threadgroup, kernel name, and source-size diagnostics."""
    diag = _classic_call().explain(_shaped(8, 8), _shaped(8, 8))
    assert diag.grid == (1, 1, 1)
    assert diag.threadgroup is None
    assert diag.msl_lines > 0
    assert f"palladium kernel {diag.name}" in str(diag)


def test_explain_explicit_threadgroup():
    """Check that an integer threadgroup option is padded to three dimensions in diagnostics."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    call = palladium.metal_call(
        kernel, out_shape=_shaped(64), compiler_params=palladium.CompilerParams(threadgroup=64)
    )
    diag = call.explain(_shaped(64))
    assert diag.threadgroup == (64, 1, 1)


def test_explain_env_logs_once_per_compile(monkeypatch, capsys, rng):
    """Check that PALLADIUM_EXPLAIN logs once for a specialization and stays silent on a cache hit."""
    monkeypatch.setenv("PALLADIUM_EXPLAIN", "1")
    call = _classic_call()
    x = rng.standard_normal((8, 8), dtype=np.float32)
    y = rng.standard_normal((8, 8), dtype=np.float32)
    call(x, y)
    call(x, y)
    assert capsys.readouterr().err.count("palladium kernel") == 1


def test_explain_env_does_not_emit_a_second_time(monkeypatch, rng):
    """Check that the PALLADIUM_EXPLAIN log line is built from the emission that is
    compiled, not from another one."""
    import palladium.diagnostics
    import palladium.ffi

    emissions = []
    for module in (palladium.ffi, palladium.diagnostics):
        emit = module.emit_msl_stats
        monkeypatch.setattr(
            module,
            "emit_msl_stats",
            lambda *a, emit=emit, **k: emissions.append(1) or emit(*a, **k),
        )
    monkeypatch.setenv("PALLADIUM_EXPLAIN", "1")
    x = rng.standard_normal((8, 8), dtype=np.float32)
    _classic_call()(x, x)
    assert len(emissions) == 1


def test_explain_string_reports_shared_storage_against_the_device_budget():
    """A cooperative kernel's report names its group and its threadgroup memory."""
    from jax.experimental import pallas as pl

    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = jnp.dot(x_ref[...], y_ref[...])

    call = palladium.metal_call(
        kernel,
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((16, 16), lambda i, j: (i, 0)),
            pl.BlockSpec((16, 32), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((16, 32), lambda i, j: (i, j)),
        out_shape=_shaped(32, 64),
        compiler_params=palladium.CompilerParams(dot_general="tensorops"),
    )
    diag = call.explain(_shaped(32, 16), _shaped(16, 64))
    text = str(diag)
    assert f"threadgroup={diag.threadgroup}" in text
    assert ("shared=" in text) == bool(diag.threadgroup_bytes)


def test_unsupported_input_dtype_is_a_dispatch_error():
    """An input whose dtype has no Metal type fails at the call, naming the dtype."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    call = palladium.metal_call(kernel, out_shape=_shaped(8))
    with pytest.raises(DispatchError, match="argument 0 has dtype complex64"):
        call(np.zeros(8, np.complex64))


def test_all_errors_are_palladium_errors():
    """Check stage-specific exception inheritance, including ValueError, TypeError, and
    NotImplementedError."""
    for exc in (TraceError, EmitError, UnsupportedPrimitiveError, DispatchError):
        assert issubclass(exc, palladium.PalladiumError)
    assert issubclass(TraceError, ValueError)
    assert issubclass(DispatchError, TypeError)
    assert issubclass(UnsupportedPrimitiveError, NotImplementedError)
    assert issubclass(UnsupportedPrimitiveError, EmitError)
