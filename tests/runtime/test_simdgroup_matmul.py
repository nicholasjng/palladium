"""`dot_general="simdgroup"`: float32 tiled matmuls built from `simdgroup_matrix`.

Supported shapes use the kernel of `emit/tensorops/simdgroup.py`; everything
else falls back to the Metal 4 `matmul2d` lowering. Both are checked against
NumPy through `metal_call`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium

SIMDGROUP = palladium.CompilerParams(dot_general="simdgroup")


def _dot(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.matmul(a_ref[...], b_ref[...])


def _relu_dot(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.maximum(jnp.matmul(a_ref[...], b_ref[...]), 0.0)


def _residual_dot(a_ref, b_ref, residual_ref, out_ref):
    out_ref[...] = jnp.matmul(a_ref[...], b_ref[...]) + residual_ref[...]


def _bias_relu_dot(a_ref, b_ref, bias_ref, out_ref):
    out_ref[...] = jnp.maximum(jnp.matmul(a_ref[...], b_ref[...]) + bias_ref[...], 0.0) * 2.0


def _dot_rhs_transposed(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.matmul(a_ref[...], jnp.swapaxes(b_ref[...], 0, 1))


def _matmul_call(kernel, m, n, k, tm, tn, *, extra_specs=(), dtype=jnp.float32):
    return palladium.metal_call(
        kernel,
        grid=(pl.cdiv(m, tm), pl.cdiv(n, tn)),
        in_specs=[
            pl.BlockSpec((tm, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, tn), lambda i, j: (0, j)),
            *extra_specs,
        ],
        out_specs=pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        compiler_params=SIMDGROUP,
    )


def _source(call, *args):
    _, msl, _ = call._spec_and_msl(tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args))
    return msl


def _operands(rng, m, n, k, dtype=np.float32):
    return (
        rng.standard_normal((m, k)).astype(dtype),
        rng.standard_normal((k, n)).astype(dtype),
    )


@pytest.mark.parametrize(
    ("m", "n", "k", "tm", "tn"),
    [(64, 64, 32, 16, 16), (128, 128, 64, 32, 32), (256, 192, 48, 64, 64), (256, 128, 80, 128, 64)],
)
def test_simdgroup_matmul_matches_numpy(metal_device, rng, m, n, k, tm, tn):
    """Supported tiles use the simdgroup kernel and match NumPy."""
    a, b = _operands(rng, m, n, k)
    call = _matmul_call(_dot, m, n, k, tm, tn)
    assert "simdgroup_mma" in _source(call, a, b)
    np.testing.assert_allclose(call(a, b), a @ b, rtol=1e-5, atol=1e-4)


def test_simdgroup_matmul_adds_residual_tile(metal_device, rng):
    """An epilogue reading another output-shaped tile indexes it by row and column."""
    a, b = _operands(rng, 64, 96, 32)
    residual = rng.standard_normal((64, 96)).astype(np.float32)
    call = _matmul_call(
        _residual_dot,
        64,
        96,
        32,
        32,
        32,
        extra_specs=(pl.BlockSpec((32, 32), lambda i, j: (i, j)),),
    )
    assert "simdgroup_mma" in _source(call, a, b, residual)
    np.testing.assert_allclose(call(a, b, residual), a @ b + residual, rtol=1e-5, atol=1e-4)


def test_simdgroup_matmul_applies_column_bias_chain(metal_device, rng):
    """A column bias, ReLU and scale compose in the per-lane epilogue."""
    a, b = _operands(rng, 64, 64, 48)
    bias = rng.standard_normal((64,)).astype(np.float32)
    call = _matmul_call(
        _bias_relu_dot,
        64,
        64,
        48,
        32,
        32,
        extra_specs=(pl.BlockSpec((32,), lambda i, j: (j,)),),
    )
    assert "simdgroup_mma" in _source(call, a, b, bias)
    np.testing.assert_allclose(
        call(a, b, bias), np.maximum(a @ b + bias, 0.0) * 2.0, rtol=1e-5, atol=1e-4
    )


def test_simdgroup_matmul_handles_batches(metal_device, rng):
    """Rank-3 batches address each program's block through its batch offset."""
    batch, m, n, k, tm, tn = 3, 64, 64, 32, 32, 32
    a = rng.standard_normal((batch, m, k)).astype(np.float32)
    b = rng.standard_normal((batch, k, n)).astype(np.float32)
    call = palladium.metal_call(
        _dot,
        grid=(batch, m // tm, n // tn),
        in_specs=[
            pl.BlockSpec((1, tm, k), lambda z, i, j: (z, i, 0)),
            pl.BlockSpec((1, k, tn), lambda z, i, j: (z, 0, j)),
        ],
        out_specs=pl.BlockSpec((1, tm, tn), lambda z, i, j: (z, i, j)),
        out_shape=jax.ShapeDtypeStruct((batch, m, n), jnp.float32),
        compiler_params=SIMDGROUP,
    )
    assert "simdgroup_mma" in _source(call, a, b)
    np.testing.assert_allclose(call(a, b), a @ b, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(
    ("case", "m", "n", "k", "tm", "tn"),
    [
        ("edge tiles", 48, 64, 32, 32, 32),
        ("k not a multiple of 16", 64, 64, 24, 32, 32),
        ("tile too large", 128, 128, 32, 128, 128),
    ],
)
def test_simdgroup_policy_falls_back_to_matmul2d(metal_device, rng, case, m, n, k, tm, tn):
    """Shapes the simdgroup kernel does not support keep `matmul2d` and stay correct."""
    a, b = _operands(rng, m, n, k)
    call = _matmul_call(_dot, m, n, k, tm, tn)
    source = _source(call, a, b)
    assert "matmul2d" in source and "simdgroup_mma" not in source, case
    np.testing.assert_allclose(call(a, b), a @ b, rtol=1e-5, atol=1e-4)


def test_simdgroup_policy_keeps_matmul2d_for_transposed_operands(metal_device, rng):
    """A lazily transposed operand keeps the `matmul2d` lowering."""
    m, n, k, tm, tn = 64, 64, 32, 32, 32
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((n, k)).astype(np.float32)
    call = palladium.metal_call(
        _dot_rhs_transposed,
        grid=(m // tm, n // tn),
        in_specs=[
            pl.BlockSpec((tm, k), lambda i, j: (i, 0)),
            pl.BlockSpec((tn, k), lambda i, j: (j, 0)),
        ],
        out_specs=pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        compiler_params=SIMDGROUP,
    )
    source = _source(call, a, b)
    assert "matmul2d" in source and "simdgroup_mma" not in source
    np.testing.assert_allclose(call(a, b), a @ b.T, rtol=1e-5, atol=1e-4)


def test_simdgroup_kernel_splits_for_the_mps_descriptor():
    """The mps path keeps the helper in the header and rebinds the new builtins."""
    from palladium.mps import kernel_prologue, split_kernel_source

    shape = jax.ShapeDtypeStruct((64, 64), jnp.float32)
    source = palladium.debug_msl(
        _dot,
        shape,
        shape,
        grid=(2, 2),
        in_specs=(
            pl.BlockSpec((32, 64), lambda i, j: (i, 0)),
            pl.BlockSpec((64, 32), lambda i, j: (0, j)),
        ),
        out_specs=pl.BlockSpec((32, 32), lambda i, j: (i, j)),
        out_shape=shape,
        compiler_params=SIMDGROUP,
    )
    header, params, body = split_kernel_source(source)
    prologue = kernel_prologue(params)
    assert "METAL_FUNC void simdgroup_mma" in header
    assert "simdgroup_mma(" in body
    for builtin in (
        "simdgroup_index_in_threadgroup",
        "thread_index_in_simdgroup",
        "thread_index_in_threadgroup",
    ):
        assert f"({builtin});" in prologue


def test_simdgroup_policy_keeps_matmul2d_for_float16(metal_device, rng):
    """Half precision stays on `matmul2d`, which is faster for it."""
    a, b = _operands(rng, 64, 64, 32, np.float16)
    call = _matmul_call(_dot, 64, 64, 32, 32, 32, dtype=jnp.float16)
    source = _source(call, a, b)
    assert "matmul2d" in source and "simdgroup_mma" not in source
    want = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(call(a, b).astype(np.float32), want, rtol=2e-2, atol=2e-2)
