"""`dot_general="simdgroup"`: tiled matmuls built from float32 `simdgroup_matrix`.

Supported shapes use the kernel of `emit/tensorops/simdgroup.py`; everything
else falls back to the Metal 4 `matmul2d` lowering. Both are checked against
NumPy through `metal_call`. Epilogues on both lowerings are in
`tests/codegen/test_tensorops_device.py`.
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


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_simdgroup_matmul_widens_half_types(metal_device, rng, dtype):
    """Half and bfloat tiles stay narrow in threadgroup memory and accumulate in
    float32, so the result is the float32 product rounded once."""
    m, n, k = 128, 64, 512
    a, b = (x.astype(dtype) for x in _operands(rng, m, n, k))
    call = _matmul_call(_relu_dot, m, n, k, 64, 32, dtype=dtype)
    source = _source(call, a, b)
    assert "simdgroup_mma" in source and "matmul2d" not in source
    assert f"threadgroup {'half' if dtype == jnp.float16 else 'bfloat'} _As" in source
    want = np.maximum(np.asarray(a, np.float32) @ np.asarray(b, np.float32), 0.0)
    got = np.asarray(call(a, b), np.float32)
    # One rounding of the float32 result: half an ulp of the output type.
    ulp = {jnp.float16: 2.0**-10, jnp.bfloat16: 2.0**-7}[dtype]
    np.testing.assert_allclose(got, want, rtol=ulp, atol=1e-3)
