"""Dot lowering shape and accumulation contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import palladium
from palladium import EmitError

F32 = jnp.float32


def _shaped(*shape):
    return jax.ShapeDtypeStruct(shape, F32)


def _run_and_compare(kernel, out_shape, *arrays, tol=1e-5):
    call = palladium.metal_call(kernel, out_shape=out_shape)
    got = call(*arrays)
    want = call.interpret(*arrays)
    np.testing.assert_allclose(got, np.asarray(want), rtol=tol, atol=tol)


def test_batched_dot_general_rejected():
    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.einsum("bij,bjk->bik", a_ref[...], b_ref[...])

    with pytest.raises(EmitError, match="batch dims"):
        palladium.debug_msl(kernel, _shaped(2, 4, 4), _shaped(2, 4, 4), out_shape=_shaped(2, 4, 4))


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
    assert not isinstance(got, tuple)  # single out_shape: never a tuple
    got = np.asarray(got)
    assert got.dtype == np.float32
    want = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)


def _matmul_kernel(a_ref, b_ref, o_ref):
    o_ref[...] = jnp.dot(a_ref[...], b_ref[...])


def test_rectangular_matmul_matches_interpret(rng):
    a = rng.standard_normal((4, 6), dtype=np.float32)
    b = rng.standard_normal((6, 10), dtype=np.float32)
    f = palladium.metal_call(_matmul_kernel, out_shape=_shaped(4, 10))
    got = f(a, b)
    want = np.asarray(f.interpret(a, b))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(got, a @ b, rtol=1e-4, atol=1e-4)


def test_non_square_inner_dim_matches_numpy(rng):
    """Inner dim 1, a degenerate but legal matmul (outer product shape)."""
    a = rng.standard_normal((5, 1), dtype=np.float32)
    b = rng.standard_normal((1, 3), dtype=np.float32)
    f = palladium.metal_call(_matmul_kernel, out_shape=_shaped(5, 3))
    got = f(a, b)
    np.testing.assert_allclose(got, a @ b, rtol=1e-4, atol=1e-4)


def test_non_standard_contraction_is_rejected():
    """Contracting lhs dim 0 (a transposed-lhs matmul) is rejected as a non-standard contraction."""

    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jax.lax.dot_general(a_ref[...], b_ref[...], (((0,), (0,)), ((), ())))

    with pytest.raises(EmitError, match="standard matmul contraction"):
        palladium.debug_msl(kernel, _shaped(4, 4), _shaped(4, 4), out_shape=_shaped(4, 4))
