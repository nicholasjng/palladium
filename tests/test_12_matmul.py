"""`dot_general` baseline: the scalar triple-nested-loop matmul.

Scoped to exactly the plain matmul contraction, rank-2 operands, no
batch dims: verified against a real jaxpr before implementing (see
`_rule_dot_general`'s docstring). Batch dims and higher rank are
unimplemented, not silently wrong; each has its own rejection test.

No threadgroup-memory tiling: this is the baseline O(M*N*K) path.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium
from palladium.emit import EmitError


def _matmul_kernel(a_ref, b_ref, o_ref):
    o_ref[...] = jnp.dot(a_ref[...], b_ref[...])


def test_square_matmul_matches_numpy(rng):
    a = rng.standard_normal((8, 8), dtype=np.float32)
    b = rng.standard_normal((8, 8), dtype=np.float32)
    f = palladium.metal_call(
        _matmul_kernel, out_shape=jax.ShapeDtypeStruct((8, 8), jnp.float32)
    )
    got = f(a, b)
    np.testing.assert_allclose(got, a @ b, rtol=1e-4, atol=1e-4)


def test_rectangular_matmul_matches_interpret(rng):
    a = rng.standard_normal((4, 6), dtype=np.float32)
    b = rng.standard_normal((6, 10), dtype=np.float32)
    f = palladium.metal_call(
        _matmul_kernel, out_shape=jax.ShapeDtypeStruct((4, 10), jnp.float32)
    )
    got = f(a, b)
    want = np.asarray(f.interpret(a, b))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(got, a @ b, rtol=1e-4, atol=1e-4)


def test_non_square_inner_dim_matches_numpy(rng):
    """Inner dim 1, a degenerate but legal matmul (outer product shape)."""
    a = rng.standard_normal((5, 1), dtype=np.float32)
    b = rng.standard_normal((1, 3), dtype=np.float32)
    f = palladium.metal_call(
        _matmul_kernel, out_shape=jax.ShapeDtypeStruct((5, 3), jnp.float32)
    )
    got = f(a, b)
    np.testing.assert_allclose(got, a @ b, rtol=1e-4, atol=1e-4)


def test_batched_matmul_is_rejected():
    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jax.lax.dot_general(
            a_ref[...], b_ref[...], (((2,), (1,)), ((0,), (0,)))
        )

    f = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((2, 4, 4), jnp.float32))
    spec = palladium.trace(
        f,
        jax.ShapeDtypeStruct((2, 4, 4), jnp.float32),
        jax.ShapeDtypeStruct((2, 4, 4), jnp.float32),
    )
    with pytest.raises(EmitError, match="batch dims"):
        palladium.emit_msl(spec)


def test_non_standard_contraction_is_rejected():
    """Contracting lhs dim 0 (a transposed-lhs matmul): not the plain
    `A @ B` pattern this rule implements."""

    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jax.lax.dot_general(
            a_ref[...], b_ref[...], (((0,), (0,)), ((), ()))
        )

    f = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((4, 4), jnp.float32))
    spec = palladium.trace(
        f,
        jax.ShapeDtypeStruct((4, 4), jnp.float32),
        jax.ShapeDtypeStruct((4, 4), jnp.float32),
    )
    with pytest.raises(EmitError, match="standard matmul contraction"):
        palladium.emit_msl(spec)
