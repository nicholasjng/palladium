"""The specialized multi-SIMD-group GEMM lowering (emit/gemm.py):
pattern gate, launch geometry, and oracle correctness across shapes,
rhs layouts, and geometry fallbacks.
"""

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
import pytest

import palladium
from palladium.emit.gemm import gemm_desc, gemm_groups


def _make_mmt(m, k, n, transpose=True):
    def kernel(a_ref, b_ref, o_ref):
        b = b_ref[...]
        o_ref[...] = jnp.dot(a_ref[...], b.T if transpose else b)

    bshape = (n, k) if transpose else (k, n)
    return palladium.metal_call_jit(
        kernel,
        grid=(m // 8,),
        in_specs=[
            pl.BlockSpec((8, k), lambda i: (i, 0)),
            pl.BlockSpec(bshape, lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((8, n), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
    )


@pytest.mark.parametrize(
    "m,k,n,transpose",
    [
        (512, 256, 256, True),
        (512, 256, 256, False),
        (256, 512, 256, True),
        (512, 1024, 1024, True),  # wide n, only expressible via this path
        (512, 96, 64, True),  # k % 64 != 0: falls to the 32-deep slab
    ],
)
def test_gemm_matches_oracle(rng, m, k, n, transpose):
    f = _make_mmt(m, k, n, transpose)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((n, k) if transpose else (k, n)).astype(np.float32)
    got = np.asarray(f(a, b))
    want = a @ (b.T if transpose else b)
    np.testing.assert_allclose(got, want, rtol=3e-4, atol=3e-4)


def test_gemm_geometry_is_consistent_everywhere():
    # emit_msl bakes the instance index from the same desc that
    # dispatch/ffi/explain size the threadgroup from; a mismatch here
    # would compute wrong instances silently.
    f = _make_mmt(512, 256, 256)
    args = (
        jax.ShapeDtypeStruct((512, 256), jnp.float32),
        jax.ShapeDtypeStruct((256, 256), jnp.float32),
    )
    diag = f.explain(*args)
    spec = palladium.trace(f._staged, *args)
    desc = gemm_desc(spec)
    assert desc is not None
    assert diag.threadgroup == (32 * desc.groups, 1, 1)
    assert gemm_groups(spec) == desc.groups
    assert f"_tgid.x * {desc.groups}u" in palladium.emit_msl(spec, cooperative=True)


def test_gemm_gate_rejects_non_matching_kernels():
    # An epilogue breaks the pattern; the kernel must keep the general
    # cooperative lowering (one SIMD-group per threadgroup).
    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.tanh(jnp.dot(a_ref[...], b_ref[...].T))

    f = palladium.metal_call_jit(
        kernel,
        grid=(64,),
        in_specs=[
            pl.BlockSpec((8, 256), lambda i: (i, 0)),
            pl.BlockSpec((256, 256), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((8, 256), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((512, 256), jnp.float32),
    )
    diag = f.explain(
        jax.ShapeDtypeStruct((512, 256), jnp.float32),
        jax.ShapeDtypeStruct((256, 256), jnp.float32),
    )
    assert diag.model == "cooperative"
    assert diag.threadgroup == (32, 1, 1)


def test_gemm_small_grid_falls_back_correctly(rng):
    # grid extent 3 fits no group configuration: general lowering, still
    # correct.
    f = _make_mmt(24, 64, 64)
    a = rng.standard_normal((24, 64)).astype(np.float32)
    b = rng.standard_normal((64, 64)).astype(np.float32)
    diag = f.explain(a, b)
    assert diag.threadgroup == (32, 1, 1)
    np.testing.assert_allclose(np.asarray(f(a, b)), a @ b.T, rtol=3e-4, atol=3e-4)
