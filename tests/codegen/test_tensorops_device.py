"""Cooperative TensorOps kernels on the device: matmul tails, edge tiles, half
precision, attention tiles small enough that MPP leaves slots unowned, and
attention with `simdgroup_matrix`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flash_attention import (
    attention_call,
    attention_kernel,
    attention_specs,
    differentiable_attention,
    reference_attention,
)
from jax.experimental import pallas as pl

import palladium

TM = 16


def _tiles(extent: int, tile: int) -> int:
    return (extent + tile - 1) // tile


@pytest.mark.parametrize(
    "m, n, k, tn, dtype, tol",
    [
        (30, 45, 144, 32, jnp.float32, 3e-3),  # K tail and output edge tiles
        (32, 64, 256, 32, jnp.float16, 2e-2),  # K loop
        (32, 64, 144, 32, jnp.bfloat16, 1e-1),  # K tail
        (32, 32, 16, 16, jnp.float32, 3e-3),  # 16x16 output tiles
    ],
)
def test_matmul(rng, m, n, k, tn, dtype, tol):
    """Compare cooperative matmul with JAX across K tails, output edge tiles, and
    float32/half/bfloat buffers."""

    def kernel(a_ref, b_ref, out_ref):
        out_ref[...] = jnp.matmul(a_ref[...], b_ref[...])

    call = palladium.metal_call(
        kernel,
        grid=(_tiles(m, TM), _tiles(n, tn)),
        in_specs=[
            pl.BlockSpec((TM, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, tn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((TM, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        compiler_params=palladium.CompilerParams(dot_general="tensorops"),
    )
    a = jnp.asarray(rng.standard_normal((m, k), dtype=np.float32), dtype)
    b = jnp.asarray(rng.standard_normal((k, n), dtype=np.float32), dtype)
    np.testing.assert_allclose(
        np.asarray(call(a, b), np.float32),
        np.asarray(jnp.matmul(a, b), np.float32),
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize("transposed", ["lhs", "rhs", "both"])
@pytest.mark.parametrize(
    "m, n, k",
    [
        (32, 64, 16),
        (32, 64, 200),  # K tail after a full chunk
        (32, 64, 40),  # K below one chunk and not a multiple of 16
        (30, 45, 144),  # output edge tiles and a K tail
    ],
)
def test_matmul_with_transposed_operands(rng, m, n, k, transposed):
    """Compare `matmul2d` products of operands stored transposed with NumPy, including K
    tails and edge tiles, where the views' extents bound what the product reads."""
    tn = 32
    lhs_t, rhs_t = transposed in ("lhs", "both"), transposed in ("rhs", "both")

    def kernel(a_ref, b_ref, out_ref):
        a, b = a_ref[...], b_ref[...]
        out_ref[...] = jnp.matmul(a.T if lhs_t else a, b.T if rhs_t else b)

    call = palladium.metal_call(
        kernel,
        grid=(_tiles(m, TM), _tiles(n, tn)),
        in_specs=[
            pl.BlockSpec((k, TM), lambda i, j: (0, i))
            if lhs_t
            else pl.BlockSpec((TM, k), lambda i, j: (i, 0)),
            pl.BlockSpec((tn, k), lambda i, j: (j, 0))
            if rhs_t
            else pl.BlockSpec((k, tn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((TM, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        compiler_params=palladium.CompilerParams(dot_general="tensorops"),
    )
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    got = call(np.ascontiguousarray(a.T) if lhs_t else a, np.ascontiguousarray(b.T) if rhs_t else b)
    np.testing.assert_allclose(got, a @ b, rtol=3e-3, atol=3e-3)


def _tanh_epilogue(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.tanh(jnp.matmul(a_ref[...], b_ref[...]))


def _shift_epilogue(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.matmul(a_ref[...], b_ref[...]) + 1.0


def _residual_epilogue(a_ref, b_ref, residual_ref, out_ref):
    out_ref[...] = jnp.tanh(jnp.matmul(a_ref[...], b_ref[...]) + residual_ref[...])


def _bias_epilogue(a_ref, b_ref, bias_ref, out_ref):
    out_ref[...] = jnp.maximum(jnp.matmul(a_ref[...], b_ref[...]) + bias_ref[...], 0.0) * 2.0


_EPILOGUE_PATHS = {
    # dot_general, (m, n, k)
    "matmul2d": ("tensorops", (32, 64, 32)),
    "matmul2d edge tiles and K tail": ("tensorops", (30, 45, 40)),
    "simdgroup_matrix": ("simdgroup", (32, 64, 32)),
}


_EPILOGUES = {
    # kernel: (the shape of its third operand, the result from product and operand)
    _tanh_epilogue: (None, lambda product, _: np.tanh(product)),
    _shift_epilogue: (None, lambda product, _: product + 1.0),
    _residual_epilogue: ("tile", lambda product, residual: np.tanh(product + residual)),
    _bias_epilogue: ("column", lambda product, bias: np.maximum(product + bias, 0.0) * 2.0),
}


@pytest.mark.parametrize("path", _EPILOGUE_PATHS)
@pytest.mark.parametrize("kernel", _EPILOGUES)
@pytest.mark.parametrize(
    "dtype, tol", [(jnp.float32, 1e-5), (jnp.float16, 8e-3), (jnp.bfloat16, 6e-2)]
)
def test_matmul_epilogue(rng, dtype, tol, kernel, path):
    """Compare epilogues with NumPy for every dtype and product path: non-idempotent ones,
    which do not match if applied twice or to an unnarrowed product, and ones that read
    another tile or a column bias."""
    dot_general, (m, n, k) = _EPILOGUE_PATHS[path]
    tn = 32
    block = pl.BlockSpec((TM, tn), lambda i, j: (i, j))
    operand, expected = _EPILOGUES[kernel]
    extra = {
        None: ((), ()),
        "tile": ((block,), ((m, n),)),
        "column": ((pl.BlockSpec((tn,), lambda i, j: (j,)),), ((n,),)),
    }[operand]
    call = palladium.metal_call(
        kernel,
        grid=(_tiles(m, TM), _tiles(n, tn)),
        in_specs=[
            pl.BlockSpec((TM, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, tn), lambda i, j: (0, j)),
            *extra[0],
        ],
        out_specs=block,
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        compiler_params=palladium.CompilerParams(dot_general=dot_general),
    )
    a = jnp.asarray(rng.standard_normal((m, k), dtype=np.float32) * 0.2, dtype)
    b = jnp.asarray(rng.standard_normal((k, n), dtype=np.float32) * 0.2, dtype)
    others = [
        jnp.asarray(rng.standard_normal(shape, dtype=np.float32), dtype) for shape in extra[1]
    ]
    _, msl, _ = call._spec_and_msl((a, b, *others))
    assert ("simdgroup_mma" in msl) == (path == "simdgroup_matrix")
    got = call(a, b, *others)
    assert got.dtype == dtype
    a, b, *others = (np.asarray(x, np.float32) for x in (a, b, *others))
    np.testing.assert_allclose(
        np.asarray(got, np.float32), expected(a @ b, *others or [None]), atol=tol
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim, tile_q, tile_k", [(16, 16, 16), (16, 16, 128), (32, 32, 32)])
def test_attention(rng, head_dim, tile_q, tile_k, causal):
    """Compare causal and noncausal cooperative attention with NumPy across small tiles and head
    dimensions."""
    shape = (1, 256, 2, head_dim)
    grid, in_specs, out_specs = attention_specs(1, 256, 2, tile_q, head_dim)
    call = palladium.metal_call(
        attention_kernel(tile_q=tile_q, tile_k=tile_k, head_dim=head_dim, causal=causal),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        compiler_params=palladium.CompilerParams(dot_general="tensorops"),
    )
    q, k, v = (rng.standard_normal(shape, dtype=np.float32) for _ in range(3))
    np.testing.assert_allclose(
        call(q, k, v), reference_attention(q, k, v, causal=causal), rtol=3e-4, atol=3e-4
    )


@pytest.mark.parametrize("dot_general", ["tensorops", "simdgroup"])
@pytest.mark.parametrize("operand, value", [(0, np.nan), (1, np.nan), (0, np.inf), (1, -np.inf)])
def test_attention_propagates_nonfinite_inputs(rng, operand, value, dot_general):
    """Check that a NaN or infinity in Q or K makes the same outputs NaN as in the Pallas
    interpreter, although the running maximum uses Metal's `max`, which drops NaNs."""
    shape = (1, 64, 2, 32)
    call = attention_call(
        shape,
        tile_q=32,
        tile_k=32,
        call=palladium.metal_call,
        compiler_params=palladium.CompilerParams(
            dot_general=dot_general, math_mode=palladium.MathMode.SAFE
        ),
    )
    inputs = [rng.standard_normal(shape, dtype=np.float32) for _ in range(3)]
    inputs[operand][0, 40, 1, 2] = value
    got, want = np.asarray(call(*inputs)), np.asarray(call.interpret(*inputs))
    assert np.isnan(want).any()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))


_SIMDGROUP_ATTENTION = [
    (64, 64, 16, 256, True),
    (16, 32, 16, 256, True),
    (48, 32, 64, 256, True),
    (64, 128, 32, 256, True),  # four fragment rows per SIMD group
    (56, 32, 8, 256, True),  # SAM2's head width and 8-key tiles: no padding
    (64, 64, 16, 128, True),  # cross attention with fewer keys (noncausal only)
    (32, 32, 128, 256, False),  # K and V tiles too large to stage: matmul2d
    (64, 16, 16, 256, False),  # query tiles below 32 rows: matmul2d
]


@pytest.mark.parametrize(
    "head_dim, tile_q, tile_k, key_length, simdgroup, causal",
    [
        (*case, causal)
        for case in _SIMDGROUP_ATTENTION
        for causal in (False, True)
        if not causal or case[3] == 256
    ],
)
def test_attention_with_simdgroup_matrix(
    rng, head_dim, tile_q, tile_k, key_length, simdgroup, causal
):
    """Compare attention under `dot_general="simdgroup"` with NumPy over two batches
    and three heads, and check which schedule the tiles select."""
    shape = (2, 256, 3, head_dim)
    kv_shape = (2, key_length, 3, head_dim)
    tiles = {"tile_q": tile_q, "tile_k": tile_k, "causal": causal, "key_length": key_length}
    avals = (jax.ShapeDtypeStruct(shape, jnp.float32),) + (
        jax.ShapeDtypeStruct(kv_shape, jnp.float32),
    ) * 2
    spec = palladium.trace(attention_call(shape, **tiles), *avals)
    assert ("simdgroup_mma" in palladium.emit_msl(spec, dot_general="simdgroup")) == simdgroup
    call = attention_call(
        shape,
        **tiles,
        call=palladium.metal_call,
        compiler_params=palladium.CompilerParams(dot_general="simdgroup"),
    )
    q = rng.standard_normal(shape, dtype=np.float32)
    k, v = (rng.standard_normal(kv_shape, dtype=np.float32) for _ in range(2))
    np.testing.assert_allclose(
        call(q, k, v), reference_attention(q, k, v, causal=causal), rtol=3e-4, atol=3e-4
    )


@pytest.mark.parametrize("causal", [False, True])
def test_attention_logsumexp(rng, causal):
    """The forward's second output is each row's logsumexp of the scaled scores."""
    shape = (1, 128, 2, 64)
    call = attention_call(
        shape, tile_q=32, tile_k=16, causal=causal, logsumexp=True, call=palladium.metal_call
    )
    q, k, v = (rng.standard_normal(shape, dtype=np.float32) for _ in range(3))
    out, lse = call(q, k, v)
    scores = np.einsum("bqhd,bkhd->bhqk", q, k).astype(np.float64) / 8.0
    if causal:
        scores = np.where(np.tril(np.ones((128, 128), bool)), scores, -np.inf)
    peak = scores.max(-1, keepdims=True)
    want = (peak + np.log(np.exp(scores - peak).sum(-1, keepdims=True)))[..., 0]
    np.testing.assert_allclose(lse, want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out, reference_attention(q, k, v, causal), rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("causal", [False, True])
def test_attention_gradients(rng, causal):
    """The backward kernels' dQ, dK and dV match JAX's attention VJP on CPU."""
    shape = (1, 128, 2, 64)
    q, k, v, d_out = (rng.standard_normal(shape, dtype=np.float32) for _ in range(4))
    f = differentiable_attention(shape, causal=causal, tile_q=32, call=palladium.metal_call)
    _, vjp = jax.vjp(f, q, k, v)
    with jax.default_device(jax.devices("cpu")[0]):
        _, reference = jax.vjp(
            lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=causal), q, k, v
        )
        want = reference(d_out)
    for got, expected in zip(vjp(d_out), want, strict=True):
        got, expected = np.asarray(got, np.float64), np.asarray(expected, np.float64)
        assert np.abs(got - expected).max() <= 1e-5 * np.abs(expected).max()
