"""Stretch 17: single-head attention, matching examples/06_flash_attention.py.

Online-softmax, streaming K/V in fixed-size blocks via dynamic ref
slicing inside a fori_loop, carrying a (running max, running sum,
running output) accumulator. Verified against an independent NumPy
implementation of the online-softmax recurrence (not just the interpret
oracle, which shares the kernel's own jaxpr) before writing any
MSL-generating code; see `examples/06_flash_attention.py`'s module
docstring for the full story, including why this replaced the deleted
MLP training kernel example.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium

pytestmark = pytest.mark.exercise


def _attention_kernel(seq_len: int, head_dim: int, block_kv: int):
    scale = 1.0 / (head_dim**0.5)

    def kernel(q_ref, k_ref, v_ref, o_ref):
        q = q_ref[...]

        def step(j, carry):
            m, l, o = carry
            kj = k_ref[pl.dslice(j * block_kv, block_kv), :]
            vj = v_ref[pl.dslice(j * block_kv, block_kv), :]
            s = jnp.dot(q, kj.T) * scale
            m_block = jnp.max(s, axis=1, keepdims=True)
            m_new = jnp.maximum(m, m_block)
            correction = jnp.exp(m - m_new)
            p = jnp.exp(s - m_new)
            l_new = l * correction + jnp.sum(p, axis=1, keepdims=True)
            o_new = o * correction + jnp.dot(p, vj)
            return (m_new, l_new, o_new)

        m0 = jnp.full((1, 1), -jnp.inf, jnp.float32)
        l0 = jnp.zeros((1, 1), jnp.float32)
        o0 = jnp.zeros((1, head_dim), jnp.float32)
        m, l, o = jax.lax.fori_loop(0, seq_len // block_kv, step, (m0, l0, o0))
        o_ref[...] = o / l

    return palladium.metal_call(
        kernel,
        grid=(seq_len,),
        in_specs=[
            pl.BlockSpec((1, head_dim), lambda i: (i, 0)),
            pl.BlockSpec((seq_len, head_dim), lambda i: (0, 0)),
            pl.BlockSpec((seq_len, head_dim), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((1, head_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((seq_len, head_dim), jnp.float32),
    )


def _naive_numpy_attention(q, k, v, scale):
    scores = (q @ k.T) * scale
    scores = scores - scores.max(axis=1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=1, keepdims=True)
    return w @ v


def test_matches_naive_numpy_softmax(rng):
    seq_len, head_dim, block_kv = 64, 8, 16
    q = rng.standard_normal((seq_len, head_dim), dtype=np.float32)
    k = rng.standard_normal((seq_len, head_dim), dtype=np.float32)
    v = rng.standard_normal((seq_len, head_dim), dtype=np.float32)

    f = _attention_kernel(seq_len, head_dim, block_kv)
    got = f(q, k, v)
    want = _naive_numpy_attention(q, k, v, 1.0 / (head_dim**0.5))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_single_kv_block_matches_naive_numpy(rng):
    """block_kv == seq_len: the fori_loop runs exactly once, exercising
    the degenerate "no actual streaming" case."""
    seq_len, head_dim, block_kv = 32, 8, 32
    q = rng.standard_normal((seq_len, head_dim), dtype=np.float32)
    k = rng.standard_normal((seq_len, head_dim), dtype=np.float32)
    v = rng.standard_normal((seq_len, head_dim), dtype=np.float32)

    f = _attention_kernel(seq_len, head_dim, block_kv)
    got = f(q, k, v)
    want = _naive_numpy_attention(q, k, v, 1.0 / (head_dim**0.5))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_large_magnitude_scores_stay_stable(rng):
    """The actual motivating case for reduce_max: without subtracting
    the running max before exp, this would overflow."""
    seq_len, head_dim, block_kv = 64, 8, 16
    q = rng.standard_normal((seq_len, head_dim), dtype=np.float32) * 20.0
    k = rng.standard_normal((seq_len, head_dim), dtype=np.float32) * 20.0
    v = rng.standard_normal((seq_len, head_dim), dtype=np.float32)

    f = _attention_kernel(seq_len, head_dim, block_kv)
    got = f(q, k, v)
    want = _naive_numpy_attention(q, k, v, 1.0 / (head_dim**0.5))
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-4)


def test_streaming_avoids_the_full_materialization_stack_limit():
    """The actual point of blocking K/V: N*head_dim well beyond the
    ~2048-element wall that broke full materialization (see the deleted
    MLP training kernel's finding) compiles and runs fine here, because
    per-thread memory is O(block_kv * head_dim), not O(seq_len * head_dim)."""
    seq_len, head_dim, block_kv = 512, 64, 16  # seq_len * head_dim = 32768
    rng = np.random.default_rng(0)
    q = rng.standard_normal((seq_len, head_dim), dtype=np.float32)
    k = rng.standard_normal((seq_len, head_dim), dtype=np.float32)
    v = rng.standard_normal((seq_len, head_dim), dtype=np.float32)

    f = _attention_kernel(seq_len, head_dim, block_kv)
    got = f(q, k, v)
    want = _naive_numpy_attention(q, k, v, 1.0 / (head_dim**0.5))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
