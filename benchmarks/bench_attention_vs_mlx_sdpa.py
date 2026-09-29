"""Palladium's Pallas attention against MLX's fused SDPA, both under jax.jit on MPS.

jax-mps routes ``jax.nn.dot_product_attention`` to MLX's fused
scaled-dot-product kernel, so this is the bar Palladium's cooperative
attention has to clear to be worth using from JAX. Each case runs one
candidate as an ordinary jitted JAX function on the ``mps`` device with
resident inputs: Palladium via ``mps_call_jit``, MLX's fused SDPA, and a
plain ``jnp`` softmax attention to show what XLA fusion alone gets. Every
candidate is checked against the NumPy reference before timing.

Run on a Metal 4 machine with the jax-mps ``palladium-dispatch`` handler:

    JAX_PLATFORMS=mps,cpu uv run mew run --random-interleaving \\
        benchmarks/bench_attention_vs_mlx_sdpa.py

``--random-interleaving`` spreads thermal drift across the candidates.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import mew
import numpy as np

from palladium.workloads.pallas_flash_attention import (
    attention_kernel,
    attention_specs,
    reference_attention,
)

BATCH = 1
CANDIDATES = ("palladium", "mlx-sdpa", "jnp-softmax")
SHAPES = [
    # (sequence, heads, head_dim, tile_q, tile_k, causal)
    (1024, 4, 64, 32, 32, False),
    (2048, 4, 64, 32, 32, False),
    (4096, 4, 64, 32, 32, False),
    (4096, 4, 64, 32, 32, True),
    (4096, 4, 64, 32, 64, False),
    # SAM2 Hiera global-attention block at 1024 px input: 4096 tokens, dim 384.
    (4096, 8, 48, 32, 32, False),
]
CASES = [
    {
        "candidate": candidate,
        "sequence": sequence,
        "heads": heads,
        "head_dim": head_dim,
        "tile_q": tile_q,
        "tile_k": tile_k,
        "causal": causal,
    }
    for sequence, heads, head_dim, tile_q, tile_k, causal in SHAPES
    for candidate in CANDIDATES
]
IDS = [
    f"{case['candidate']}-s{case['sequence']}-h{case['heads']}-d{case['head_dim']}"
    f"-tq{case['tile_q']}-tk{case['tile_k']}-{'causal' if case['causal'] else 'full'}"
    for case in CASES
]


@functools.cache
def _inputs(sequence: int, heads: int, head_dim: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(0)
    shape = (BATCH, sequence, heads, head_dim)
    return tuple(rng.standard_normal(shape, dtype=np.float32) for _ in range(3))


@functools.cache
def _expected(sequence: int, heads: int, head_dim: int, causal: bool) -> np.ndarray:
    return reference_attention(*_inputs(sequence, heads, head_dim), causal=causal)


def _palladium(sequence, heads, head_dim, tile_q, tile_k, causal):
    import palladium

    grid, in_specs, out_specs = attention_specs(BATCH, sequence, heads, tile_q, head_dim)
    return palladium.mps_call_jit(
        attention_kernel(tile_q=tile_q, tile_k=tile_k, head_dim=head_dim, causal=causal),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=jax.ShapeDtypeStruct((BATCH, sequence, heads, head_dim), jnp.float32),
        dot_general="tensorops",
        fallback="error",
    )


def _jnp_softmax(q, k, v, causal):
    scores = jnp.einsum("bshd,bthd->bhst", q, k) * (q.shape[-1] ** -0.5)
    if causal:
        mask = jnp.tril(jnp.ones((q.shape[1], k.shape[1]), dtype=bool))
        scores = jnp.where(mask, scores, -jnp.inf)
    return jnp.einsum("bhst,bthd->bshd", jax.nn.softmax(scores, axis=-1), v)


def _candidate(name, sequence, heads, head_dim, tile_q, tile_k, causal):
    if name == "palladium":
        return _palladium(sequence, heads, head_dim, tile_q, tile_k, causal)
    if name == "mlx-sdpa":
        return lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=causal)
    return lambda q, k, v: _jnp_softmax(q, k, v, causal)


@mew.parametrize(CASES, ids=IDS, tags="attention-vs-mlx-sdpa", use_real_time=True, unit="ms")
def bench_attention_vs_mlx_sdpa(
    state: mew.State,
    candidate: str,
    sequence: int,
    heads: int,
    head_dim: int,
    tile_q: int,
    tile_k: int,
    causal: bool,
) -> None:
    try:
        device = jax.devices("mps")[0]
    except (RuntimeError, IndexError):
        state.skip_with_error("requires the jax-mps plugin and the mps platform")
        return

    fn = jax.jit(_candidate(candidate, sequence, heads, head_dim, tile_q, tile_k, causal))
    with jax.default_device(device):
        q, k, v = (jnp.asarray(x) for x in _inputs(sequence, heads, head_dim))
        got = np.asarray(jax.block_until_ready(fn(q, k, v)))
        np.testing.assert_allclose(
            got, _expected(sequence, heads, head_dim, causal), rtol=2e-3, atol=2e-3
        )
        for _ in range(3):
            jax.block_until_ready(fn(q, k, v))
        for _ in state:
            jax.block_until_ready(fn(q, k, v))
