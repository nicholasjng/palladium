"""Pallas-authored online-softmax attention prototype."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

HEAD_DIM = 64
TILE_Q = 32
TILE_K = 64


def _attention_kernel(q_ref, k_ref, v_ref, out_ref, *, tile_q: int, tile_k: int, causal: bool):
    """Compute one query tile with the streaming softmax recurrence."""
    query_block = pl.program_id(2)
    q_start = query_block * tile_q
    q = q_ref[0, :, 0, :]
    q_indices = q_start + jnp.arange(tile_q)
    scale = jnp.asarray(HEAD_DIM**-0.5, dtype=jnp.float32)

    running_max = jnp.full((tile_q,), -jnp.inf, dtype=jnp.float32)
    running_sum = jnp.zeros((tile_q,), dtype=jnp.float32)
    running_output = jnp.zeros((tile_q, HEAD_DIM), dtype=jnp.float32)
    num_key_blocks = k_ref.shape[1] // tile_k

    def update(key_block, state):
        running_max, running_sum, running_output = state
        k_start = key_block * tile_k
        k = k_ref[0, pl.dslice(k_start, tile_k), 0, :]
        v = v_ref[0, pl.dslice(k_start, tile_k), 0, :]

        scores = (q @ k.T) * scale
        if causal:
            key_indices = k_start + jnp.arange(tile_k)
            active = key_indices[None, :] <= q_indices[:, None]
            scores = jnp.where(active, scores, -jnp.inf)
        else:
            active = jnp.ones((tile_q, tile_k), dtype=jnp.bool_)

        block_max = jnp.max(scores, axis=1)
        new_max = jnp.maximum(running_max, block_max)
        old_scale = jnp.where(running_sum > 0.0, jnp.exp(running_max - new_max), 0.0)
        probabilities = jnp.where(active, jnp.exp(scores - new_max[:, None]), 0.0)
        new_sum = old_scale * running_sum + jnp.sum(probabilities, axis=1)
        new_output = old_scale[:, None] * running_output + probabilities @ v
        return new_max, new_sum, new_output

    running_max, running_sum, running_output = jax.lax.fori_loop(
        0,
        num_key_blocks,
        update,
        (running_max, running_sum, running_output),
    )
    out_ref[0, :, 0, :] = running_output / running_sum[:, None]


def attention_kernel(*, tile_q: int = TILE_Q, tile_k: int = TILE_K, causal: bool = False):
    """Return the Pallas kernel body for one query-tile program."""

    def kernel(q_ref, k_ref, v_ref, out_ref):
        _attention_kernel(
            q_ref,
            k_ref,
            v_ref,
            out_ref,
            tile_q=tile_q,
            tile_k=tile_k,
            causal=causal,
        )

    return kernel


def attention_specs(batch: int, sequence_length: int, heads: int, tile_q: int):
    """Return the attention Pallas grid and block mappings."""
    query_spec = pl.BlockSpec(
        (1, tile_q, 1, HEAD_DIM),
        lambda b, h, qb: (b, qb, h, 0),
    )
    full_kv_spec = pl.BlockSpec(
        (1, sequence_length, 1, HEAD_DIM),
        lambda b, h, qb: (b, 0, h, 0),
    )
    return (
        (batch, heads, sequence_length // tile_q),
        (query_spec, full_kv_spec, full_kv_spec),
        query_spec,
    )


def make_pallas_flash_attention(
    shape: tuple[int, int, int, int],
    *,
    tile_q: int = TILE_Q,
    tile_k: int = TILE_K,
    causal: bool = False,
    interpret: bool = False,
):
    """Build a Pallas call for one program per batch, head, and query tile.

    The full K/V refs are sliced by the kernel's static loop. This gives the
    Pallas jaxpr the streaming algorithm while keeping this first version a
    correctness prototype for the supported TensorOps attention lowering.
    """
    if len(shape) != 4:
        raise ValueError("shape must be [batch, sequence, heads, D]")
    batch, sequence_length, heads, head_dim = shape
    if head_dim != HEAD_DIM:
        raise ValueError(f"head dimension must be {HEAD_DIM}")
    if tile_q < 1 or tile_k < 1:
        raise ValueError("tile sizes must be positive")
    if sequence_length < tile_q or sequence_length % tile_q or sequence_length % tile_k:
        raise ValueError("sequence length must be divisible by tile_q and tile_k")

    grid, in_specs, out_specs = attention_specs(batch, sequence_length, heads, tile_q)
    return pl.pallas_call(
        attention_kernel(tile_q=tile_q, tile_k=tile_k, causal=causal),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        interpret=interpret,
    )


def pallas_flash_attention(
    q,
    k,
    v,
    *,
    tile_q: int = TILE_Q,
    tile_k: int = TILE_K,
    causal: bool = False,
    interpret: bool = False,
):
    """Run the Pallas online-softmax prototype on same-shaped rank-4 inputs."""
    if q.shape != k.shape or q.shape != v.shape or len(q.shape) != 4:
        raise ValueError("q, k, and v must have the same [batch, sequence, heads, D] shape")
    if q.dtype != jnp.float32 or k.dtype != jnp.float32 or v.dtype != jnp.float32:
        raise ValueError("attention prototype currently supports float32 inputs")
    return make_pallas_flash_attention(
        q.shape, tile_q=tile_q, tile_k=tile_k, causal=causal, interpret=interpret
    )(q, k, v)


def reference_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool) -> np.ndarray:
    """Compute chunked attention with NumPy for correctness checks."""
    output = np.empty_like(q)
    key_indices = np.arange(k.shape[1])
    for start in range(0, q.shape[1], TILE_Q):
        stop = min(start + TILE_Q, q.shape[1])
        scores = np.einsum("bshd,bthd->bhst", q[:, start:stop], k) * (HEAD_DIM**-0.5)
        if causal:
            query_indices = np.arange(start, stop)
            active = query_indices[:, None] >= key_indices[None, :]
            scores = np.where(active[None, None, :, :], scores, -np.inf)
        scores -= np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        output[:, start:stop] = np.einsum("bhst,bthd->bshd", weights, v)
    return output
