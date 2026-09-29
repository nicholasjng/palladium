"""Pallas-authored online-softmax attention prototype."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

HEAD_DIM = 64
TILE_Q = 32
TILE_K = 64


def _attention_kernel(
    q_ref, k_ref, v_ref, out_ref, *, tile_q: int, tile_k: int, head_dim: int, causal: bool
):
    """Compute one query tile with the streaming softmax recurrence."""
    query_block = pl.program_id(2)
    q_start = query_block * tile_q
    q = q_ref[0, :, 0, :]
    q_indices = q_start + jnp.arange(tile_q)
    scale = jnp.asarray(head_dim**-0.5, dtype=jnp.float32)

    running_max = jnp.full((tile_q,), -jnp.inf, dtype=jnp.float32)
    running_sum = jnp.zeros((tile_q,), dtype=jnp.float32)
    running_output = jnp.zeros((tile_q, head_dim), dtype=jnp.float32)
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


def attention_kernel(
    *,
    tile_q: int = TILE_Q,
    tile_k: int = TILE_K,
    head_dim: int = HEAD_DIM,
    causal: bool = False,
):
    """Return the Pallas kernel body for one query-tile program."""

    def kernel(q_ref, k_ref, v_ref, out_ref):
        _attention_kernel(
            q_ref,
            k_ref,
            v_ref,
            out_ref,
            tile_q=tile_q,
            tile_k=tile_k,
            head_dim=head_dim,
            causal=causal,
        )

    return kernel


def attention_specs(
    batch: int,
    query_length: int,
    heads: int,
    tile_q: int,
    head_dim: int = HEAD_DIM,
    *,
    key_length: int | None = None,
):
    """Return the attention Pallas grid and block mappings."""
    key_length = query_length if key_length is None else key_length
    query_spec = pl.BlockSpec(
        (1, tile_q, 1, head_dim),
        lambda b, h, qb: (b, qb, h, 0),
    )
    full_kv_spec = pl.BlockSpec(
        (1, key_length, 1, head_dim),
        lambda b, h, qb: (b, 0, h, 0),
    )
    return (
        (batch, heads, query_length // tile_q),
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
    key_length: int | None = None,
):
    """Build a Pallas call for one program per batch, head, and query tile.

    The kernel slices full K/V refs inside its key loop; the resulting jaxpr
    is the form the TensorOps attention lowering recognizes.
    """
    if len(shape) != 4:
        raise ValueError("shape must be [batch, sequence, heads, D]")
    batch, query_length, heads, head_dim = shape
    key_length = query_length if key_length is None else key_length
    if head_dim < 1:
        raise ValueError("head dimension must be positive")
    if tile_q < 1 or tile_k < 1:
        raise ValueError("tile sizes must be positive")
    if query_length < tile_q or query_length % tile_q or key_length < tile_k or key_length % tile_k:
        raise ValueError("query and key lengths must be divisible by their tile sizes")
    if causal and query_length != key_length:
        raise ValueError("causal attention currently requires equal query and key lengths")

    grid, in_specs, out_specs = attention_specs(
        batch, query_length, heads, tile_q, head_dim, key_length=key_length
    )
    return pl.pallas_call(
        attention_kernel(tile_q=tile_q, tile_k=tile_k, head_dim=head_dim, causal=causal),
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
    """Run streaming attention on rank-4 Q, K, and V arrays."""
    if (
        len(q.shape) != 4
        or len(k.shape) != 4
        or len(v.shape) != 4
        or k.shape != v.shape
        or (q.shape[0], q.shape[2:]) != (k.shape[0], k.shape[2:])
    ):
        raise ValueError("q, k, and v must have compatible [batch, sequence, heads, D] shapes")
    if q.dtype != jnp.float32 or k.dtype != jnp.float32 or v.dtype != jnp.float32:
        raise ValueError("attention prototype currently supports float32 inputs")
    return make_pallas_flash_attention(
        q.shape,
        tile_q=tile_q,
        tile_k=tile_k,
        causal=causal,
        interpret=interpret,
        key_length=k.shape[1],
    )(q, k, v)


def reference_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool) -> np.ndarray:
    """Compute attention with NumPy for correctness checks."""
    output = np.empty_like(q)
    key_indices = np.arange(k.shape[1])
    for start in range(0, q.shape[1], TILE_Q):
        stop = min(start + TILE_Q, q.shape[1])
        scores = np.einsum("bshd,bthd->bhst", q[:, start:stop], k) * (q.shape[-1] ** -0.5)
        if causal:
            query_indices = np.arange(start, stop)
            active = query_indices[:, None] >= key_indices[None, :]
            scores = np.where(active[None, None, :, :], scores, -np.inf)
        scores -= np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        output[:, start:stop] = np.einsum("bhst,bthd->bshd", weights, v)
    return output
