"""Online-softmax (flash) attention written in plain Pallas.

One program per (batch, head, query tile) streams the key/value blocks
through a running max and sum. Palladium recognizes this jaxpr and lowers
both matmuls to Metal 4 TensorOps. The benchmarks and tests import the
kernel from here.

    uv run python examples/flash_attention.py
"""

from __future__ import annotations

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl


def _attention(q_ref, k_ref, v_ref, out_ref, *, tile_q, tile_k, head_dim, causal):
    q_start = pl.program_id(2) * tile_q
    q = q_ref[0, :, 0, :]
    q_indices = q_start + jnp.arange(tile_q)
    scale = jnp.asarray(head_dim**-0.5, dtype=jnp.float32)

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

    initial = (
        jnp.full((tile_q,), -jnp.inf, dtype=jnp.float32),
        jnp.zeros((tile_q,), dtype=jnp.float32),
        jnp.zeros((tile_q, head_dim), dtype=jnp.float32),
    )
    _, running_sum, running_output = jax.lax.fori_loop(0, k_ref.shape[1] // tile_k, update, initial)
    out_ref[0, :, 0, :] = running_output / running_sum[:, None]


def attention_kernel(*, tile_q: int, tile_k: int, head_dim: int, causal: bool = False):
    """Construct the Pallas body for one streaming query-tile attention program.

    Parameters
    ----------
    tile_q : int
        Query rows per program; cooperative emission requires a positive multiple of
        sixteen.
    tile_k : int
        Keys processed per scan step; key length must divide evenly into tiles and
        cooperative emission requires a multiple of sixteen.
    head_dim : int
        Features per head; cooperative emission requires a positive multiple of sixteen.
    causal : bool, optional
        Apply the absolute-position causal mask, default False. Cooperative causal
        attention requires matching sequence lengths.

    Returns
    -------
    callable
        Kernel accepting Q, K, V, and output Refs in that order.

    Notes
    -----
    Q/output expose one query tile, while K/V expose the full sequence for one batch/head.
    This factory binds constants but does not validate tile divisibility; use matching
    attention_specs metadata.
    """
    return functools.partial(
        _attention, tile_q=tile_q, tile_k=tile_k, head_dim=head_dim, causal=causal
    )


def attention_specs(
    batch: int,
    query_length: int,
    heads: int,
    tile_q: int,
    head_dim: int,
    *,
    key_length: int | None = None,
):
    """Build standard grid and block maps for tiled rank-4 attention.

    Parameters
    ----------
    batch : int
        Number of batches.
    query_length : int
        Query sequence length, evenly divisible by tile_q.
    heads : int
        Number of heads.
    tile_q : int
        Query rows per program; cooperative emission requires a positive multiple of
        sixteen.
    head_dim : int
        Features per head; cooperative emission requires a positive multiple of sixteen.
    key_length : int or None, optional
        Key/value sequence length; None uses the query length.

    Returns
    -------
    grid : tuple of int
        Program grid in batch, head, query-tile order.
    in_specs : tuple of BlockSpec
        Q tile map and full-sequence K/V maps.
    out_specs : BlockSpec
        Output map matching the Q tile.

    Notes
    -----
    Arrays use [batch, sequence, heads, head_dim] order. The query-tile count uses integer
    division; callers must pad and crop sequences that do not divide evenly.
    """
    key_length = query_length if key_length is None else key_length
    query_spec = pl.BlockSpec((1, tile_q, 1, head_dim), lambda b, h, qb: (b, qb, h, 0))
    kv_spec = pl.BlockSpec((1, key_length, 1, head_dim), lambda b, h, qb: (b, 0, h, 0))
    return (batch, heads, query_length // tile_q), (query_spec, kv_spec, kv_spec), query_spec


def attention_call(
    shape: tuple[int, int, int, int],
    *,
    tile_q: int,
    tile_k: int,
    causal: bool = False,
    key_length: int | None = None,
    call=pl.pallas_call,
    **kwargs,
):
    """Create a float32 Pallas attention callable for a fixed query layout.

    Parameters
    ----------
    shape : tuple of int
        Query/output shape [batch, query_length, heads, head_dim].
    tile_q : int
        Query rows per program; cooperative emission requires a positive multiple of
        sixteen.
    tile_k : int
        Keys processed per scan step; key length must divide evenly into tiles and
        cooperative emission requires a multiple of sixteen.
    causal : bool, optional
        Apply the absolute-position causal mask, default False. Cooperative causal
        attention requires matching sequence lengths.
    key_length : int or None, optional
        Key/value sequence length; None uses the query length.
    call : callable, optional
        Kernel factory, default pl.pallas_call; use palladium.metal_call for CPU FFI
        dispatch.
    **kwargs : dict
        Additional call-factory keywords, such as compiler_params or interpret for plain
        Pallas.

    Returns
    -------
    callable
        Function of Q, K, V returning an array with shape equal to the query shape.

    Notes
    -----
    K/V may have a distinct key_length for noncausal attention. Both lengths must tile
    evenly; this wrapper does not pad inputs. Cooperative support is validated when
    tracing/emitting, not by this factory.
    """
    batch, query_length, heads, head_dim = shape
    grid, in_specs, out_specs = attention_specs(
        batch, query_length, heads, tile_q, head_dim, key_length=key_length
    )
    return call(
        attention_kernel(tile_q=tile_q, tile_k=tile_k, head_dim=head_dim, causal=causal),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        **kwargs,
    )


def reference_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool) -> np.ndarray:
    """Compute a stable NumPy attention reference in chunks of 32 query rows.

    Parameters
    ----------
    q : numpy.ndarray
        Queries of shape (batch, query_length, heads, head_dim).
    k : numpy.ndarray
        Keys of shape (batch, key_length, heads, head_dim).
    v : numpy.ndarray
        Values with the same shape as k.
    causal : bool
        Mask keys whose absolute position exceeds the query position.

    Returns
    -------
    numpy.ndarray
        Attention output with q shape and dtype.

    Notes
    -----
    Chunking bounds temporary score memory. Positive head dimension and nonempty compatible
    sequences are expected; this reference does not validate layouts or handle arbitrary
    padding masks.
    """
    output = np.empty_like(q)
    key_indices = np.arange(k.shape[1])
    for start in range(0, q.shape[1], 32):
        stop = min(start + 32, q.shape[1])
        scores = np.einsum("bshd,bthd->bhst", q[:, start:stop], k) * (q.shape[-1] ** -0.5)
        if causal:
            active = np.arange(start, stop)[:, None] >= key_indices[None, :]
            scores = np.where(active[None, None, :, :], scores, -np.inf)
        scores -= np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        output[:, start:stop] = np.einsum("bhst,bthd->bshd", weights, v)
    return output


def main() -> None:
    """Check and time the fixed attention example against NumPy and JAX attention."""
    import palladium

    shape = (1, 4096, 4, 64)
    rng = np.random.default_rng(0)
    q, k, v = (jnp.asarray(rng.standard_normal(shape, dtype=np.float32)) for _ in range(3))
    kernel = jax.jit(attention_call(shape, tile_q=16, tile_k=128, call=palladium.metal_call))
    dense = jax.jit(lambda q, k, v: jax.nn.dot_product_attention(q, k, v))
    expected = reference_attention(np.asarray(q), np.asarray(k), np.asarray(v), causal=False)

    for name, fn in (("palladium (Metal, TensorOps)", kernel), ("jax.nn on CPU", dense)):
        got = jax.block_until_ready(fn(q, k, v))
        start = time.perf_counter()
        for _ in range(10):
            jax.block_until_ready(fn(q, k, v))
        elapsed = (time.perf_counter() - start) / 10 * 1000
        error = float(np.max(np.abs(np.asarray(got) - expected)))
        print(f"{name:30s} {elapsed:8.2f} ms  max |error| {error:.1e}")


if __name__ == "__main__":
    main()
