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
    """The Pallas kernel body for one query-tile program."""
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
    """The (grid, in_specs, out_specs) for [batch, sequence, heads, head_dim] arrays."""
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
    """Attention over [batch, sequence, heads, head_dim] Q (and K/V with
    `key_length`), built with `call`: `pl.pallas_call` or
    `palladium.metal_call`. `kwargs` pass through, e.g. `compiler_params`."""
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
    """Softmax attention in NumPy, 32 query rows at a time to bound memory."""
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
    import palladium

    shape = (1, 4096, 4, 64)
    rng = np.random.default_rng(0)
    q, k, v = (jnp.asarray(rng.standard_normal(shape, dtype=np.float32)) for _ in range(3))
    kernel = jax.jit(attention_call(shape, tile_q=16, tile_k=128, call=palladium.metal_call))
    dense = jax.jit(lambda q, k, v: jax.nn.dot_product_attention(q, k, v))
    expected = reference_attention(*map(np.asarray, (q, k, v)), causal=False)

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
