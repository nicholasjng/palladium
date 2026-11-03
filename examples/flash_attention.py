"""Online-softmax (flash) attention written in plain Pallas, with its gradient.

One program per (batch, head, query tile) streams the key/value blocks
through a running max and sum. Palladium recognizes this jaxpr and lowers
both matmuls to Metal 4 TensorOps or SIMD-group matrices; it can also store
each row's logsumexp for the backward pass.

The backward pass is two `plmtl` kernels, after FlashAttention-2: one
program per key tile accumulates dK and dV over the query tiles, and one per
query tile accumulates dQ over the key tiles. Both recompute the
probabilities from the logsumexp and skip fully masked tiles when causal.
`differentiable_attention` pairs them with the forward through
`palladium.with_vjp`. The benchmarks and tests import the kernels from here.

    uv run python examples/flash_attention.py
"""

from __future__ import annotations

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl


def _attention(q_ref, k_ref, v_ref, out_ref, *logsumexp_ref, tile_q, tile_k, head_dim, causal):
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
    running_max, running_sum, running_output = jax.lax.fori_loop(
        0, k_ref.shape[1] // tile_k, update, initial
    )
    out_ref[0, :, 0, :] = running_output / running_sum[:, None]
    for ref in logsumexp_ref:
        ref[0, 0, :] = running_max + jnp.log(running_sum)


def attention_kernel(*, tile_q: int, tile_k: int, head_dim: int, causal: bool = False):
    """Construct the Pallas body for one streaming query-tile attention program.

    Parameters
    ----------
    tile_q : int
        Query rows per program; cooperative emission requires a positive multiple of
        sixteen.
    tile_k : int
        Keys processed per scan step; key length must divide evenly into tiles and
        cooperative emission requires a multiple of sixteen,
        or of eight for the `simdgroup_matrix` schedule.
    head_dim : int
        Features per head; cooperative emission requires a positive multiple of sixteen,
        or of eight for the `simdgroup_matrix` schedule.
    causal : bool, optional
        Apply the absolute-position causal mask, default False. Cooperative causal
        attention requires matching sequence lengths.

    Returns
    -------
    callable
        Kernel accepting Q, K, V and output Refs in that order, and optionally a
        logsumexp Ref of shape (batch, heads, query_length) last.

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
        Features per head; cooperative emission requires a positive multiple of sixteen,
        or of eight for the `simdgroup_matrix` schedule.
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
    logsumexp: bool = False,
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
        cooperative emission requires a multiple of sixteen,
        or of eight for the `simdgroup_matrix` schedule.
    causal : bool, optional
        Apply the absolute-position causal mask, default False. Cooperative causal
        attention requires matching sequence lengths.
    key_length : int or None, optional
        Key/value sequence length; None uses the query length.
    logsumexp : bool, optional
        Also return each query row's logsumexp of the scaled scores, shape
        (batch, heads, query_length), as the backward pass needs.
    call : callable, optional
        Kernel factory, default pl.pallas_call; use palladium.metal_call for CPU FFI
        dispatch.
    **kwargs : dict
        Additional call-factory keywords, such as compiler_params or interpret for plain
        Pallas.

    Returns
    -------
    callable
        Function of Q, K, V returning an array with shape equal to the query shape,
        and with `logsumexp` also the logsumexp.

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
    out_shape = jax.ShapeDtypeStruct(shape, jnp.float32)
    if logsumexp:
        rows = pl.BlockSpec((1, 1, tile_q), lambda b, h, qb: (b, h, qb))
        out_specs = (out_specs, rows)
        out_shape = (out_shape, jax.ShapeDtypeStruct((batch, heads, query_length), jnp.float32))
    return call(
        attention_kernel(tile_q=tile_q, tile_k=tile_k, head_dim=head_dim, causal=causal),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        **kwargs,
    )


_NT = (((1,), (1,)), ((), ()))  # a @ b.T


def _dkdv(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, dk_ref, dv_ref, *tiles, **geometry):
    """dK and dV for one key tile, accumulated over the query tiles that see it."""
    k_tile, v_tile, q_tile, do_tile, p_tile = tiles
    rows, step, length, scale, causal = (
        geometry[name] for name in ("rows", "step", "length", "scale", "causal")
    )
    j, h, b = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    k_start = j * rows
    k_tile[...] = k_ref[b, pl.ds(k_start, rows), h, :]
    v_tile[...] = v_ref[b, pl.ds(k_start, rows), h, :]
    k_index = k_start + jnp.arange(rows)

    def query_tile(i, carry):
        dk, dv = carry
        q_start = i * step
        q_tile[...] = q_ref[b, pl.ds(q_start, step), h, :]
        do_tile[...] = do_ref[b, pl.ds(q_start, step), h, :]
        lse = lse_ref[b, h, pl.ds(q_start, step)]
        delta = delta_ref[b, h, pl.ds(q_start, step)]
        # Probabilities and their gradient, transposed: keys by queries.
        p = jnp.exp(lax.dot_general(k_tile[...], q_tile[...], _NT) * scale - lse[None, :])
        if causal:
            q_index = q_start + jnp.arange(step)
            p = jnp.where(k_index[:, None] <= q_index[None, :], p, 0.0)
        p_tile[...] = p
        dv = dv + jnp.dot(p_tile[...], do_tile[...])
        ds = p * (lax.dot_general(v_tile[...], do_tile[...], _NT) - delta[None, :]) * scale
        p_tile[...] = ds
        return dk + jnp.dot(p_tile[...], q_tile[...]), dv

    zeros = jnp.zeros(k_tile.shape, jnp.float32)
    first = k_start // step if causal else 0  # earlier query tiles see none of these keys
    dk, dv = lax.fori_loop(first, length // step, query_tile, (zeros, zeros))
    dk_ref[b, pl.ds(k_start, rows), h, :] = dk
    dv_ref[b, pl.ds(k_start, rows), h, :] = dv


def _dq(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, dq_ref, *tiles, **geometry):
    """dQ for one query tile, accumulated over the key tiles it sees."""
    q_tile, do_tile, k_tile, v_tile, s_tile = tiles
    rows, step, length, scale, causal = (
        geometry[name] for name in ("rows", "step", "length", "scale", "causal")
    )
    i, h, b = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    q_start = i * rows
    q_tile[...] = q_ref[b, pl.ds(q_start, rows), h, :]
    do_tile[...] = do_ref[b, pl.ds(q_start, rows), h, :]
    lse = lse_ref[b, h, pl.ds(q_start, rows)]
    delta = delta_ref[b, h, pl.ds(q_start, rows)]
    q_index = q_start + jnp.arange(rows)

    def key_tile(j, dq):
        k_start = j * step
        k_tile[...] = k_ref[b, pl.ds(k_start, step), h, :]
        v_tile[...] = v_ref[b, pl.ds(k_start, step), h, :]
        p = jnp.exp(lax.dot_general(q_tile[...], k_tile[...], _NT) * scale - lse[:, None])
        if causal:
            k_index = k_start + jnp.arange(step)
            p = jnp.where(k_index[None, :] <= q_index[:, None], p, 0.0)
        s_tile[...] = p * (lax.dot_general(do_tile[...], v_tile[...], _NT) - delta[:, None]) * scale
        return dq + jnp.dot(s_tile[...], k_tile[...])

    last = (q_start + rows + step - 1) // step if causal else length // step
    dq = lax.fori_loop(0, last, key_tile, jnp.zeros(q_tile.shape, jnp.float32))
    dq_ref[b, pl.ds(q_start, rows), h, :] = dq


def attention_backward_calls(
    shape: tuple[int, int, int, int], *, causal: bool = False, rows: int = 32, step: int = 16
):
    """Create the dK/dV and dQ kernels of attention over `shape`.

    Parameters
    ----------
    shape : tuple of int
        Q, K and V shape [batch, length, heads, head_dim]; query and key lengths
        match.
    causal : bool, optional
        Whether the forward pass applied the causal mask.
    rows : int, optional
        Rows of each program's own tile (keys for dK/dV, queries for dQ), a
        multiple of 16.
    step : int, optional
        Rows of the other operand per loop step, a multiple of 16. Each kernel
        stages four tiles of `rows` or `step` rows of head_dim floats, plus one
        of rows by step, in 32 KB of threadgroup memory.

    Returns
    -------
    dkdv : callable
        `dkdv(q, k, v, do, lse, delta)` returns (dK, dV); `lse` is the forward's
        logsumexp and `delta` the per-row sum of dO times O, both of shape
        (batch, heads, length).
    dq : callable
        `dq(q, k, v, do, lse, delta)` returns dQ.
    """
    from palladium import plmtl

    batch, length, heads, dim = shape
    if length % rows or length % step:
        raise ValueError(f"length {length} must be a multiple of {rows} and {step}")
    geometry = {"step": step, "length": length, "scale": dim**-0.5, "causal": causal}
    mesh = plmtl.Mesh((length // rows, heads, batch))
    f32 = jnp.float32

    def tiles(*sizes):
        return [plmtl.THREADGROUP(size, f32) for size in sizes]

    gradient = jax.ShapeDtypeStruct(shape, f32)
    dkdv = plmtl.kernel(
        functools.partial(_dkdv, rows=rows, **geometry),
        (gradient, gradient),
        mesh=mesh,
        scratch_types=tiles((rows, dim), (rows, dim), (step, dim), (step, dim), (rows, step)),
        name="attention_dkdv",
    )
    dq = plmtl.kernel(
        functools.partial(_dq, rows=rows, **geometry),
        gradient,
        mesh=mesh,
        scratch_types=tiles((rows, dim), (rows, dim), (step, dim), (step, dim), (rows, step)),
        name="attention_dq",
    )
    return dkdv, dq


def differentiable_attention(
    shape: tuple[int, int, int, int],
    *,
    causal: bool = False,
    tile_q: int = 64,
    tile_k: int = 16,
    call=pl.pallas_call,
    **kwargs,
):
    """Attention over `shape` whose gradient runs the backward kernels.

    Parameters
    ----------
    shape : tuple of int
        Q, K and V shape [batch, length, heads, head_dim].
    causal : bool, optional
        Apply the causal mask.
    tile_q, tile_k : int, optional
        Forward tiles, as for `attention_call`.
    call : callable, optional
        Kernel factory of the forward pass, as for `attention_call`.
    **kwargs : dict
        Further forward-call keywords.

    Returns
    -------
    callable
        `f(q, k, v)`, differentiable with `jax.grad` and `jax.vjp`.
    """
    import palladium

    forward = attention_call(
        shape, tile_q=tile_q, tile_k=tile_k, causal=causal, logsumexp=True, call=call, **kwargs
    )
    dkdv, dq = attention_backward_calls(shape, causal=causal)

    def run(q, k, v):
        out, lse = forward(q, k, v)
        return out, out, lse  # the output again, and the logsumexp, as residuals

    def backward(q, k, v, out, lse, d_out):
        delta = jnp.sum(d_out * out, axis=-1).transpose(0, 2, 1)  # (batch, heads, length)
        dk, dv = dkdv(q, k, v, d_out, lse, delta)
        return dq(q, k, v, d_out, lse, delta), dk, dv

    return palladium.with_vjp(run, backward, residuals=2)


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
