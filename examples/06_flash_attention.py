"""Example 6: single-head scaled-dot-product attention, palladium
(online-softmax, streaming K/V) vs plain jax.jit (the idiomatic JAX way).

Splash Attention's whole reason for the online-softmax reformulation
(https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas)
is bandwidth: avoid materializing the full seq_len x seq_len score
matrix. For palladium it turned out to be load-bearing for a different
reason: one Metal thread per query, materializing a full seq_len-long
score row as thread-local storage, hits the same per-thread stack wall
example 06 (the deleted MLP training kernel) found. Streaming K/V in
fixed-size blocks, carrying a running (max, sum, output) online-softmax
accumulator, keeps per-thread memory at O(block_kv * head_dim),
independent of sequence length, verified during scoping: N=4096,
head_dim=64 (N*head_dim=262144) compiles and runs fine this way, versus
~2048 elements being the wall for full materialization.

Grid = seq_len, one thread per query; K and V are passed as full,
unblocked refs (every thread reads all of them), and the "blocking"
happens inside the kernel via dynamic ref slicing
(`k_ref[pl.dslice(j * BLOCK_KV, BLOCK_KV), :]`) inside a fori_loop,
verified against an independent NumPy implementation of the online
recurrence before writing any MSL-generating code, and against the
running kernel afterward (max abs diff ~2e-7 across several sequence
lengths).

Single head, single sequence (no batch/heads grid dims yet): batching
those in is a straightforward grid extension (no dot_general widening
needed, everything stays rank-2 per thread), not done here.

Honest result: correct, and genuinely parallel across the grid (unlike
the training kernel this replaced, which needed cross-thread gradient
reduction this repo doesn't build and so ran on a single thread), but
still roughly 15-20x *slower* than jax.jit here, and that ratio
doesn't close as N grows or as `BLOCK_KV` is tuned (checked 4 through
128; 128 hits the same stack-pressure wall again at N*head_dim scale).
The bottleneck isn't missing parallelism this time, it's that
`_rule_dot_general` is a naive scalar triple-nested loop (no
vectorization), competing against XLA's heavily vectorized, jit-compiled
CPU matmul. See ROADMAP stretch 17 for what closing that gap would need.
"""

import time

import numpy as np

SEQ_LEN = 1024
HEAD_DIM = 64
BLOCK_KV = 16
SEED = 0


def make_qkv(seq_len: int, head_dim: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((seq_len, head_dim), dtype=np.float32)
    k = rng.standard_normal((seq_len, head_dim), dtype=np.float32)
    v = rng.standard_normal((seq_len, head_dim), dtype=np.float32)
    return q, k, v


def golden_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def attn(q, k, v):
        scores = jnp.dot(q, k.T) * scale
        scores = scores - jnp.max(scores, axis=-1, keepdims=True)
        w = jnp.exp(scores)
        w = w / jnp.sum(w, axis=-1, keepdims=True)
        return jnp.dot(w, v)

    qj, kj, vj = jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)
    attn(qj, kj, vj).block_until_ready()  # compile outside the clock
    t0 = time.perf_counter()
    out = attn(qj, kj, vj)
    out.block_until_ready()
    dt = time.perf_counter() - t0
    return np.asarray(out), dt


def palladium_attention(seq_len: int, head_dim: int, block_kv: int):
    """Online-softmax attention: one thread per query, streaming K/V in
    `block_kv`-sized chunks, carrying (running max, running sum, running
    output) across the fori_loop, normalizing once at the end.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    import palladium

    scale = 1.0 / (head_dim**0.5)

    def kernel(q_ref, k_ref, v_ref, o_ref):
        q = q_ref[...]  # (1, head_dim)

        def step(j, carry):
            m, l, o = carry
            kj = k_ref[pl.dslice(j * block_kv, block_kv), :]
            vj = v_ref[pl.dslice(j * block_kv, block_kv), :]
            s = jnp.dot(q, kj.T) * scale  # (1, block_kv)
            m_block = jnp.max(s, axis=1, keepdims=True)
            m_new = jnp.maximum(m, m_block)
            correction = jnp.exp(m - m_new)
            p = jnp.exp(s - m_new)  # (1, block_kv)
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


def main():
    q, k, v = make_qkv(SEQ_LEN, HEAD_DIM)
    scale = 1.0 / (HEAD_DIM**0.5)
    print(f"Attention: seq_len={SEQ_LEN}, head_dim={HEAD_DIM}, block_kv={BLOCK_KV}")

    want, t_golden = golden_attention(q, k, v, scale)
    print(f"jax.jit (naive softmax, CPU):          {t_golden * 1e3:8.3f} ms")

    attn = palladium_attention(SEQ_LEN, HEAD_DIM, BLOCK_KV)
    attn(q, k, v)  # trace + emit + Metal compile outside the clock
    t0 = time.perf_counter()
    got = attn(q, k, v)
    t_metal = time.perf_counter() - t0
    print(
        f"palladium (Apple GPU, streaming K/V):  {t_metal * 1e3:8.3f} ms "
        f"({t_golden / t_metal:.3f}x)"
    )

    err = np.abs(np.asarray(got) - want).max()
    print(f"max abs deviation from jax.jit: {err:.2e}")
    print(
        "not faster here: dot_general is a naive scalar loop, competing "
        "against XLA's vectorized CPU matmul (see module docstring)"
    )


if __name__ == "__main__":
    main()
