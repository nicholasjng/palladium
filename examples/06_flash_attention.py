"""Example 6: single-head scaled-dot-product attention, palladium
(online-softmax, streaming K/V) vs plain jax.jit (the idiomatic JAX way).

Docs: companion to docs/performance.md (the cooperative/MMA path).

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

Honest result: correct, and — after the emitter work this example drove
(see `docs/notes/reward-spec-matmul-emitter.md` for the scoring methodology
and history) — **faster than `jax.jit` CPU at every measured shape**:
2.8x at seq 1024 (0.49ms vs 1.36ms wall) and 3.8x at seq 4096 (3.8ms vs
14.2ms), block_kv=32, measured heat-soaked. The GPU advantage grows with
seq_len, which is what reward spec v0.3.0 scores (worst-case across
shapes, normalized to the ~4x practical ceiling; current score 0.693).
Five emitter changes closed the original ~17x gap and pushed well past
parity, each verified against the `.interpret` oracle and `jax.jit` at
three seeds:

1. Ref slices became pointer views instead of element-wise thread-local
   copies (the two streamed K/V blocks alone were 2048 scalar device
   loads plus ~12KB of stack per fori_loop iteration).
2. `kj.T` fuses into `dot_general` (a lazy `transposed` view), making
   QK^T a unit-stride float4 row-dot on both operands.
3. Kernels like this one lower to a simdgroup-cooperative execution
   model (`emit.is_simdgroup_cooperative`): one 32-thread SIMD-group per
   program instance, with lane-sharded values and simd_sum/max
   reductions — 32x the threads at seq_len=1024, which was ~2x
   under-occupied per the scaling measurement.
4. The cooperative model generalized from one query row per instance to
   R rows (`block_q` below; columns-per-lane layout), so each K/V
   element loaded from device serves R rows — the query-blocking lever
   `docs/notes/query-blocking-scratch.md` measured at 2x on hand-written
   stand-ins, transferring to this generated kernel within ~11%.
5. Both cooperative dots lower to the SIMD-group matrix units
   (`simdgroup_float8x8`) when tile-divisible, with the softmax
   probabilities staged through a threadgroup tile — 1.9x more device
   time at the worst shape (0.468ms -> 0.242ms at seq 1024).

Thermal caveat, measured: the GPU down-clocks up to ~3x under sustained
dispatch load (the effect `docs/notes/simdgroup-matmul-design.md` documented)
while the CPU side barely moves. The current ~1.4x margin absorbs most
of it; deep heat-soak can still push individual readings below parity
without any code change.
"""

import time

import numpy as np

SEQ_LEN = 1024
HEAD_DIM = 64
BLOCK_KV = 32
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


def palladium_attention(seq_len: int, head_dim: int, block_kv: int, block_q: int = 8):
    """Online-softmax attention, blocked over queries *and* keys: each
    program instance owns `block_q` query rows, streaming K/V in
    `block_kv`-sized chunks, carrying (running max, running sum, running
    output) per row across the fori_loop, normalizing once at the end.

    `block_q` is the R of the emitter's cooperative model: with one
    32-lane SIMD-group per instance, every K/V element loaded from device
    serves `block_q` rows instead of one. 8 measured fastest here, the
    same knee `docs/notes/query-blocking-scratch.md` found on hand-written
    stand-ins (reuse and SIMD-group count trade off inversely; 8 is a
    property of seq_len=1024, not a constant of the algorithm).
    `block_q=1` reproduces the previous one-row-per-instance kernel.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    import palladium

    scale = 1.0 / (head_dim**0.5)

    def kernel(q_ref, k_ref, v_ref, o_ref):
        q = q_ref[...]  # (block_q, head_dim)

        def step(j, carry):
            m, l, o = carry
            kj = k_ref[pl.dslice(j * block_kv, block_kv), :]
            vj = v_ref[pl.dslice(j * block_kv, block_kv), :]
            s = jnp.dot(q, kj.T) * scale  # (block_q, block_kv)
            m_block = jnp.max(s, axis=1, keepdims=True)
            m_new = jnp.maximum(m, m_block)
            correction = jnp.exp(m - m_new)
            p = jnp.exp(s - m_new)  # (block_q, block_kv)
            l_new = l * correction + jnp.sum(p, axis=1, keepdims=True)
            o_new = o * correction + jnp.dot(p, vj)
            return (m_new, l_new, o_new)

        m0 = jnp.full((block_q, 1), -jnp.inf, jnp.float32)
        l0 = jnp.zeros((block_q, 1), jnp.float32)
        o0 = jnp.zeros((block_q, head_dim), jnp.float32)
        _, l, o = jax.lax.fori_loop(0, seq_len // block_kv, step, (m0, l0, o0))
        o_ref[...] = o / l

    return palladium.metal_call(
        kernel,
        grid=(seq_len // block_q,),
        in_specs=[
            pl.BlockSpec((block_q, head_dim), lambda i: (i, 0)),
            pl.BlockSpec((seq_len, head_dim), lambda i: (0, 0)),
            pl.BlockSpec((seq_len, head_dim), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((block_q, head_dim), lambda i: (i, 0)),
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
        "single-shot timing above is illustrative only; see "
        "benchmarks/reward_matmul_emitter.py for the soaked-median "
        "methodology (and the module docstring for the thermal caveat)"
    )


if __name__ == "__main__":
    main()
