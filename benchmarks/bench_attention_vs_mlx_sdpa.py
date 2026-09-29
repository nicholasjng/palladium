"""Palladium's Pallas attention against MLX's fused SDPA, both under jax.jit on MPS.

jax-mps routes ``jax.nn.dot_product_attention`` to MLX's fused
scaled-dot-product kernel, so this is the bar Palladium's cooperative
attention has to clear to be worth using from JAX. Both run as ordinary
jitted JAX functions on the ``mps`` device, on identical resident arrays,
interleaved (A, B, A, B) to spread thermal drift, synchronized per sample.
A plain ``jnp`` softmax attention is timed too, to show what XLA fusion
alone gets before either fused kernel.

Run on a Metal 4 machine with the jax-mps ``palladium-dispatch`` handler:

    JAX_PLATFORMS=mps,cpu uv run python benchmarks/bench_attention_vs_mlx_sdpa.py

``--smoke`` runs tiny shapes on whatever platform is selected, with the
Pallas interpreter standing in for the Metal kernel, to check the script.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

import palladium
from palladium.workloads.pallas_flash_attention import (
    attention_kernel,
    attention_specs,
    reference_attention,
)

BATCH = 1
CASES = [
    # (sequence, heads, head_dim, tile_q, tile_k, causal)
    (1024, 4, 64, 32, 32, False),
    (2048, 4, 64, 32, 32, False),
    (4096, 4, 64, 32, 32, False),
    (4096, 4, 64, 32, 32, True),
    (4096, 4, 64, 32, 64, False),
    # SAM2 Hiera global-attention block at 1024 px input: 4096 tokens, dim 384.
    (4096, 8, 48, 32, 32, False),
]
SMOKE_CASES = [(128, 2, 16, 16, 16, False), (128, 2, 16, 16, 16, True)]


def palladium_attention(sequence, heads, head_dim, tile_q, tile_k, causal, *, fallback):
    grid, in_specs, out_specs = attention_specs(BATCH, sequence, heads, tile_q, head_dim)
    return palladium.mps_call_jit(
        attention_kernel(tile_q=tile_q, tile_k=tile_k, head_dim=head_dim, causal=causal),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=jax.ShapeDtypeStruct((BATCH, sequence, heads, head_dim), jnp.float32),
        dot_general="tensorops",
        fallback=fallback,
    )


def plain_attention(q, k, v, causal):
    scores = jnp.einsum("bshd,bthd->bhst", q, k) * (q.shape[-1] ** -0.5)
    if causal:
        mask = jnp.tril(jnp.ones((q.shape[1], k.shape[1]), dtype=bool))
        scores = jnp.where(mask, scores, -jnp.inf)
    return jnp.einsum("bhst,bthd->bshd", jax.nn.softmax(scores, axis=-1), v)


def timed(fn, *args, warmup=3, samples=15):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - start) * 1e3)
    return times


def run_case(sequence, heads, head_dim, tile_q, tile_k, causal, *, device, fallback, samples):
    rng = np.random.default_rng(0)
    shape = (BATCH, sequence, heads, head_dim)
    q_np, k_np, v_np = (rng.standard_normal(shape, dtype=np.float32) for _ in range(3))
    expected = reference_attention(q_np, k_np, v_np, causal)

    candidates = {
        "palladium": jax.jit(
            palladium_attention(
                sequence, heads, head_dim, tile_q, tile_k, causal, fallback=fallback
            )
        ),
        "mlx-sdpa": jax.jit(
            lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=causal)
        ),
        "jnp-softmax": jax.jit(lambda q, k, v: plain_attention(q, k, v, causal)),
    }
    with jax.default_device(device):
        q, k, v = (jnp.asarray(x) for x in (q_np, k_np, v_np))
        for name, fn in candidates.items():
            got = np.asarray(jax.block_until_ready(fn(q, k, v)))
            np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3, err_msg=name)
        # Interleave: one sample from each candidate per round.
        times = {name: [] for name in candidates}
        for name, fn in candidates.items():
            timed(fn, q, k, v, warmup=3, samples=0)
        for _ in range(samples):
            for name, fn in candidates.items():
                times[name].extend(timed(fn, q, k, v, warmup=0, samples=1))
    medians = {name: statistics.median(ts) for name, ts in times.items()}
    label = f"S={sequence} H={heads} D={head_dim} tq{tile_q}-tk{tile_k} {'causal' if causal else 'full'}"
    print(f"{label:44s}", end="")
    for name in candidates:
        print(f"  {name} {medians[name]:8.3f} ms", end="")
    print(f"  palladium/mlx {medians['palladium'] / medians['mlx-sdpa']:.2f}x")
    return {
        "sequence": sequence,
        "heads": heads,
        "head_dim": head_dim,
        "tile_q": tile_q,
        "tile_k": tile_k,
        "causal": causal,
        "median_ms": medians,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", action="store_true", help="tiny shapes, any platform")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--out", default="attention_vs_mlx_sdpa.json")
    args = parser.parse_args()

    if args.smoke:
        device = jax.devices()[0]
        cases, fallback = SMOKE_CASES, "interpret"
    else:
        device = jax.devices("mps")[0]
        cases, fallback = CASES, "error"
    print(f"host {platform.machine()} {platform.platform()}, jax {jax.__version__}, {device}")
    results = [
        run_case(*case, device=device, fallback=fallback, samples=args.samples) for case in cases
    ]
    with open(args.out, "w") as f:
        json.dump(
            {"host": platform.platform(), "jax": jax.__version__, "results": results}, f, indent=1
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
