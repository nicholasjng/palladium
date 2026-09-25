"""Compare Palladium's dot lowering with the TensorOps GEMM lowering.

Run on a Mac with a Metal 4-capable OS using ``python examples/pallas_tensorops_dot.py``.
The TensorOps kernel assigns one output tile to each threadgroup.
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import metal_runtime as mr
import numpy as np

import palladium

TILE_M, TILE_N = 16, 32


def pallas_dot(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.matmul(a_ref[...], b_ref[...])


def timed(call, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if min(args.m, args.n, args.k) < 1:
        parser.error("matrix dimensions must be positive")
    if args.batch < 1:
        parser.error("batch must be positive")
    if args.m % TILE_M or args.n % TILE_N:
        parser.error(f"M must be divisible by {TILE_M} and N by {TILE_N}")
    batch, m, n, k = args.batch, args.m, args.n, args.k

    try:
        device = mr.device_info()
    except mr.DeviceError as error:
        raise SystemExit(f"No Metal device available: {error}") from error

    rng = np.random.default_rng(17)
    a = rng.standard_normal((batch, m, k), dtype=np.float32)
    b = rng.standard_normal((batch, k, n), dtype=np.float32)
    want = np.matmul(a, b)

    call_kwargs = {
        "grid": (batch, m // TILE_M, n // TILE_N),
        "in_specs": [
            pl.BlockSpec((1, TILE_M, k), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((1, k, TILE_N), lambda b, i, j: (b, 0, j)),
        ],
        "out_specs": pl.BlockSpec((1, TILE_M, TILE_N), lambda b, i, j: (b, i, j)),
        "out_shape": jax.ShapeDtypeStruct((batch, m, n), jnp.float32),
        "math_mode": mr.MathMode.SAFE,
    }
    eager = palladium.metal_call(
        pallas_dot,
        **call_kwargs,
    )
    got_palladium = eager(a, b)
    np.testing.assert_allclose(got_palladium, want, rtol=2e-4, atol=2e-4)
    pinned_palladium = eager.pin(a, b)

    tensorops = palladium.metal_call(pallas_dot, dot_general="tensorops", **call_kwargs)
    got_tensorops = tensorops(a, b)
    np.testing.assert_allclose(got_tensorops, want, rtol=2e-4, atol=2e-4)
    pinned_tensorops = tensorops.pin(a, b)

    for _ in range(3):
        pinned_palladium()
        pinned_tensorops()
    palladium_upload_samples = timed(lambda: eager(a, b), args.repeats)
    palladium_pinned_samples = timed(pinned_palladium, args.repeats)
    tensorops_pinned_samples = timed(pinned_tensorops, args.repeats)
    tensorops_upload_samples = timed(lambda: tensorops(a, b), args.repeats)

    print(f"Metal device: {device['name']}")
    print(
        f"TensorOps dot: batch={batch}, {m}x{k} @ {k}x{n}, "
        f"{TILE_M}x{TILE_N} tiles, {batch}x{m // TILE_M}x{n // TILE_N} threadgroups"
    )
    print(f"Palladium pinned median: {statistics.median(palladium_pinned_samples):.3f} ms")
    print(f"Palladium NumPy-upload median: {statistics.median(palladium_upload_samples):.3f} ms")
    print(f"TensorOps pinned median: {statistics.median(tensorops_pinned_samples):.3f} ms")
    print(f"TensorOps NumPy-upload median: {statistics.median(tensorops_upload_samples):.3f} ms")
    print("Both outputs match NumPy within rtol=2e-4, atol=2e-4.")


if __name__ == "__main__":
    main()
