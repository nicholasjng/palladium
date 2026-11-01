"""Reproducible M1 measurements; run in a fresh process for each backend.

    JAX_PLATFORMS=metal,cpu uv run python examples/pjrt_m1_benchmark.py --platform metal
    JAX_PLATFORMS=mps,cpu uv run python examples/pjrt_m1_benchmark.py --platform mps

Use the jax-mps environment for the second command. JSON goes to stdout or
--output; compare results from the same machine. Metal additionally measures
the copy kernel from bench_roofline.bench_copy_bandwidth. Compilation caching
in JAX is disabled; the Metal driver's disk cache is left intact.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np


def measure(backend: str, repeats: int, calls: int, mib: int) -> dict:
    jax.config.update("jax_enable_compilation_cache", False)
    device = jax.devices(backend)[0]
    result = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "platform": backend,
        "device": device.device_kind,
        "os": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "cache": "fresh process; JAX persistent cache disabled; Metal driver cache unchanged",
        "repeats": repeats,
        "small_calls_per_batch": calls,
        "large_mib_per_buffer": mib,
    }
    for label, size, batch in (("small", 1024, calls), ("large", mib * (1 << 20) // 4, 1)):
        host = np.full(size, 0.25, np.float32)
        x = jax.device_put(host, device)
        run = jax.jit(lambda x: jnp.tanh(2 * x + 1) * x).lower(x).compile()
        for _ in range(3):
            y = run(x)
            y.block_until_ready()
        np.testing.assert_allclose(np.asarray(y), np.tanh(2 * host + 1) * host, rtol=2e-5)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(batch):
                y = run(x)
            y.block_until_ready()
            samples.append((time.perf_counter() - start) / batch)
        result[f"{label}_seconds_per_call"] = samples
        if label == "small":
            result["small_median_us"] = statistics.median(samples) * 1e6
        else:
            result["elementwise_median_gbps"] = 2 * host.nbytes / statistics.median(samples) / 1e9
        del x, y, host, run

    if backend == "metal":
        from palladium import _runtime

        kernel = _runtime.Kernel(
            """
#include <metal_stdlib>
using namespace metal;
kernel void copy4(device const float4* a [[buffer(0)]], device float4* b [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) { b[i] = a[i]; }
""",
            "copy4",
        )
        nbytes = mib * (1 << 20)
        buffers = [np.ones(nbytes // 4, np.float32), np.zeros(nbytes // 4, np.float32)]
        for _ in range(3):
            _runtime.run(kernel, buffers, nbytes // 16, 256)
        samples = [_runtime.run(kernel, buffers, nbytes // 16, 256) for _ in range(repeats)]
        result["copy_gpu_seconds"] = samples
        result["copy_median_gbps"] = 2 * nbytes / statistics.median(samples) / 1e9
        result["elementwise_fraction_of_copy"] = (
            result["elementwise_median_gbps"] / result["copy_median_gbps"]
        )
        del buffers, kernel

    # Initialize on CPU so unsupported setup operations cannot obscure the step.
    from training_steps import steps

    with jax.default_device(jax.devices("cpu")[0]):
        fn, *args = steps()["sbi"]
    args = jax.device_put(args, device)
    jax.block_until_ready(args)
    start = time.perf_counter()
    output = jax.jit(fn)(*args)
    jax.block_until_ready(output)
    result["sbi_compile_and_first_call_seconds"] = time.perf_counter() - start
    result["sbi_first_loss"] = float(output[1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("metal", "mps", "cpu"), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--mib", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.repeats, args.calls, args.mib) <= 0:
        parser.error("repeats, calls and mib must be positive")
    result = measure(args.platform, args.repeats, args.calls, args.mib)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
