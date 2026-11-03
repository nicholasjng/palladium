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
import math
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np


def _revision() -> dict:
    root = Path(__file__).resolve().parents[1]

    def read(*args):
        try:
            return subprocess.check_output(
                args, cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = read("git", "status", "--porcelain")
    return {
        "git_commit": read("git", "rev-parse", "HEAD"),
        "jj_commit": read(
            "jj", "log", "--ignore-working-copy", "-r", "@", "--no-graph", "-T", "commit_id"
        ),
        "working_copy_dirty": None if status is None else bool(status),
    }


def _samples(batch, repeats: int, minimum_calls: int, seconds: float):
    # Calibrate outside the samples; no clock reads inside the dispatch loop.
    count = minimum_calls
    while True:
        start = time.perf_counter()
        batch(count)
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break
        count = max(count + 1, math.ceil(count * min(4, seconds / max(elapsed, 1e-9)) * 1.1))
    return [batch(count) / count for _ in range(repeats)], count


def _elementwise(
    device, label: str, repeats: int, calls: int, mib: int, sample_seconds: float
) -> dict:
    size = 1024 if label == "small" else mib * (1 << 20) // 4
    host = np.full(size, 0.25, np.float32)
    x = jax.device_put(host, device)
    run = jax.jit(lambda x: jnp.tanh(2 * x + 1) * x).lower(x).compile()
    for _ in range(3):
        y = run(x)
        y.block_until_ready()
    np.testing.assert_allclose(np.asarray(y), np.tanh(2 * host + 1) * host, rtol=2e-5)

    def batch(count):
        start = time.perf_counter()
        for _ in range(count):
            y = run(x)
            if label == "large":
                y.block_until_ready()  # bound memory use for 256 MiB outputs
        y.block_until_ready()
        return time.perf_counter() - start

    samples, count = _samples(batch, repeats, calls if label == "small" else 1, sample_seconds)
    result = {f"{label}_seconds_per_call": samples, f"{label}_calls_per_sample": count}
    if label == "small":
        result["small_median_us"] = statistics.median(samples) * 1e6
    else:
        result["elementwise_median_gbps"] = 2 * host.nbytes / statistics.median(samples) / 1e9
    return result


def _copy(repeats: int, mib: int, sample_seconds: float) -> dict:
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

    def batch(count):
        return sum(_runtime.run(kernel, buffers, nbytes // 16, 256) for _ in range(count))

    samples, count = _samples(batch, repeats, 1, sample_seconds)
    return {
        "copy_gpu_seconds": samples,
        "copy_calls_per_sample": count,
        "copy_median_gbps": 2 * nbytes / statistics.median(samples) / 1e9,
    }


def _sbi(device) -> dict:
    # Initialize on CPU so unsupported setup operations cannot obscure the step.
    from training_steps import workload

    with jax.default_device(jax.devices("cpu")[0]):
        fn, *args = workload("sbi")
    args = jax.device_put(args, device)
    jax.block_until_ready(args)
    start = time.perf_counter()
    output = jax.jit(fn)(*args)
    jax.block_until_ready(output)
    return {
        "sbi_compile_and_first_call_seconds": time.perf_counter() - start,
        "sbi_first_loss": float(output[1]),
    }


def _record_section(result: dict, name: str, run: Callable[[], dict], checkpoint) -> None:
    result["sections"][name] = {"status": "running"}
    checkpoint(result)
    try:
        result.update(run())
    except Exception as exc:  # noqa: BLE001 - persist failed sections and continue the others
        result["sections"][name] = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    else:
        result["sections"][name] = {"status": "ok"}
    checkpoint(result)


def measure(
    backend: str,
    repeats: int,
    calls: int,
    mib: int,
    sample_seconds: float = 0.2,
    checkpoint: Callable[[dict], None] = lambda _: None,
) -> dict:
    jax.config.update("jax_enable_compilation_cache", False)
    result = {
        "schema_version": 2,
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "platform": backend,
        "os": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "revision": _revision(),
        "cache": "fresh process; JAX persistent cache disabled; Metal driver cache unchanged",
        "repeats": repeats,
        "small_min_calls_per_sample": calls,
        "minimum_sample_wall_seconds": sample_seconds,
        "large_mib_per_buffer": mib,
        "sections": {},
    }
    checkpoint(result)
    devices = []

    def discover():
        devices.append(jax.devices(backend)[0])
        return {"device": devices[0].device_kind}

    _record_section(result, "device", discover, checkpoint)
    if devices:
        for label in ("small", "large"):
            _record_section(
                result,
                label,
                lambda label=label: _elementwise(
                    devices[0], label, repeats, calls, mib, sample_seconds
                ),
                checkpoint,
            )
        if backend == "metal":
            _record_section(result, "copy", lambda: _copy(repeats, mib, sample_seconds), checkpoint)
            if "elementwise_median_gbps" in result and "copy_median_gbps" in result:
                result["elementwise_fraction_of_copy"] = (
                    result["elementwise_median_gbps"] / result["copy_median_gbps"]
                )
                checkpoint(result)
        _record_section(result, "sbi", lambda: _sbi(devices[0]), checkpoint)
    result["complete"] = all(s["status"] == "ok" for s in result["sections"].values())
    checkpoint(result)
    return result


def _write_json(path: Path, result: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("metal", "mps", "cpu"), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--mib", type=int, default=256)
    parser.add_argument("--sample-seconds", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.repeats, args.calls, args.mib, args.sample_seconds) <= 0 or not math.isfinite(
        args.sample_seconds
    ):
        parser.error("repeats, calls, mib and sample-seconds must be positive and finite")
    checkpoint = (
        (lambda result: _write_json(args.output, result)) if args.output else lambda _: None
    )
    result = measure(
        args.platform, args.repeats, args.calls, args.mib, args.sample_seconds, checkpoint
    )
    if not args.output:
        print(json.dumps(result, indent=2, allow_nan=False))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
