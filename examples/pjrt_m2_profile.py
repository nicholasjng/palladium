"""Where an SBI step's wall time goes on the `metal` plugin (M2's GPU-time target).

    JAX_PLATFORMS=metal,cpu uv run python examples/pjrt_m2_profile.py [--output m2.json]

For the full step and its two halves (the 200-step RK4 simulation; the MDN
loss, gradients and AdamW), and for the CNF update of `cnf_density.py` (256
points, 16 RK4 steps; Palladium's adjoint kernels through `pallas_call`, or
plain JAX), runs calls pipelined, as training does, and splits the time per
step into:

- wall: elapsed time per step over a batch, synchronizing only at its end;
- host: time to return from each call, minus waits for the in-flight bound,
  i.e. jax's dispatch plus the plugin's encoding;
- encode: the plugin's binding, encoding and commit inside Execute;
- gpu: the union of command-buffer GPU intervals (GPU timestamps);
- gpu_fraction: gpu / wall, which M2 requires to be most of the wall time.

Each figure is the median of five batches, each calibrated to at least
--sample-seconds. Timings are of fixed resident inputs, as in the M1 benchmark.
"""

from __future__ import annotations

import argparse
import datetime
import math
import platform
import statistics
import time
from pathlib import Path

import cnf_density
import jax
import jaxlib
import numpy as np
from pjrt_m1_benchmark import _revision, _write_json
from training_steps import adamw, sbi_loss, sbi_trajectory, steps

from palladium import _pjrt


def _workloads(device) -> dict:
    with jax.default_device(jax.devices("cpu")[0]):
        fn, params, opt, key = steps()["sbi"]
    rng = np.random.default_rng(17)
    theta = rng.uniform(0.5, 1.5, (512, 4)).astype(np.float32)
    x = rng.standard_normal((512, 6)).astype(np.float32)

    def dense(params, opt, x, theta):
        loss, grads = jax.value_and_grad(sbi_loss)(params, x, theta)
        return adamw(params, grads, opt), loss

    with jax.default_device(jax.devices("cpu")[0]):
        cnf_state, cnf_data = cnf_density.initial_state(), cnf_density.mixture_data(256)
    cnf_inputs = jax.device_put((cnf_state, cnf_data), device)
    return {
        "sbi_step": (fn, jax.device_put((params, opt, key), device)),
        "simulation": (sbi_trajectory, jax.device_put((theta,), device)),
        "dense_update": (dense, jax.device_put((params, opt, x, theta), device)),
        "cnf_update": (cnf_density.make_training_step(256)[0], cnf_inputs),
        "cnf_update_jax": (cnf_density.make_training_step(256, variant="jax")[0], cnf_inputs),
    }


def _batch(run, args, count: int) -> dict:
    _pjrt.stats(reset=True)
    start = time.perf_counter()
    for _ in range(count):
        out = run(*args)
    dispatched = time.perf_counter() - start
    jax.block_until_ready(out)
    wall = time.perf_counter() - start
    s = _pjrt.stats(reset=True)
    return {
        "wall": wall / count,
        "host": (dispatched - s["throttle_seconds"]) / count,
        "encode": s["encode_seconds"] / count,
        "gpu": s["gpu_busy_seconds"] / count,
        "launches": s["launches"] / count,
    }


def profile(name: str, fn, args, sample_seconds: float, repeats: int) -> dict:
    run = jax.jit(fn)
    start = time.perf_counter()
    jax.block_until_ready(run(*args))
    first = time.perf_counter() - start
    count = 1
    while (elapsed := _batch(run, args, count)["wall"] * count) < sample_seconds:
        count = max(count + 1, math.ceil(count * min(4, sample_seconds / max(elapsed, 1e-9))))
    samples = [_batch(run, args, count) for _ in range(repeats)]
    result = {key: statistics.median(s[key] for s in samples) for key in samples[0]}
    result["gpu_fraction"] = result["gpu"] / result["wall"]
    return {
        "workload": name,
        "first_call_seconds": first,
        "calls_per_sample": count,
        **result,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sample-seconds", type=float, default=0.3)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--workload",
        choices=("sbi_step", "simulation", "dense_update", "cnf_update", "cnf_update_jax"),
        action="append",
        help="profile only these (repeatable); default: all",
    )
    args = parser.parse_args()
    jax.config.update("jax_enable_compilation_cache", False)
    device = jax.devices("metal")[0]
    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "device": device.device_kind,
        "os": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "revision": _revision(),
        "workloads": [
            profile(name, fn, inputs, args.sample_seconds, args.repeats)
            for name, (fn, inputs) in _workloads(device).items()
            if not args.workload or name in args.workload
        ],
    }
    print(
        f"{'workload':14s} {'wall ms':>8s} {'host ms':>8s} {'encode ms':>9s} "
        f"{'gpu ms':>7s} {'gpu/wall':>8s} {'launches':>8s}"
    )
    for w in result["workloads"]:
        print(
            f"{w['workload']:14s} {w['wall'] * 1e3:8.3f} {w['host'] * 1e3:8.3f} "
            f"{w['encode'] * 1e3:9.3f} {w['gpu'] * 1e3:7.3f} {w['gpu_fraction']:8.1%} "
            f"{w['launches']:8.0f}"
        )
    if args.output:
        _write_json(args.output, result)


if __name__ == "__main__":
    main()
