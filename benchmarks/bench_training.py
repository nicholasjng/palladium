"""Whole training steps on the default JAX device, compared across platforms.

The rows carry no platform in their names: run this file once per platform
and let ``mew compare`` line them up (``docs/benchmarking.md``)::

    JAX_PLATFORMS=metal,cpu uv run mew run benchmarks/bench_training.py -o metal.jsonl
    JAX_PLATFORMS=cpu uv run mew run benchmarks/bench_training.py -o cpu.jsonl
    uv run mew compare cpu.jsonl metal.jsonl

Steps run pipelined, as in training: inputs stay resident, and each timed
batch of 64 calls synchronizes once at its end. Rows report wall time per
step. On ``metal`` they also count, per step, the plugin's GPU busy time as a
fraction of wall time (``gpu_fraction``), its launches, and its host time in
Execute (``encode_us``). ``*_first_call`` rows compile and run a step once
with JAX's caches cleared.

Workloads: the SBI step of ``training_steps.py`` and its halves (the 200-step
RK4 simulation; the MDN loss, gradients and AdamW), the census LLM step, and
the CNF update of ``cnf_density.py`` in plain JAX and with Palladium's adjoint
kernels (``metal`` only). A workload a platform cannot run is skipped.
"""

from __future__ import annotations

import os
import sys
import time

import jax
import mew
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
import cnf_density
from training_steps import adamw, sbi_loss, sbi_trajectory, workload

BATCH = 64


def _sbi_dense(params, opt, x, theta):
    loss, grads = jax.value_and_grad(sbi_loss)(params, x, theta)
    return adamw(params, grads, opt), loss


def _build(name: str, device):
    """The step function and its device-resident arguments; built on CPU first."""
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        if name in ("sbi_step", "llm_step"):
            fn, *args = workload(name.removesuffix("_step"))
        elif name in ("sbi_simulation", "sbi_dense_update"):
            rng = np.random.default_rng(17)
            theta = rng.uniform(0.5, 1.5, (512, 4)).astype(np.float32)
            if name == "sbi_simulation":
                fn, args = sbi_trajectory, [theta]
            else:
                _, params, opt, _ = workload("sbi")
                x = rng.standard_normal((512, 6)).astype(np.float32)
                fn, args = _sbi_dense, [params, opt, x, theta]
        else:
            if name == "cnf_update_kernels" and device.platform != "metal":
                raise RuntimeError("Palladium's CNF kernels run inside jit on metal only")
            variant = "jax" if name == "cnf_update_jax" else "reverse"
            fn, _ = cnf_density.make_training_step(256, variant=variant)
            args = [cnf_density.initial_state(), cnf_density.mixture_data(256)]
    return jax.jit(fn), jax.device_put(args, device)


def _stats(reset: bool):
    from palladium import _pjrt

    return _pjrt.stats(reset=reset)


WORKLOADS = [
    "sbi_step",
    "sbi_simulation",
    "sbi_dense_update",
    "llm_step",
    "cnf_update_jax",
    "cnf_update_kernels",
]


@mew.parametrize(
    [{"workload": name} for name in WORKLOADS],
    ids=WORKLOADS,
    use_real_time=True,
    unit="us",
    tags="training",
)
def bench_step(state: mew.State, workload: str) -> None:
    """Pipelined steps per second on the default device, with plugin counters on metal."""
    device = jax.devices()[0]
    try:
        run, args = _build(workload, device)
        out = jax.block_until_ready(run(*args))
    except Exception as e:  # noqa: BLE001 - a platform that cannot run a workload skips it
        state.skip_with_error(f"{device.platform}: {type(e).__name__}: {str(e)[:200]}")
        return
    for _ in range(3):
        out = run(*args)
    jax.block_until_ready(out)
    metal = device.platform == "metal"
    if metal:
        _stats(reset=True)
    steps, start = 0, time.perf_counter()
    for count in state.batches(BATCH):
        for _ in range(count):
            out = run(*args)
        jax.block_until_ready(out)
        steps += count
    wall = time.perf_counter() - start
    if metal and steps:
        s = _stats(reset=True)
        state.set_counter("gpu_fraction", s["gpu_busy_seconds"] / wall)
        state.set_counter("launches", s["launches"] / steps)
        state.set_counter("encode_us", s["encode_seconds"] / steps * 1e6)


@mew.parametrize(
    [{"workload": name} for name in ("sbi_step", "llm_step", "cnf_update_kernels")],
    ids=["sbi_step", "llm_step", "cnf_update_kernels"],
    use_manual_time=True,
    min_time="3x",
    unit="ms",
    tags="training",
)
def bench_first_call(state: mew.State, workload: str) -> None:
    """Compilation plus the first execution of a fresh jit, JAX's caches cleared."""
    device = jax.devices()[0]
    jax.config.update("jax_enable_compilation_cache", False)
    for _ in state:
        jax.clear_caches()
        try:
            run, args = _build(workload, device)
            jax.block_until_ready(args)
            start = time.perf_counter()
            jax.block_until_ready(run(*args))
        except Exception as e:  # noqa: BLE001 - a platform that cannot run a workload skips it
            state.skip_with_error(f"{device.platform}: {type(e).__name__}: {str(e)[:200]}")
            return
        state.set_iteration_time(time.perf_counter() - start)
