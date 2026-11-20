"""Whole training steps on the default JAX device, compared across platforms.

The rows carry no platform in their names: run this file once per platform
and let ``mew compare`` line them up (``docs/benchmarking.md``)::

    JAX_PLATFORMS=metal,cpu uv run mew run benchmarks/bench_training.py -o metal.jsonl
    JAX_PLATFORMS=cpu uv run mew run benchmarks/bench_training.py -o cpu.jsonl
    uv run mew compare cpu.jsonl metal.jsonl

Steps run pipelined, as in training: inputs stay resident, and each timed
batch of 64 calls synchronizes once at its end. Rows report wall time per
step. On ``metal`` they also carry the plugin's counters per step
(``plugin_stats.py``), with ``gpu_fraction``, the GPU's busy time over wall
time. ``*_first_call`` rows compile and run a step once with JAX's caches
cleared; the Metal driver's shader cache is not, so they depend on what the
machine compiled before.

Workloads: the SBI step of ``training_steps.py`` and its halves (the 200-step
RK4 simulation; the MDN loss, gradients and AdamW), the census LLM and
SAM2-like steps, and the CNF update of ``cnf_density.py`` in plain JAX and
with Palladium's adjoint kernels (``metal`` only). A workload a platform does
not support is skipped; any other failure on ``metal`` is raised, so the row
reports its traceback.
"""

from __future__ import annotations

import time

import cnf_density
import jax
import mew
import numpy as np
import plugin_stats
from training_steps import adamw, sbi_loss, sbi_trajectory, workload

BATCH = 64


class _Unsupported(Exception):
    """The platform does not run this workload, by design."""


def _skip(state: mew.State, device, error: Exception) -> None:
    """Skip the row for an unsupported workload; re-raise a failure of the plugin.

    On ``metal`` only UNIMPLEMENTED is a reason to skip: anything else is a
    defect that a missing row would hide. Other backends skip on any failure.
    """
    unsupported = isinstance(error, _Unsupported) or "UNIMPLEMENTED" in str(error)
    if device.platform == "metal" and not unsupported:
        raise error
    state.skip_with_error(f"{device.platform}: {type(error).__name__}: {str(error)[:200]}")


def _sbi_dense(params, opt, x, theta):
    loss, grads = jax.value_and_grad(sbi_loss)(params, x, theta)
    return adamw(params, grads, opt), loss


def _build(name: str, device):
    """The step function and its device-resident arguments; built on CPU first."""
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        if name in ("sbi_step", "llm_step", "sam_step"):
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
                raise _Unsupported("Palladium's CNF kernels run inside jit on metal only")
            variant = "jax" if name == "cnf_update_jax" else "reverse"
            fn, _ = cnf_density.make_training_step(256, variant=variant)
            args = [cnf_density.initial_state(), cnf_density.mixture_data(256)]
    return jax.jit(fn), jax.device_put(args, device)


WORKLOADS = [
    "sbi_step",
    "sbi_simulation",
    "sbi_dense_update",
    "llm_step",
    "sam_step",
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
    except Exception as e:  # noqa: BLE001 - _skip re-raises what is not a missing feature
        _skip(state, device, e)
        return
    for _ in range(3):
        out = run(*args)
    jax.block_until_ready(out)
    plugin_stats.reset()
    steps, start = 0, time.perf_counter()
    for count in state.batches(BATCH):
        for _ in range(count):
            out = run(*args)
        jax.block_until_ready(out)
        steps += count
    plugin_stats.report(state, steps, wall=time.perf_counter() - start)


@mew.parametrize(
    [{"workload": name} for name in ("sbi_step", "llm_step", "sam_step", "cnf_update_kernels")],
    ids=["sbi_step", "llm_step", "sam_step", "cnf_update_kernels"],
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
        except Exception as e:  # noqa: BLE001 - _skip re-raises what is not a missing feature
            _skip(state, device, e)
            return
        state.set_iteration_time(time.perf_counter() - start)
