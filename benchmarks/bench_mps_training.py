"""Benchmark complete ODE training updates with Mew.

Run with ``JAX_PLATFORMS=mps,cpu uv run mew run --random-interleaving benchmarks/``.
"""

import time

import jax
import mew
import numpy as np

from palladium.workloads.ode_training import (
    initial_state,
    make_solver,
    make_training_step,
    recovery_problem,
)

N, STEPS = 4096, 100
INTERVALS = (1, 5, 10, 25)
CASES = [
    {"variant": variant, "interval": interval}
    for variant, interval in (("jax", 1), ("reference", 1), ("tangent", 1))
]
CASES += [{"variant": "reverse", "interval": interval} for interval in INTERVALS]
IDS = [f"{case['variant']}-k{case['interval']}" for case in CASES]


@mew.parametrize(CASES, ids=IDS, tags="ode-training", use_real_time=True, unit="ms")
def bench_ode_training(state: mew.State, variant: str, interval: int) -> None:
    with jax.default_device(jax.devices("mps")[0]):
        data = recovery_problem(N, steps=STEPS)
        initial = initial_state()
        jax.block_until_ready((data, initial))

        update, _ = make_training_step(
            make_solver(N, steps=STEPS, interval=interval, variant=variant)
        )
        start = time.perf_counter()
        executable = update.lower(initial, data).compile()
        compile_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        actual = jax.block_until_ready(executable(initial, data))
        first_execute_ms = (time.perf_counter() - start) * 1000

        reference_update, _ = make_training_step(make_solver(N, steps=STEPS, variant="jax"))
        expected = jax.block_until_ready(reference_update(initial, data))
        for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(got, want, rtol=3e-4, atol=3e-6)

        state.set_counter("compile_ms", compile_ms)
        state.set_counter("first_execute_ms", first_execute_ms)
        state.set_counter(
            "checkpoint_bytes",
            8 * N * ((STEPS + interval - 1) // interval) if variant == "reverse" else 0,
        )
        for _ in range(3):
            jax.block_until_ready(executable(initial, data))
        for _ in state:
            jax.block_until_ready(executable(initial, data))
