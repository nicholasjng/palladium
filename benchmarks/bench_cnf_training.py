"""Benchmark complete CNF training updates with Mew.

Run with ``JAX_PLATFORMS=mps,cpu uv run mew run --random-interleaving benchmarks/``.
"""

import time

import jax
import mew
import numpy as np

from palladium.workloads.cnf_training import initial_state, make_training_step, mixture_data

N, WIDTH, STEPS = 256, 4, 16
INTERVALS = (1, 4, 8)
CASES = [
    {"platform": "cpu", "variant": "jax", "interval": 1},
    {"platform": "mps", "variant": "jax", "interval": 1},
    {"platform": "mps", "variant": "reference", "interval": 1},
]
CASES += [{"platform": "mps", "variant": "reverse", "interval": interval} for interval in INTERVALS]
IDS = [f"{case['platform']}-{case['variant']}-k{case['interval']}" for case in CASES]


@mew.parametrize(CASES, ids=IDS, tags="cnf-training", use_real_time=True, unit="ms")
def bench_cnf_training(state: mew.State, platform: str, variant: str, interval: int) -> None:
    with jax.default_device(jax.devices(platform)[0]):
        initial = initial_state(WIDTH)
        data = mixture_data(N)
        update, _ = make_training_step(
            N,
            width=WIDTH,
            steps=STEPS,
            interval=interval,
            variant=variant,
        )
        jax.block_until_ready((initial, data))
        start = time.perf_counter()
        executable = update.lower(initial, data).compile()
        compile_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        actual = jax.block_until_ready(executable(initial, data))
        first_execute_ms = (time.perf_counter() - start) * 1000

        reference_update, _ = make_training_step(N, width=WIDTH, steps=STEPS, variant="jax")
        expected = jax.block_until_ready(reference_update(initial, data))
        for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(got, want, rtol=4e-4, atol=3e-6)

        state.set_counter("compile_ms", compile_ms)
        state.set_counter("first_execute_ms", first_execute_ms)
        state.set_counter(
            "checkpoint_bytes",
            12 * N * ((STEPS + interval - 1) // interval) if variant == "reverse" else 0,
        )
        for _ in range(3):
            jax.block_until_ready(executable(initial, data))
        for _ in state:
            jax.block_until_ready(executable(initial, data))
