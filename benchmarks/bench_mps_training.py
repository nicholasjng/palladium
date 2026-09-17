"""Paired, synchronized timings of complete jitted Adam updates.

Run in jax-mps's environment with JAX_PLATFORMS=mps,cpu. Input transfer,
reference observation generation and warmup are excluded. Every sample
starts at the same optimizer state; variant order rotates to reduce bias.
"""

import argparse
import json
import platform
import statistics
import time

import jax
import numpy as np

from palladium.ode_training import (
    initial_state,
    make_solver,
    make_training_step,
    recovery_problem,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--intervals", type=int, nargs="+", default=[1, 5, 10, 25])
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    rows, jobs = {}, {}
    variants = [("jax", 1), ("reference", 1), ("tangent", 1)]
    variants += [("reverse", interval) for interval in args.intervals]
    with jax.default_device(jax.devices("mps")[0]):
        data = recovery_problem(args.n, steps=args.steps)
        state = initial_state()
        jax.block_until_ready((data, state))
        expected = None
        for variant, interval in variants:
            name = f"{variant}/k={interval}"
            update, _ = make_training_step(
                make_solver(
                    args.n, steps=args.steps, interval=interval, variant=variant
                )
            )
            start = time.perf_counter()
            executable = update.lower(state, data).compile()
            compile_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            result = jax.block_until_ready(executable(state, data))
            first_ms = (time.perf_counter() - start) * 1000
            if expected is None:
                expected = jax.device_get(result)
            for actual, want in zip(
                jax.tree.leaves(result), jax.tree.leaves(expected), strict=True
            ):
                np.testing.assert_allclose(actual, want, rtol=3e-4, atol=3e-6)
            for _ in range(3):
                jax.block_until_ready(executable(state, data))
            jobs[name] = executable
            rows[name] = {
                "compile_ms": compile_ms,
                "first_execute_ms": first_ms,
                "checkpoint_bytes": (
                    8 * args.n * ((args.steps + interval - 1) // interval)
                    if variant == "reverse"
                    else 0
                ),
                "samples_ms": [],
            }
        names = list(jobs)
        for iteration in range(args.repeats):
            order = names[iteration % len(names) :] + names[: iteration % len(names)]
            for name in order:
                start = time.perf_counter()
                jax.block_until_ready(jobs[name](state, data))
                rows[name]["samples_ms"].append((time.perf_counter() - start) * 1000)
    for row in rows.values():
        row["median_ms"] = statistics.median(row["samples_ms"])
        row["min_ms"] = min(row["samples_ms"])
        row["max_ms"] = max(row["samples_ms"])
    print(
        json.dumps(
            {
                "jax": jax.__version__,
                "machine": platform.machine(),
                "n": args.n,
                "steps": args.steps,
                "repeats": args.repeats,
                "note": "checkpoint_bytes counts only saved x/y states, not total device memory; "
                "first_execute may include lazy Metal compilation; persistent caches were not cleared",
                "results": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
