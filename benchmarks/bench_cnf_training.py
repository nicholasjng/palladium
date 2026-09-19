"""Synchronized, interleaved complete CNF training steps on MPS and CPU.

Run with JAX_PLATFORMS=mps,cpu from the modified jax-mps environment.
MLP width and RK4 step count are identical in every variant.
"""

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from palladium.cnf_training import initial_state, make_training_step, mixture_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--intervals", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if min(args.n, args.width, args.steps, args.repeats, *args.intervals) < 1:
        parser.error("all sizes and repetition counts must be positive")
    rows, jobs = {}, {}
    variants = [("cpu", "jax", 1), ("mps", "jax", 1), ("mps", "reference", 1)]
    variants += [("mps", "reverse", k) for k in args.intervals]
    expected = None
    for platform, variant, interval in variants:
        name = f"{platform}/{variant}/k={interval}"
        with jax.default_device(jax.devices(platform)[0]):
            state = initial_state(args.width)
            data = jnp.asarray(mixture_data(args.n))
            update, _ = make_training_step(
                args.n,
                width=args.width,
                steps=args.steps,
                interval=interval,
                variant=variant,
            )
            jax.block_until_ready((state, data))
            start = time.perf_counter()
            executable = update.lower(state, data).compile()
            compile_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            result = jax.block_until_ready(executable(state, data))
            first_ms = (time.perf_counter() - start) * 1000
            if expected is None:
                expected = jax.device_get(result)
            for got, want in zip(
                jax.tree.leaves(result), jax.tree.leaves(expected), strict=True
            ):
                np.testing.assert_allclose(got, want, rtol=4e-4, atol=3e-6)
            for _ in range(3):
                jax.block_until_ready(executable(state, data))
            jobs[name] = executable, state, data
            rows[name] = {
                "compile_ms": compile_ms,
                "first_execute_ms": first_ms,
                "checkpoint_bytes": 12
                * args.n
                * ((args.steps + interval - 1) // interval)
                if variant == "reverse"
                else None,
                "samples_ms": [],
            }
    names = list(jobs)
    for repeat in range(args.repeats):
        offset = repeat % len(names)
        for name in names[offset:] + names[:offset]:
            executable, state, data = jobs[name]
            start = time.perf_counter()
            jax.block_until_ready(executable(state, data))
            rows[name]["samples_ms"].append((time.perf_counter() - start) * 1000)
    for row in rows.values():
        samples = row["samples_ms"]
        row.update(
            median_ms=statistics.median(samples),
            min_ms=min(samples),
            max_ms=max(samples),
        )
    print(
        json.dumps(
            {
                "jax": jax.__version__,
                "config": vars(args),
                "note": "Identical starting Adam states; rotated order, synchronized samples. "
                "Checkpoint bytes exclude weights, cotangents, and other memory. "
                "First execution may include lazy Metal compilation; caches not cleared.",
                "results": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
