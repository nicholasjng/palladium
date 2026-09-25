"""Recover shared LV parameters with fully jitted loss, VJP, and Adam.

Run in the modified jax-mps environment with JAX_PLATFORMS=mps,cpu.
CPU uses the portable Pallas interpreter; choose --variant jax for CPU speed.
"""

import argparse
import time

import jax
import numpy as np

from palladium.workloads.ode_training import (
    initial_state,
    make_solver,
    make_training_step,
    recovery_problem,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument(
        "--variant",
        choices=("jax", "reference", "tangent", "reverse"),
        default="reverse",
    )
    args = parser.parse_args()
    with jax.default_device(jax.devices(args.device)[0]):
        data = recovery_problem(args.n, steps=args.steps, dt=args.dt)
        solve = make_solver(
            args.n,
            steps=args.steps,
            dt=args.dt,
            interval=args.interval,
            variant=args.variant,
        )
        update, loss = make_training_step(solve)
        state = initial_state()
        start = time.perf_counter()
        jax.block_until_ready(update(state, data))
        print(f"First training call: {1000 * (time.perf_counter() - start):.2f} ms")
        initial_loss = float(loss(state[0], data))
        start = time.perf_counter()
        for _ in range(args.iterations):
            state, _ = update(state, data)
        jax.block_until_ready(state)
        elapsed = time.perf_counter() - start
        final_loss = float(loss(state[0], data))
        print(f"{args.iterations} steps: {elapsed:.3f} s")
        print(f"Loss: {initial_loss:.6g} -> {final_loss:.6g}")
        print(f"Recovered: {np.asarray(state[0])}; truth: [1.1, 0.4, 0.1, 0.4]")


if __name__ == "__main__":
    main()
