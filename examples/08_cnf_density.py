"""Fit a 2D Gaussian mixture with a checkpointed neural ODE density model.

Run in the modified jax-mps environment with JAX_PLATFORMS=mps,cpu.
The base is a standard normal; density evaluation integrates from t=1 to 0.
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from palladium.cnf_training import (
    initial_state,
    make_training_step,
    mixture_data,
    mixture_log_prob,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument(
        "--variant", choices=("reverse", "reference", "jax"), default="reverse"
    )
    args = parser.parse_args()
    with jax.default_device(jax.devices(args.device)[0]):
        train = jnp.asarray(mixture_data(args.n, 17))
        heldout = jnp.asarray(mixture_data(args.n, 29))
        state = initial_state(args.width)
        update, loss = make_training_step(
            args.n,
            width=args.width,
            steps=args.steps,
            interval=args.interval,
            variant=args.variant,
            learning_rate=args.learning_rate,
        )
        start = time.perf_counter()
        jax.block_until_ready(update(state, train))
        first_ms = (time.perf_counter() - start) * 1000
        initial_nll = float(loss(state[0], heldout))
        curve = []
        start = time.perf_counter()
        for index in range(args.iterations):
            state, value = update(state, train)
            if (index + 1) % 100 == 0:
                curve.append(
                    {"iteration": index + 1, "pre_update_train_nll": float(value)}
                )
        jax.block_until_ready(state)
        train_seconds = time.perf_counter() - start
        final_nll = float(loss(state[0], heldout))
        params = np.asarray(state[0])
    # Evaluate the trained weights with an independent backend and finer time grid.
    with jax.default_device(jax.devices("cpu")[0]):
        _, reference_loss = make_training_step(
            args.n, width=args.width, steps=args.steps, variant="jax"
        )
        _, refined_loss = make_training_step(
            args.n, width=args.width, steps=2 * args.steps, variant="jax"
        )
        cpu_data, cpu_params = (
            jnp.asarray(mixture_data(args.n, 29)),
            jnp.asarray(params),
        )
        reference_nll = float(reference_loss(cpu_params, cpu_data))
        refined_nll = float(refined_loss(cpu_params, cpu_data))
        target_nll = -float(jnp.mean(mixture_log_prob(cpu_data)))
    print(
        json.dumps(
            {
                "config": vars(args),
                "first_training_call_ms": first_ms,
                "training_seconds": train_seconds,
                "initial_heldout_nll": initial_nll,
                "final_heldout_nll": final_nll,
                "cpu_reference_nll": reference_nll,
                "cpu_double_steps_nll": refined_nll,
                "true_distribution_nll": target_nll,
                "training_curve": curve,
                "parameters": params.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
