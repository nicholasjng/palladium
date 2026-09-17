"""Fit shared Lotka--Volterra parameters through a fused MPS RK4 solve.

Run from the jax-mps checkout::

    JAX_PLATFORMS=mps env -u VIRTUAL_ENV \
      uv run --project . python ../palladium/examples/07_mps_parameter_recovery.py

The forward solve is one ``palladium.dispatch`` custom call.  Its custom VJP
uses the pure-JAX RK4 reference as a temporary correctness oracle, so this is
an end-to-end ``value_and_grad`` example, *not* a claim that backward execution
is fused yet.  Replacing that reference VJP with a tested Pallas discrete
adjoint is the next performance milestone.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

import palladium

DT, STEPS, N, ITERATIONS, LEARNING_RATE = 0.01, 100, 4_096, 500, 3e-2


def reference_solve(x0, y0, a, b, c, d):
    """Vectorized fixed-step RK4 reference with per-trajectory parameters."""

    def rhs(x, y):
        return a * x - b * x * y, c * x * y - d * y

    def step(_, carry):
        x, y = carry
        k1x, k1y = rhs(x, y)
        k2x, k2y = rhs(x + 0.5 * DT * k1x, y + 0.5 * DT * k1y)
        k3x, k3y = rhs(x + 0.5 * DT * k2x, y + 0.5 * DT * k2y)
        k4x, k4y = rhs(x + DT * k3x, y + DT * k3y)
        return (
            x + DT / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x),
            y + DT / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y),
        )

    return jax.lax.fori_loop(0, STEPS, step, (x0, y0))


def make_fused_solve():
    def rk4_kernel(x_ref, y_ref, a_ref, b_ref, c_ref, d_ref, xo_ref, yo_ref):
        a, b, c, d = a_ref[...], b_ref[...], c_ref[...], d_ref[...]

        def rhs(x, y):
            return a * x - b * x * y, c * x * y - d * y

        def step(_, carry):
            x, y = carry
            k1x, k1y = rhs(x, y)
            k2x, k2y = rhs(x + 0.5 * DT * k1x, y + 0.5 * DT * k1y)
            k3x, k3y = rhs(x + 0.5 * DT * k2x, y + 0.5 * DT * k2y)
            k4x, k4y = rhs(x + DT * k3x, y + DT * k3y)
            return (
                x + DT / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x),
                y + DT / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y),
            )

        x, y = jax.lax.fori_loop(0, STEPS, step, (x_ref[...], y_ref[...]))
        xo_ref[...] = x
        yo_ref[...] = y

    point = pl.BlockSpec((1,), lambda i: (i,))
    return palladium.mps_call_jit(
        rk4_kernel,
        grid=(N,),
        in_specs=[point] * 6,
        out_specs=(point, point),
        out_shape=(
            jax.ShapeDtypeStruct((N,), jnp.float32),
            jax.ShapeDtypeStruct((N,), jnp.float32),
        ),
        vjp_reference=reference_solve,
    )


def main() -> None:
    device = jax.devices("mps")[0]
    rng = np.random.default_rng(17)
    with jax.default_device(device):
        x0 = jnp.asarray(rng.uniform(0.8, 1.2, N), dtype=jnp.float32)
        y0 = jnp.asarray(rng.uniform(0.8, 1.2, N), dtype=jnp.float32)
        truth = tuple(
            jnp.asarray([value], dtype=jnp.float32) for value in (1.1, 0.4, 0.1, 0.4)
        )
        observed_x, observed_y = reference_solve(x0, y0, *truth)
        solve = make_fused_solve()

        def loss(parameters):
            # MLX represents literal-size inputs in Metal's ``constant``
            # address space, whereas Palladium kernels use device pointers.
            # Expand the shared parameters at this boundary; the transpose of
            # broadcast_to sums the per-trajectory VJP back to four scalars.
            expanded = tuple(
                jnp.broadcast_to(parameter, (N,)) for parameter in parameters
            )
            got_x, got_y = solve(x0, y0, *expanded)
            return jnp.mean((got_x - observed_x) ** 2 + (got_y - observed_y) ** 2)

        value_and_grad = jax.jit(jax.value_and_grad(loss))
        parameters = tuple(
            jnp.asarray([value], dtype=jnp.float32) for value in (0.9, 0.5, 0.08, 0.5)
        )
        first_moment = tuple(jnp.zeros_like(parameter) for parameter in parameters)
        second_moment = tuple(jnp.zeros_like(parameter) for parameter in parameters)
        value, gradients = value_and_grad(parameters)  # Compile outside the clock.
        jax.block_until_ready((value, gradients))
        start = time.perf_counter()
        for step in range(1, ITERATIONS + 1):
            value, gradients = value_and_grad(parameters)
            first_moment = tuple(
                0.9 * moment + 0.1 * gradient
                for moment, gradient in zip(first_moment, gradients, strict=True)
            )
            second_moment = tuple(
                0.999 * moment + 0.001 * gradient**2
                for moment, gradient in zip(second_moment, gradients, strict=True)
            )
            parameters = tuple(
                parameter
                - LEARNING_RATE
                * (moment / (1 - 0.9**step))
                / (jnp.sqrt(variance / (1 - 0.999**step)) + 1e-8)
                for parameter, moment, variance in zip(
                    parameters, first_moment, second_moment, strict=True
                )
            )
        jax.block_until_ready((value, parameters))

    recovered = np.asarray(jnp.concatenate(parameters))
    print(f"{ITERATIONS} optimizer steps: {(time.perf_counter() - start) * 1e3:.1f} ms")
    print(f"final loss: {float(value):.6e}")
    print(f"recovered (a, b, c, d): {recovered}")
    print("truth     (a, b, c, d): [1.1 0.4 0.1 0.4]")


if __name__ == "__main__":
    main()
