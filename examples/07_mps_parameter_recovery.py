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

    def rk4_tangent_vjp_kernel(
        x_ref,
        y_ref,
        a_ref,
        b_ref,
        c_ref,
        d_ref,
        cotangent_x_ref,
        cotangent_y_ref,
        x_gradient_ref,
        y_gradient_ref,
        a_gradient_ref,
        b_gradient_ref,
        c_gradient_ref,
        d_gradient_ref,
    ):
        """VJP through the discrete RK4 update via six forward sensitivities.

        This is a fused tangent-transpose kernel: it recomputes each RK4 stage
        and propagates the state tangent for each input direction. It is a
        useful correctness and ABI milestone for this six-input problem. A
        reverse-time/checkpointed adjoint is the scalable replacement when the
        state or parameter dimension grows.
        """

        a, b, c, d = a_ref[...], b_ref[...], c_ref[...], d_ref[...]

        def rhs(x, y):
            return a * x - b * x * y, c * x * y - d * y

        def tangent_rhs(x, y, dx, dy, da, db, dc, dd):
            return (
                (a - b * y) * dx - b * x * dy + x * da - x * y * db,
                c * y * dx + (c * x - d) * dy + x * y * dc - y * dd,
            )

        def tangent_step(x, y, dx, dy, da, db, dc, dd):
            k1x, k1y = rhs(x, y)
            dk1x, dk1y = tangent_rhs(x, y, dx, dy, da, db, dc, dd)
            x2, y2 = x + 0.5 * DT * k1x, y + 0.5 * DT * k1y
            dx2, dy2 = dx + 0.5 * DT * dk1x, dy + 0.5 * DT * dk1y
            k2x, k2y = rhs(x2, y2)
            dk2x, dk2y = tangent_rhs(x2, y2, dx2, dy2, da, db, dc, dd)
            x3, y3 = x + 0.5 * DT * k2x, y + 0.5 * DT * k2y
            dx3, dy3 = dx + 0.5 * DT * dk2x, dy + 0.5 * DT * dk2y
            k3x, k3y = rhs(x3, y3)
            dk3x, dk3y = tangent_rhs(x3, y3, dx3, dy3, da, db, dc, dd)
            x4, y4 = x + DT * k3x, y + DT * k3y
            dx4, dy4 = dx + DT * dk3x, dy + DT * dk3y
            k4x, k4y = rhs(x4, y4)
            dk4x, dk4y = tangent_rhs(x4, y4, dx4, dy4, da, db, dc, dd)
            return (
                x + DT / 6 * (k1x + 2 * k2x + 2 * k3x + k4x),
                y + DT / 6 * (k1y + 2 * k2y + 2 * k3y + k4y),
                dx + DT / 6 * (dk1x + 2 * dk2x + 2 * dk3x + dk4x),
                dy + DT / 6 * (dk1y + 2 * dk2y + 2 * dk3y + dk4y),
            )

        zero = x_ref[...] * 0
        one = zero + 1

        def step(_, carry):
            x, y, dxx, dyx, dxy, dyy, dxa, dya, dxb, dyb, dxc, dyc, dxd, dyd = carry
            x_next, y_next, dxx, dyx = tangent_step(
                x, y, dxx, dyx, zero, zero, zero, zero
            )
            _, _, dxy, dyy = tangent_step(x, y, dxy, dyy, zero, zero, zero, zero)
            _, _, dxa, dya = tangent_step(x, y, dxa, dya, one, zero, zero, zero)
            _, _, dxb, dyb = tangent_step(x, y, dxb, dyb, zero, one, zero, zero)
            _, _, dxc, dyc = tangent_step(x, y, dxc, dyc, zero, zero, one, zero)
            _, _, dxd, dyd = tangent_step(x, y, dxd, dyd, zero, zero, zero, one)
            return (
                x_next,
                y_next,
                dxx,
                dyx,
                dxy,
                dyy,
                dxa,
                dya,
                dxb,
                dyb,
                dxc,
                dyc,
                dxd,
                dyd,
            )

        _, _, dxx, dyx, dxy, dyy, dxa, dya, dxb, dyb, dxc, dyc, dxd, dyd = (
            jax.lax.fori_loop(
                0,
                STEPS,
                step,
                (
                    x_ref[...],
                    y_ref[...],
                    one,
                    zero,
                    zero,
                    one,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                ),
            )
        )
        cotangent_x, cotangent_y = cotangent_x_ref[...], cotangent_y_ref[...]
        x_gradient_ref[...] = cotangent_x * dxx + cotangent_y * dyx
        y_gradient_ref[...] = cotangent_x * dxy + cotangent_y * dyy
        a_gradient_ref[...] = cotangent_x * dxa + cotangent_y * dya
        b_gradient_ref[...] = cotangent_x * dxb + cotangent_y * dyb
        c_gradient_ref[...] = cotangent_x * dxc + cotangent_y * dyc
        d_gradient_ref[...] = cotangent_x * dxd + cotangent_y * dyd

    point = pl.BlockSpec((1,), lambda i: (i,))
    forward = palladium.mps_call_jit(
        rk4_kernel,
        grid=(N,),
        in_specs=[point] * 6,
        out_specs=(point, point),
        out_shape=(
            jax.ShapeDtypeStruct((N,), jnp.float32),
            jax.ShapeDtypeStruct((N,), jnp.float32),
        ),
    )
    backward = palladium.mps_call_jit(
        rk4_tangent_vjp_kernel,
        grid=(N,),
        in_specs=[point] * 8,
        out_specs=(point,) * 6,
        out_shape=(jax.ShapeDtypeStruct((N,), jnp.float32),) * 6,
    )
    return forward.with_vjp(backward)


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

        def expanded_parameters(parameters):
            # MLX represents literal-size inputs in Metal's ``constant``
            # address space, whereas Palladium kernels use device pointers.
            # Expand the shared parameters at this boundary; the transpose of
            # broadcast_to sums the per-trajectory VJP back to four scalars.
            return tuple(jnp.broadcast_to(parameter, (N,)) for parameter in parameters)

        def loss(parameters):
            expanded = expanded_parameters(parameters)
            got_x, got_y = solve(x0, y0, *expanded)
            return jnp.mean((got_x - observed_x) ** 2 + (got_y - observed_y) ** 2)

        def reference_loss(parameters):
            got_x, got_y = reference_solve(x0, y0, *expanded_parameters(parameters))
            return jnp.mean((got_x - observed_x) ** 2 + (got_y - observed_y) ** 2)

        value_and_grad = jax.jit(jax.value_and_grad(loss))
        reference_value_and_grad = jax.jit(jax.value_and_grad(reference_loss))
        parameters = tuple(
            jnp.asarray([value], dtype=jnp.float32) for value in (0.9, 0.5, 0.08, 0.5)
        )
        first_moment = tuple(jnp.zeros_like(parameter) for parameter in parameters)
        second_moment = tuple(jnp.zeros_like(parameter) for parameter in parameters)
        value, gradients = value_and_grad(parameters)  # Compile outside the clock.
        reference_value, reference_gradients = reference_value_and_grad(parameters)
        jax.block_until_ready((value, gradients, reference_value, reference_gradients))
        np.testing.assert_allclose(value, reference_value, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(gradients, reference_gradients, rtol=2e-4, atol=2e-6)
        initial_max_gradient_error = max(
            float(jnp.max(jnp.abs(actual - expected)))
            for actual, expected in zip(gradients, reference_gradients, strict=True)
        )
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
    print(f"initial VJP max abs error: {initial_max_gradient_error:.3e}")
    print(f"recovered (a, b, c, d): {recovered}")
    print("truth     (a, b, c, d): [1.1 0.4 0.1 0.4]")


if __name__ == "__main__":
    main()
