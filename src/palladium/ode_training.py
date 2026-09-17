"""Fixed-step Lotka–Volterra training workload and explicit Pallas VJPs.

This is an experimental workload, not a general-purpose ODE solver.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

import palladium


def reference_solve(x0, y0, a, b, c, d, *, steps=100, dt=0.01):
    """Vectorized fixed-step RK4 reference with per-trajectory parameters."""

    def rhs(x, y):
        return a * x - b * x * y, c * x * y - d * y

    def step(_, carry):
        x, y = carry
        k1x, k1y = rhs(x, y)
        k2x, k2y = rhs(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
        k3x, k3y = rhs(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
        k4x, k4y = rhs(x + dt * k3x, y + dt * k3y)
        return (
            x + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x),
            y + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y),
        )

    return jax.lax.fori_loop(0, steps, step, (x0, y0))


def make_solver(n, *, steps=100, dt=0.01, interval=10, variant="reverse"):
    """Build a fixed-step solver. Interval 1 is the full-history reverse oracle.

    Reverse recomputation uses O(steps * interval) work and
    2*n*ceil(steps/interval) float32 checkpoint elements.
    """
    if n < 2 or steps < 1 or interval < 1 or dt <= 0:
        raise ValueError("require n >= 2, steps >= 1, interval >= 1, dt > 0")
    if variant not in ("jax", "reference", "tangent", "reverse", "forward"):
        raise ValueError(f"unknown variant {variant!r}")
    reference = functools.partial(reference_solve, steps=steps, dt=dt)
    if variant == "jax":
        return reference
    checkpoints = (steps + interval - 1) // interval

    def rk4_kernel(
        x_ref,
        y_ref,
        a_ref,
        b_ref,
        c_ref,
        d_ref,
        xo_ref,
        yo_ref,
        *checkpoint_refs,
    ):
        a, b, c, d = a_ref[...], b_ref[...], c_ref[...], d_ref[...]

        def rhs(x, y):
            return a * x - b * x * y, c * x * y - d * y

        def step(_, carry):
            x, y = carry
            k1x, k1y = rhs(x, y)
            k2x, k2y = rhs(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
            k3x, k3y = rhs(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
            k4x, k4y = rhs(x + dt * k3x, y + dt * k3y)
            return (
                x + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x),
                y + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y),
            )

        def segment(index, carry):
            x, y = carry
            if variant == "reverse":
                checkpoint_refs[0][:, index] = x
                checkpoint_refs[1][:, index] = y

            def bounded_step(i, state):
                return jax.lax.cond(
                    index * interval + i < steps,
                    lambda value: step(i, value),
                    lambda value: value,
                    state,
                )

            return jax.lax.fori_loop(0, interval, bounded_step, (x, y))

        x, y = jax.lax.fori_loop(0, checkpoints, segment, (x_ref[...], y_ref[...]))
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
            x2, y2 = x + 0.5 * dt * k1x, y + 0.5 * dt * k1y
            dx2, dy2 = dx + 0.5 * dt * dk1x, dy + 0.5 * dt * dk1y
            k2x, k2y = rhs(x2, y2)
            dk2x, dk2y = tangent_rhs(x2, y2, dx2, dy2, da, db, dc, dd)
            x3, y3 = x + 0.5 * dt * k2x, y + 0.5 * dt * k2y
            dx3, dy3 = dx + 0.5 * dt * dk2x, dy + 0.5 * dt * dk2y
            k3x, k3y = rhs(x3, y3)
            dk3x, dk3y = tangent_rhs(x3, y3, dx3, dy3, da, db, dc, dd)
            x4, y4 = x + dt * k3x, y + dt * k3y
            dx4, dy4 = dx + dt * dk3x, dy + dt * dk3y
            k4x, k4y = rhs(x4, y4)
            dk4x, dk4y = tangent_rhs(x4, y4, dx4, dy4, da, db, dc, dd)
            return (
                x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x),
                y + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y),
                dx + dt / 6 * (dk1x + 2 * dk2x + 2 * dk3x + dk4x),
                dy + dt / 6 * (dk1y + 2 * dk2y + 2 * dk3y + dk4y),
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
                steps,
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

    def rk4_reverse_vjp_kernel(
        x_ref,
        y_ref,
        a_ref,
        b_ref,
        c_ref,
        d_ref,
        checkpoint_x_ref,
        checkpoint_y_ref,
        cotangent_x_ref,
        cotangent_y_ref,
        x_gradient_ref,
        y_gradient_ref,
        a_gradient_ref,
        b_gradient_ref,
        c_gradient_ref,
        d_gradient_ref,
    ):
        """Discrete reverse RK4 with bounded recomputation from sparse checkpoints."""
        del x_ref, y_ref
        a, b, c, d = a_ref[...], b_ref[...], c_ref[...], d_ref[...]

        def rhs(x, y):
            return a * x - b * x * y, c * x * y - d * y

        def rhs_t(x, y, fx, fy):
            return (
                (a - b * y) * fx + c * y * fy,
                -b * x * fx + (c * x - d) * fy,
                x * fx,
                -x * y * fx,
                x * y * fy,
                -y * fy,
            )

        def reverse_step(index, carry):
            lx, ly, ga, gb, gc, gd = carry
            step_index = steps - 1 - index
            checkpoint_index = step_index // interval
            local_step = step_index % interval
            x, y = (
                checkpoint_x_ref[:, checkpoint_index],
                checkpoint_y_ref[:, checkpoint_index],
            )

            def advance(_, state):
                sx, sy = state
                sk1x, sk1y = rhs(sx, sy)
                sk2x, sk2y = rhs(sx + 0.5 * dt * sk1x, sy + 0.5 * dt * sk1y)
                sk3x, sk3y = rhs(sx + 0.5 * dt * sk2x, sy + 0.5 * dt * sk2y)
                sk4x, sk4y = rhs(sx + dt * sk3x, sy + dt * sk3y)
                return (
                    sx + dt / 6 * (sk1x + 2 * sk2x + 2 * sk3x + sk4x),
                    sy + dt / 6 * (sk1y + 2 * sk2y + 2 * sk3y + sk4y),
                )

            def bounded_advance(i, state):
                return jax.lax.cond(
                    i < local_step,
                    lambda current: advance(i, current),
                    lambda current: current,
                    state,
                )

            x, y = jax.lax.fori_loop(0, interval, bounded_advance, (x, y))
            k1x, k1y = rhs(x, y)
            x2, y2 = x + 0.5 * dt * k1x, y + 0.5 * dt * k1y
            k2x, k2y = rhs(x2, y2)
            x3, y3 = x + 0.5 * dt * k2x, y + 0.5 * dt * k2y
            k3x, k3y = rhs(x3, y3)
            x4, y4 = x + dt * k3x, y + dt * k3y
            k4x, k4y = rhs(x4, y4)
            del k4x, k4y
            q1x, q1y, q2x, q2y = dt / 6 * lx, dt / 6 * ly, dt / 3 * lx, dt / 3 * ly
            q3x, q3y, q4x, q4y = q2x, q2y, q1x, q1y
            bx4, by4, da, db, dc, dd = rhs_t(x4, y4, q4x, q4y)
            ga, gb, gc, gd = ga + da, gb + db, gc + dc, gd + dd
            lx, ly = lx + bx4, ly + by4
            q3x, q3y = q3x + dt * bx4, q3y + dt * by4
            bx3, by3, da, db, dc, dd = rhs_t(x3, y3, q3x, q3y)
            ga, gb, gc, gd = ga + da, gb + db, gc + dc, gd + dd
            lx, ly, q2x, q2y = (
                lx + bx3,
                ly + by3,
                q2x + 0.5 * dt * bx3,
                q2y + 0.5 * dt * by3,
            )
            bx2, by2, da, db, dc, dd = rhs_t(x2, y2, q2x, q2y)
            ga, gb, gc, gd = ga + da, gb + db, gc + dc, gd + dd
            lx, ly, q1x, q1y = (
                lx + bx2,
                ly + by2,
                q1x + 0.5 * dt * bx2,
                q1y + 0.5 * dt * by2,
            )
            bx1, by1, da, db, dc, dd = rhs_t(x, y, q1x, q1y)
            return lx + bx1, ly + by1, ga + da, gb + db, gc + dc, gd + dd

        zero = a * 0
        lx, ly, ga, gb, gc, gd = jax.lax.fori_loop(
            0,
            steps,
            reverse_step,
            (cotangent_x_ref[...], cotangent_y_ref[...], zero, zero, zero, zero),
        )
        x_gradient_ref[...] = lx
        y_gradient_ref[...] = ly
        a_gradient_ref[...] = ga
        b_gradient_ref[...] = gb
        c_gradient_ref[...] = gc
        d_gradient_ref[...] = gd

    point = pl.BlockSpec((1,), lambda i: (i,))
    checkpoint_row = pl.BlockSpec((1, checkpoints), lambda i: (i, 0))
    forward = palladium.mps_call_jit(
        rk4_kernel,
        grid=(n,),
        in_specs=[point] * 6,
        out_specs=(point, point)
        + ((checkpoint_row, checkpoint_row) if variant == "reverse" else ()),
        out_shape=(
            jax.ShapeDtypeStruct((n,), jnp.float32),
            jax.ShapeDtypeStruct((n,), jnp.float32),
        )
        + (
            (jax.ShapeDtypeStruct((n, checkpoints), jnp.float32),) * 2
            if variant == "reverse"
            else ()
        ),
    )
    if variant == "forward":
        return forward
    if variant == "reference":
        return forward.with_reference_vjp(reference)
    backward = palladium.mps_call_jit(
        rk4_tangent_vjp_kernel if variant == "tangent" else rk4_reverse_vjp_kernel,
        grid=(n,),
        in_specs=[point] * 6
        + ([checkpoint_row, checkpoint_row] if variant == "reverse" else [])
        + [point] * 2,
        out_specs=(point,) * 6,
        out_shape=(jax.ShapeDtypeStruct((n,), jnp.float32),) * 6,
    )
    return (
        forward.with_auxiliary_vjp(backward, output_count=2)
        if variant == "reverse"
        else forward.with_vjp(backward)
    )


def recovery_problem(n, *, steps=100, dt=0.01, seed=17):
    """Deterministic observations, generated independently on CPU."""
    rng = np.random.default_rng(seed)
    host = tuple(rng.uniform(0.8, 1.2, n).astype(np.float32) for _ in range(2))
    truth = np.array([1.1, 0.4, 0.1, 0.4], np.float32)
    with jax.default_device(jax.devices("cpu")[0]):
        observed = jax.jit(functools.partial(reference_solve, steps=steps, dt=dt))(
            *map(jnp.asarray, host), *map(jnp.asarray, truth)
        )
        observed = tuple(np.asarray(value) for value in observed)
    return (*map(jnp.asarray, host), *map(jnp.asarray, observed))


def initial_state():
    parameters = jnp.array([0.9, 0.5, 0.08, 0.5], jnp.float32)
    return (
        parameters,
        jnp.zeros_like(parameters),
        jnp.zeros_like(parameters),
        jnp.int32(0),
    )


def make_training_step(solve, *, learning_rate=0.03):
    """One jit includes loss, VJP, moment updates, and parameter updates."""

    def loss(parameters, data):
        x, y, target_x, target_y = data
        expanded = tuple(jnp.broadcast_to(parameters[i], x.shape) for i in range(4))
        result_x, result_y = solve(x, y, *expanded)
        return jnp.mean((result_x - target_x) ** 2 + (result_y - target_y) ** 2)

    @jax.jit
    def update(state, data):
        parameters, moment, variance, count = state
        value, gradient = jax.value_and_grad(loss)(parameters, data)
        count = count + 1
        moment = 0.9 * moment + 0.1 * gradient
        variance = 0.999 * variance + 0.001 * gradient**2
        parameters = parameters - learning_rate * (moment / (1 - 0.9**count)) / (
            jnp.sqrt(variance / (1 - 0.999**count)) + 1e-8
        )
        return (parameters, moment, variance, count), value

    return update, jax.jit(loss)
