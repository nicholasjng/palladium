"""Kernels shared by several test modules."""

import jax

DT, STEPS = 0.01, 500


def lv_kernel(x_ref, y_ref, a_ref, b_ref, c_ref, d_ref, xo_ref, yo_ref):
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
            x + DT / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
            y + DT / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
        )

    x, y = jax.lax.fori_loop(0, STEPS, step, (x_ref[...], y_ref[...]))
    xo_ref[...] = x
    yo_ref[...] = y
