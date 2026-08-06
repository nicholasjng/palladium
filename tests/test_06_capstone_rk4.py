"""Exercise 5, the capstone: a batched RK4 Lotka-Volterra integrator.

No new emitter rules: if exercises 1-4 are green, this composes them.
One Metal thread integrates one (prey, predator) system over `STEPS` RK4
steps with its own parameters; the grid is the ensemble. This is the
kernel that examples/01_ode_ensembles.py benchmarks against Diffrax.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium

pytestmark = pytest.mark.exercise

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


def make_solver(n: int):
    spec_1 = pl.BlockSpec((1,), lambda i: (i,))
    return palladium.metal_call(
        lv_kernel,
        grid=(n,),
        in_specs=[spec_1] * 6,
        out_specs=(spec_1, spec_1),
        out_shape=(
            jax.ShapeDtypeStruct((n,), jnp.float32),
            jax.ShapeDtypeStruct((n,), jnp.float32),
        ),
    )


def _ensemble(rng, n):
    x0 = rng.uniform(0.8, 1.2, n).astype(np.float32)
    y0 = rng.uniform(0.8, 1.2, n).astype(np.float32)
    a = rng.uniform(0.9, 1.3, n).astype(np.float32)
    b = rng.uniform(0.3, 0.5, n).astype(np.float32)
    c = rng.uniform(0.08, 0.12, n).astype(np.float32)
    d = rng.uniform(0.3, 0.5, n).astype(np.float32)
    return x0, y0, a, b, c, d


def test_matches_interpret_oracle(rng):
    n = 512
    f = make_solver(n)
    args = _ensemble(rng, n)
    got_x, got_y = f(*args)
    want_x, want_y = f.interpret(*args)
    # 500 RK4 steps of f32 accumulate real drift between FAST-math GPU and
    # CPU; 1e-3 relative is the honest bar (tighten with MathMode.SAFE or
    # the df32 prelude later, and see how far you get).
    np.testing.assert_allclose(got_x, np.asarray(want_x), rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(got_y, np.asarray(want_y), rtol=1e-3, atol=1e-5)


def test_matches_diffrax(rng):
    diffrax = pytest.importorskip("diffrax")
    n = 64
    f = make_solver(n)
    x0, y0, a, b, c, d = _ensemble(rng, n)
    got_x, got_y = f(x0, y0, a, b, c, d)

    def rhs(t, u, p):
        x, y = u
        a_, b_, c_, d_ = p
        return jnp.stack([a_ * x - b_ * x * y, c_ * x * y - d_ * y])

    @jax.vmap
    def solve(u0, p):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            diffrax.Tsit5(),
            t0=0.0,
            t1=DT * STEPS,
            dt0=DT,
            y0=u0,
            args=p,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
        )
        return sol.ys[-1]

    want = solve(np.stack([x0, y0], axis=1), np.stack([a, b, c, d], axis=1))
    np.testing.assert_allclose(got_x, np.asarray(want)[:, 0], rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(got_y, np.asarray(want)[:, 1], rtol=2e-3, atol=1e-4)
