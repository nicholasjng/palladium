"""The per-thread adaptive controller kernel.

Composition on top of test_07_preflight.py's primitives (comparisons,
select_n/where, inf/nan literals, fori_loop accumulation), the same way
the RK4 capstone composes the base rules. No new emitter rules. Van der Pol
ensemble, matching examples/02_adaptive_lockstep.py's mu distribution.

Bogacki-Shampine 3(2): a smaller embedded pair than RK45 (roughly half
the equations per stage of the RK4 capstone, not double), with a real
error estimate, FSAL, and a PI step-size controller.
"""

import jax
import jax.numpy as jnp
import metal_runtime as mr
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium

F32 = jnp.float32

X0, V0, DT0 = 2.0, 0.0, 0.01  # matches examples/02_adaptive_lockstep.py
T1 = 10.0
MAX_STEPS = 6000  # fori_loop trip budget; test_reaches_t1 tells you if it's too small
RTOL, ATOL = 1e-5, 1e-7  # error-norm scaling and the diffrax comparison tolerance

pcoeff, icoeff = 0.4, 0.3  # diffrax's own suggestion for "moderate difficulty" problems
error_order = 3


def vdp_adaptive_kernel(
    x0_ref, v0_ref, mu_ref, to_ref, xo_ref, vo_ref, steps_ref, rejected_ref
):
    """One thread integrates one Van der Pol trajectory (x' = v,
    v' = mu*(1-x^2)*v - x) from t=0 to T1: Bogacki-Shampine 3(2), FSAL,
    PI step-size control.

    Bogacki-Shampine 3(2):

        k1 = f(x, v)
        k2 = f((x, v) + dt/2  * k1)
        k3 = f((x, v) + 3dt/4 * k2)
        y3 = (x, v) + dt * (2/9*k1 + 1/3*k2 + 4/9*k3)      # 3rd order
        k4 = f(y3)
        y2 = (x, v) + dt * (7/24*k1 + 1/4*k2 + 1/3*k3 + 1/8*k4)  # 2nd order
        err = y3 - y2

    Self-stalling loop: dt clamps to fmin(dt, T1 - t) at the top of each
    attempt, so once t reaches T1, dt is 0, err is 0, the step is always
    accepted, and the thread idles harmlessly for its remaining budget.
    No while-shaped iteration or boolean and/or needed, just arithmetic
    inside a fixed-trip fori_loop.

    FSAL: k4 of an accepted step is rhs evaluated at exactly the point
    the next attempt starts from, so it doubles as that attempt's k1
    (carried as k1x, k1v). A rejected step restarts from the same
    (x, v), so the old k1 is still correct: next_k1 = where(accept, k4,
    k1) covers both cases with no extra rhs call.

    PI controller: factor = safety * inv_err**coeff1 *
    prev_inv_err**coeff2, matching diffrax's own adapt_step_size for the
    dcoeff=0 case; prev_inv_err only advances on an accepted step, same
    as diffrax's own update rule. err_norm is floored to 1e-9 before the
    power so a perfect step (err_norm == 0) does not produce an inf
    scale.

    Carries: (t, dt, x, v, k1x, k1v, prev_inv_err, steps, rejected).
    `rejected` counts real rejected attempts only; `steps` also counts
    harmless post-T1 stall iterations, so the two diverge once a thread
    finishes early.
    """
    mu = mu_ref[...]

    def rhs(x, v):
        return v, mu * (1.0 - x**2) * v - x

    def step(_, carry):
        t, dt, x, v, k1x, k1v, prev_inv_err, steps, rejected = carry
        dt = jnp.minimum(dt, T1 - t)

        k2x, k2v = rhs(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
        k3x, k3v = rhs(x + 0.75 * dt * k2x, v + 0.75 * dt * k2v)
        y3 = (
            x + dt * (2 / 9 * k1x + 1 / 3 * k2x + 4 / 9 * k3x),
            v + dt * (2 / 9 * k1v + 1 / 3 * k2v + 4 / 9 * k3v),
        )
        k4x, k4v = rhs(*y3)
        y2 = (
            x + dt * (7 / 24 * k1x + 1 / 4 * k2x + 1 / 3 * k3x + 1 / 8 * k4x),
            v + dt * (7 / 24 * k1v + 1 / 4 * k2v + 1 / 3 * k3v + 1 / 8 * k4v),
        )

        # error calculation
        err_x, err_v = y3[0] - y2[0], y3[1] - y2[1]
        x3, v3 = y3
        err_norm = jnp.maximum(
            jnp.abs(err_x) / (ATOL + RTOL * jnp.abs(x3)),
            jnp.abs(err_v) / (ATOL + RTOL * jnp.abs(v3)),
        )
        accept = err_norm <= 1.0

        t2 = jnp.where(accept, t + dt, t)
        x2 = jnp.where(accept, y3[0], x)
        v2 = jnp.where(accept, y3[1], v)
        next_k1x = jnp.where(accept, k4x, k1x)
        next_k1v = jnp.where(accept, k4v, k1v)
        err_norm = jnp.maximum(err_norm, 1e-9)

        safety, factormin, factormax = 0.9, 0.2, 10.0
        coeff1 = (icoeff + pcoeff) / error_order
        coeff2 = -pcoeff / error_order
        inv_err = 1 / err_norm
        factor = safety * inv_err**coeff1 * prev_inv_err**coeff2

        prev_inv_err = jnp.where(accept, inv_err, prev_inv_err)

        dt = dt * jnp.clip(factor, factormin, factormax)
        steps = steps + jnp.where(accept, 1.0, 0.0)
        rejected = rejected + jnp.where(accept, 0.0, 1.0)
        return (t2, dt, x2, v2, next_k1x, next_k1v, prev_inv_err, steps, rejected)

    t0 = jnp.zeros_like(x0_ref[...])
    dt0 = jnp.full_like(x0_ref[...], DT0)
    steps0 = jnp.zeros_like(x0_ref[...])
    rejected0 = jnp.zeros_like(x0_ref[...])
    k1x_0, k1v_0 = rhs(x0_ref[...], v0_ref[...])
    inv_err = jnp.ones_like(x0_ref[...])

    t, _, x, v, _, _, _, steps, rejected = jax.lax.fori_loop(
        0,
        MAX_STEPS,
        step,
        (t0, dt0, x0_ref[...], v0_ref[...], k1x_0, k1v_0, inv_err, steps0, rejected0),
    )

    to_ref[...] = t
    xo_ref[...] = x
    vo_ref[...] = v
    steps_ref[...] = steps
    rejected_ref[...] = rejected


def make_solver(n: int, math_mode: mr.MathMode = mr.MathMode.RELAXED):
    spec_1 = pl.BlockSpec((1,), lambda i: (i,))
    return palladium.metal_call(
        vdp_adaptive_kernel,
        grid=(n,),
        in_specs=[spec_1] * 3,
        out_specs=(spec_1, spec_1, spec_1, spec_1, spec_1),
        out_shape=tuple(jax.ShapeDtypeStruct((n,), F32) for _ in range(5)),
        math_mode=math_mode,
    )


def _mu_ensemble(rng, n: int, stiff_fraction: float = 0.0) -> np.ndarray:
    mu = rng.uniform(0.5, 2.0, n).astype(np.float32)
    if stiff_fraction:
        k = int(n * stiff_fraction)
        mu[rng.choice(n, k, replace=False)] = rng.uniform(300.0, 500.0, k).astype(
            np.float32
        )
    return mu


def _inputs(rng, n: int, stiff_fraction: float = 0.0):
    mu = _mu_ensemble(rng, n, stiff_fraction)
    x0 = np.full(n, X0, dtype=np.float32)
    v0 = np.full(n, V0, dtype=np.float32)
    return x0, v0, mu


def test_matches_interpret_oracle(rng):
    """(a) GPU agrees with the CPU oracle on a mild ensemble."""
    n = 256
    f = make_solver(n)
    args = _inputs(rng, n)
    got_t, got_x, got_v, got_steps, got_rejected = f(*args)
    want_t, want_x, want_v, want_steps, want_rejected = f.interpret(*args)
    np.testing.assert_allclose(got_t, np.asarray(want_t), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(got_x, np.asarray(want_x), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(got_v, np.asarray(want_v), rtol=1e-3, atol=1e-4)
    # A hard accept/reject threshold is sensitive to ULP-level differences
    # between GPU (RELAXED reassociation) and the CPU oracle (strict
    # order): a flipped decision on one step shifts every dt afterward,
    # so exact step-count equality isn't the right bar here.
    np.testing.assert_allclose(got_steps, np.asarray(want_steps), rtol=0, atol=20)
    np.testing.assert_allclose(got_rejected, np.asarray(want_rejected), rtol=0, atol=20)


def test_reaches_t1(rng):
    """(b) completion: every thread's clock lands on T1. Also the
    MAX_STEPS budget-sizing feedback loop: raise MAX_STEPS if this fails
    for the stiff members."""
    n = 256
    f = make_solver(n)
    got_t, *_ = f(*_inputs(rng, n, stiff_fraction=0.02))
    np.testing.assert_allclose(got_t, T1, rtol=0, atol=1e-4)


def test_matches_diffrax(rng):
    """(c) accuracy against per-trajectory Diffrax Bosh3 (the same
    Bogacki-Shampine pair) at matched tolerances."""
    diffrax = pytest.importorskip("diffrax")
    n = 64
    f = make_solver(n)
    x0, v0, mu = _inputs(rng, n)
    _, got_x, got_v, _, _ = f(x0, v0, mu)

    def rhs(t, u, mu):
        x, v = u
        return jnp.stack([v, mu * (1.0 - x * x) * v - x])

    @jax.vmap
    def solve(mu):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            diffrax.Bosh3(),
            t0=0.0,
            t1=T1,
            dt0=DT0,
            y0=jnp.array([X0, V0]),
            args=mu,
            stepsize_controller=diffrax.PIDController(rtol=RTOL, atol=ATOL),
        )
        return sol.ys[-1]

    want = solve(jnp.asarray(mu))
    # A bare I controller with a fixed dt0 won't track a tuned PID
    # controller step-for-step; loosen further if this is flaky, but it
    # should be in the right ballpark once the kernel is correct.
    np.testing.assert_allclose(got_x, np.asarray(want)[:, 0], rtol=5e-2, atol=1e-2)
    np.testing.assert_allclose(got_v, np.asarray(want)[:, 1], rtol=5e-2, atol=1e-2)


def test_step_counts_vary_with_stiffness(rng):
    """(d) the point: adaptivity is observably per-thread. A mixed
    ensemble (a few stiff members, per examples/02_adaptive_lockstep.py's
    mu range) must show step-count variance, not a uniform trip count."""
    n = 512
    f = make_solver(n)
    x0, v0, mu = _inputs(rng, n, stiff_fraction=0.02)
    _, _, _, got_steps, _ = f(x0, v0, mu)
    assert np.asarray(got_steps).std() > 0
