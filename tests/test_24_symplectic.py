"""Velocity-Verlet on the Kepler problem: a structure-preserving kernel.

No new emitter rules -- it composes load/store, elementwise, sqrt, grids
and fori_loop -- but it is the one kernel whose correctness has a check
sharper than "matches the oracle": the exact invariants of the continuous
problem. Initial conditions are set at perihelion of a unit-semi-major-axis
orbit, where E = -1/2 and L = sqrt(1 - e^2) hold in closed form for every
member. Velocity-Verlet conserves L exactly for a central force (up to
round-off) and holds E inside an O(h^2) bound forever, so a lowering bug
that a loose float32 oracle tolerance would wave through -- a dropped
carry copy-back, a sign flip in the second kick -- breaks an invariant by
orders of magnitude.

This is the kernel examples/06_symplectic_longrun.py measures.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium

H = 0.002
STEPS = 20_000  # ~6.4 orbits


def make_verlet_kernel(steps):
    def kernel(qx_ref, qy_ref, px_ref, py_ref, qxo, qyo, pxo, pyo):
        def accel(qx, qy):
            r2 = qx * qx + qy * qy
            inv_r3 = -1.0 / (r2 * jnp.sqrt(r2))
            return inv_r3 * qx, inv_r3 * qy

        def step(_, carry):  # kick, drift, kick
            qx, qy, px, py = carry
            ax, ay = accel(qx, qy)
            px, py = px + 0.5 * H * ax, py + 0.5 * H * ay
            qx, qy = qx + H * px, qy + H * py
            ax, ay = accel(qx, qy)
            return qx, qy, px + 0.5 * H * ax, py + 0.5 * H * ay

        carry = (qx_ref[...], qy_ref[...], px_ref[...], py_ref[...])
        qxo[...], qyo[...], pxo[...], pyo[...] = jax.lax.fori_loop(
            0, steps, step, carry
        )

    return kernel


def make_solver(n, steps=STEPS, **kwargs):
    spec_1 = pl.BlockSpec((1,), lambda i: (i,))
    out = jax.ShapeDtypeStruct((n,), jnp.float32)
    return palladium.metal_call(
        make_verlet_kernel(steps),
        grid=(n,),
        in_specs=[spec_1] * 4,
        out_specs=(spec_1,) * 4,
        out_shape=(out,) * 4,
        **kwargs,
    )


def _ensemble(n):
    """Perihelion start, eccentricities spread over [0, 0.4]."""
    e = np.linspace(0.0, 0.4, n)
    return (
        (1.0 - e).astype(np.float32),
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.sqrt((1.0 + e) / (1.0 - e)).astype(np.float32),
        e,
    )


def _energy(qx, qy, px, py):
    q = np.stack([qx, qy]).astype(np.float64)
    p = np.stack([px, py]).astype(np.float64)
    return 0.5 * np.sum(p * p, axis=0) - 1.0 / np.sqrt(np.sum(q * q, axis=0))


def _angular_momentum(qx, qy, px, py):
    qx, qy, px, py = (np.asarray(v, dtype=np.float64) for v in (qx, qy, px, py))
    return qx * py - qy * px


def test_matches_interpret_oracle():
    n = 256
    f = make_solver(n)
    *args, _ = _ensemble(n)
    got = f(*args)
    want = f.interpret(*args)
    # 20k Verlet steps of f32: the GPU and the CPU oracle diverge in phase
    # along the orbit long before they diverge in the invariants, so this
    # is a loose positional bar by construction. The invariant tests below
    # are the sharp ones.
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, np.asarray(w), rtol=2e-2, atol=2e-2)


def test_conserves_angular_momentum():
    """Exact for a central force under Verlet, so the only budget here is
    round-off over 20k float32 steps: eps*sqrt(steps) ~ 1e-5, times a
    small constant, and FAST math reassociates on top of that. 1e-4 is
    the honest bar and still four orders below any structural bug.
    """
    n = 256
    f = make_solver(n)
    *args, e = _ensemble(n)
    got = _angular_momentum(*f(*args))
    np.testing.assert_allclose(got, np.sqrt(1.0 - e**2), rtol=1e-4, atol=0.0)


def test_energy_error_stays_inside_the_step_size_bound():
    """The symplectic property: the error is set by h, not by the horizon.

    At h=0.002 the modified-Hamiltonian bound for these orbits is ~1e-6;
    5e-5 leaves room for the float32 round-off walk over 20k steps and
    still fails loudly if the kernel is not conserving anything.
    """
    n = 256
    f = make_solver(n)
    *args, _ = _ensemble(n)
    start = _energy(*args)
    end = _energy(*f(*args))
    assert np.max(np.abs(end - start)) < 5e-5


def test_energy_bound_tracks_step_size_not_horizon():
    """Double the horizon at fixed h: the bound must not double with it.

    A method that drifted secularly (or a kernel that had quietly stopped
    integrating a Hamiltonian) would show the error scaling with the step
    count. Round-off contributes its own sqrt(2) at most.
    """
    n = 256
    *args, _ = _ensemble(n)
    start = _energy(*args)
    short = np.max(np.abs(_energy(*make_solver(n, STEPS)(*args)) - start))
    long = np.max(np.abs(_energy(*make_solver(n, 2 * STEPS)(*args)) - start))
    assert long < 3.0 * short


@pytest.mark.parametrize("steps", [1, 2, 64])
def test_short_horizons_match_a_numpy_reference(steps):
    """Bit-level-ish agreement while round-off has not accumulated: this
    is what catches an off-by-one in the loop or a mis-sequenced kick."""
    n = 64
    *args, _ = _ensemble(n)
    q = np.stack(args[:2]).astype(np.float64)
    p = np.stack(args[2:]).astype(np.float64)

    def accel(q):
        r2 = np.sum(q * q, axis=0)
        return -q / (r2 * np.sqrt(r2))

    for _ in range(steps):
        p = p + 0.5 * H * accel(q)
        q = q + H * p
        p = p + 0.5 * H * accel(q)

    got = make_solver(n, steps)(*args)
    want = (q[0], q[1], p[0], p[1])
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-6)
