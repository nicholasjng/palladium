"""The sbibm SIR simulator on the GPU, as an `sbi` simulator.

`sbi.inference.simulate_for_sbi` draws parameters from the prior and
calls a user simulator on them, batched through joblib on the CPU. This
example ports the SIR task from the sbibm benchmark suite (Lueckmann et
al. 2021) to a Pallas kernel, one Metal thread per parameter draw, and
plugs it into `simulate_for_sbi` unchanged.

The task as sbibm defines it: beta ~ LogNormal(log 0.4, 0.5), gamma ~
LogNormal(log 0.125, 0.2); a population of N = 1e6 with one initial
infection; the SIR ODE integrated over 160 days; the infected fraction
I(t)/N read at t = 0, 17, ..., 153 (every 17th unit-spaced sample); the
observation x = Binomial(1000, I/N) at those ten times. The GPU kernel
integrates the ODE (RK4, fixed step) and returns the ten fractions; the
binomial draw stays in NumPy on both paths, so the comparison is
simulator against simulator.

Three simulators are timed on the same parameter draws:
- NumPy: the same RK4 scheme, vectorized over the whole batch; the
  strongest CPU baseline within NumPy.
- SciPy: `solve_ivp` per draw, the idiomatic sbi ODE simulator, run
  through `simulate_for_sbi` with joblib workers. Needs `sbi` installed
  (`uv run --with sbi examples/sbi_sir.py`).
- palladium: the Pallas kernel through `metal_call`, NumPy in and out
  around a jitted dispatch, the callable shape `simulate_for_sbi` expects.
"""

from __future__ import annotations

import argparse
import functools
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

import palladium

POPULATION = 1_000_000.0
INITIAL_INFECTED = 1.0
DAYS_BETWEEN_OBSERVATIONS = 17
OBSERVATIONS = 10
TOTAL_COUNT = 1000
DT = 0.1
STEPS_PER_OBSERVATION = round(DAYS_BETWEEN_OBSERVATIONS / DT)
PRIOR_LOC = np.array([np.log(0.4), np.log(0.125)], np.float32)
PRIOR_SCALE = np.array([0.5, 0.2], np.float32)


def sample_prior(n: int, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(PRIOR_LOC + PRIOR_SCALE * rng.standard_normal((n, 2))).astype(np.float32)


def observe(fractions: np.ndarray, seed: int = 29) -> np.ndarray:
    """sbibm's observation model: Binomial(1000, I/N) counts at each time."""
    rng = np.random.default_rng(seed)
    return rng.binomial(TOTAL_COUNT, np.clip(fractions, 0.0, 1.0)).astype(np.float32)


# ------------------------------------------------------------------ NumPy
def numpy_simulator(theta: np.ndarray) -> np.ndarray:
    """Vectorized RK4 over the batch; the same scheme as the kernel."""
    beta = theta[:, 0].astype(np.float32)
    gamma = theta[:, 1].astype(np.float32)
    s = np.full_like(beta, POPULATION - INITIAL_INFECTED)
    i = np.full_like(beta, INITIAL_INFECTED)
    out = np.empty((theta.shape[0], OBSERVATIONS), np.float32)
    dt = np.float32(DT)

    def rhs(s, i):
        infections = beta * s * i / np.float32(POPULATION)
        return -infections, infections - gamma * i

    for k in range(OBSERVATIONS):
        out[:, k] = i / np.float32(POPULATION)
        for _ in range(STEPS_PER_OBSERVATION):
            k1s, k1i = rhs(s, i)
            k2s, k2i = rhs(s + 0.5 * dt * k1s, i + 0.5 * dt * k1i)
            k3s, k3i = rhs(s + 0.5 * dt * k2s, i + 0.5 * dt * k2i)
            k4s, k4i = rhs(s + dt * k3s, i + dt * k3i)
            s = s + dt / 6 * (k1s + 2 * k2s + 2 * k3s + k4s)
            i = i + dt / 6 * (k1i + 2 * k2i + 2 * k3i + k4i)
    return out


# ------------------------------------------------------------------ SciPy
def scipy_simulator(theta: np.ndarray) -> np.ndarray:
    """One `solve_ivp` call per draw: the idiomatic sbi user simulator."""
    from scipy.integrate import solve_ivp

    times = np.arange(OBSERVATIONS) * DAYS_BETWEEN_OBSERVATIONS
    out = np.empty((theta.shape[0], OBSERVATIONS), np.float32)
    for row, (beta, gamma) in enumerate(np.asarray(theta, np.float64).tolist()):

        def rhs(_, u, beta=beta, gamma=gamma):
            s, i, _r = u
            infections = beta * s * i / POPULATION
            return [-infections, infections - gamma * i, gamma * i]

        solution = solve_ivp(
            rhs,
            (0.0, float(times[-1])),
            [POPULATION - INITIAL_INFECTED, INITIAL_INFECTED, 0.0],
            t_eval=times,
            rtol=1e-6,
            atol=1e-6,
        )
        out[row] = solution.y[1] / POPULATION
    return out


# -------------------------------------------------------------- palladium
def sir_kernel(beta_ref, gamma_ref, out_ref):
    beta, gamma = beta_ref[...], gamma_ref[...]
    population = jnp.float32(POPULATION)

    def rhs(s, i):
        infections = beta * s * i / population
        return -infections, infections - gamma * i

    def step(_, carry):
        s, i = carry
        k1s, k1i = rhs(s, i)
        k2s, k2i = rhs(s + 0.5 * DT * k1s, i + 0.5 * DT * k1i)
        k3s, k3i = rhs(s + 0.5 * DT * k2s, i + 0.5 * DT * k2i)
        k4s, k4i = rhs(s + DT * k3s, i + DT * k3i)
        return (
            s + DT / 6.0 * (k1s + 2.0 * k2s + 2.0 * k3s + k4s),
            i + DT / 6.0 * (k1i + 2.0 * k2i + 2.0 * k3i + k4i),
        )

    def observation(k, carry):
        s, i = carry
        out_ref[0, k] = (i / population)[0]
        return jax.lax.fori_loop(0, STEPS_PER_OBSERVATION, step, (s, i))

    s0 = beta * 0.0 + (POPULATION - INITIAL_INFECTED)
    i0 = beta * 0.0 + INITIAL_INFECTED
    jax.lax.fori_loop(0, OBSERVATIONS, observation, (s0, i0))


@functools.lru_cache(maxsize=8)
def metal_call_for(n: int):
    point = pl.BlockSpec((1,), lambda i: (i,))
    return jax.jit(
        palladium.metal_call(
            sir_kernel,
            grid=(n,),
            in_specs=[point, point],
            out_specs=pl.BlockSpec((1, OBSERVATIONS), lambda i: (i, 0)),
            out_shape=jax.ShapeDtypeStruct((n, OBSERVATIONS), jnp.float32),
        )
    )


def palladium_simulator(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, np.float32)
    call = metal_call_for(theta.shape[0])
    return np.asarray(call(np.ascontiguousarray(theta[:, 0]), np.ascontiguousarray(theta[:, 1])))


# ------------------------------------------------------------------- sbi
def as_sbi_simulator(fractions):
    """Wrap a fraction simulator into the callable `simulate_for_sbi` wants:
    a parameter batch in, the observation batch out, as torch tensors."""
    import torch

    def simulator(theta):
        theta_np = theta.detach().cpu().numpy() if torch.is_tensor(theta) else np.asarray(theta)
        return torch.as_tensor(observe(fractions(theta_np)), dtype=torch.float32)

    return simulator


def run_sbi(n: int, scipy_n: int, workers: int) -> None:
    import torch
    from sbi.inference import simulate_for_sbi
    from sbi.utils import process_prior, process_simulator

    prior, *_ = process_prior(
        torch.distributions.Independent(
            torch.distributions.LogNormal(torch.as_tensor(PRIOR_LOC), torch.as_tensor(PRIOR_SCALE)),
            1,
        )
    )
    print(f"\nsbi.simulate_for_sbi, {n} simulations (scipy rows run {scipy_n} and extrapolate)")
    rows = []
    for name, fractions, count, kwargs in (
        (
            "scipy, 1 worker",
            scipy_simulator,
            scipy_n,
            {"num_workers": 1, "simulation_batch_size": 1},
        ),
        (
            f"scipy, {workers} workers",
            scipy_simulator,
            scipy_n,
            {"num_workers": workers, "simulation_batch_size": max(1, scipy_n // (4 * workers))},
        ),
        ("numpy, batched", numpy_simulator, n, {"num_workers": 1, "simulation_batch_size": None}),
        (
            "palladium, batched",
            palladium_simulator,
            n,
            {"num_workers": 1, "simulation_batch_size": None},
        ),
    ):
        simulator = process_simulator(as_sbi_simulator(fractions), prior, False)
        simulate_for_sbi(simulator, prior, 64, show_progress_bar=False, **kwargs)  # warm up
        t0 = time.perf_counter()
        theta, x = simulate_for_sbi(
            simulator, prior, count, seed=1, show_progress_bar=False, **kwargs
        )
        seconds = (time.perf_counter() - t0) * n / count
        rows.append((name, seconds, tuple(theta.shape), tuple(x.shape)))
    base = rows[0][1]
    for name, seconds, theta_shape, x_shape in rows:
        print(
            f"  {name:22s} {seconds:10.3f} s  ({base / seconds:8.0f}x)  theta{theta_shape} x{x_shape}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=100_000, help="parameter draws")
    parser.add_argument(
        "--scipy-n", type=int, default=2_000, help="draws for the per-draw SciPy run"
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--sbi", action="store_true", help="also time through sbi (needs sbi)")
    args = parser.parse_args()

    theta = sample_prior(args.n)
    print(f"SIR ensemble: {args.n} draws, RK4 dt={DT}, {OBSERVATIONS} observations")

    t0 = time.perf_counter()
    reference = numpy_simulator(theta)
    t_numpy = time.perf_counter() - t0
    print(f"numpy, vectorized RK4 (CPU):        {t_numpy * 1e3:9.1f} ms")

    palladium_simulator(theta)  # trace + emit + Metal compile outside the clock
    t0 = time.perf_counter()
    got = palladium_simulator(theta)
    t_metal = time.perf_counter() - t0
    print(f"palladium, RK4 (Apple GPU):         {t_metal * 1e3:9.1f} ms ({t_numpy / t_metal:.1f}x)")
    print(f"  max abs deviation from numpy: {np.max(np.abs(got - reference)):.2e} (fractions)")

    small = theta[: args.scipy_n]
    t0 = time.perf_counter()
    scipy_reference = scipy_simulator(small)
    t_scipy = time.perf_counter() - t0
    per_draw = t_scipy / args.scipy_n
    print(
        f"scipy, solve_ivp per draw (CPU):    {t_scipy:.3f} s for {args.scipy_n} draws, "
        f"{per_draw * args.n:.1f} s extrapolated to {args.n} ({per_draw * args.n / t_metal:.0f}x)"
    )
    print(
        f"  max abs deviation, palladium vs scipy: {np.max(np.abs(got[: args.scipy_n] - scipy_reference)):.2e}"
    )

    x = observe(got)
    print(f"observations: shape {x.shape}, dtype {x.dtype}, first row {x[0].astype(int).tolist()}")

    if args.sbi:
        run_sbi(args.n, args.scipy_n, args.workers)


if __name__ == "__main__":
    main()
