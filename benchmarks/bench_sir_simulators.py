"""The SIR simulator from examples/sbi_sir.py against NumPy and torch.

Every candidate integrates the same batch of parameter draws with the same
RK4 scheme and is checked against the NumPy result before timing. The
torch candidates are eager on CPU and MPS, and ``torch.compile`` with the
Inductor Metal backend fusing either one RK4 step or one 170-step
observation interval per kernel. Compiling the full 1,700-step simulator
does not finish in reasonable time and is not a case. The first call,
which includes compilation, is reported as the ``first_call_ms`` counter.

torch is not a project dependency:

    uv run --with torch mew run --random-interleaving benchmarks/bench_sir_simulators.py
"""

from __future__ import annotations

import functools
import os
import sys
import time

import mew
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
from sbi_sir import (
    DT,
    INITIAL_INFECTED,
    OBSERVATIONS,
    POPULATION,
    STEPS_PER_OBSERVATION,
    numpy_simulator,
    palladium_simulator,
    sample_prior,
)

CANDIDATES = (
    "numpy",
    "torch-cpu",
    "torch-mps",
    "torch-compile-step",
    "torch-compile-interval",
    "palladium",
)
SIZES = (100_000, 1_000_000)
CASES = [{"candidate": c, "draws": n} for n in SIZES for c in CANDIDATES]
IDS = [f"{case['candidate']}-n{case['draws']}" for case in CASES]


@functools.cache
def _theta(n: int) -> np.ndarray:
    return sample_prior(n)


@functools.cache
def _expected(n: int) -> np.ndarray:
    return numpy_simulator(_theta(n))


def _rk4_step(s, i, beta, gamma):
    def rhs(s, i):
        infections = beta * s * i / POPULATION
        return -infections, infections - gamma * i

    k1s, k1i = rhs(s, i)
    k2s, k2i = rhs(s + 0.5 * DT * k1s, i + 0.5 * DT * k1i)
    k3s, k3i = rhs(s + 0.5 * DT * k2s, i + 0.5 * DT * k2i)
    k4s, k4i = rhs(s + DT * k3s, i + DT * k3i)
    return (
        s + DT / 6 * (k1s + 2 * k2s + 2 * k3s + k4s),
        i + DT / 6 * (k1i + 2 * k2i + 2 * k3i + k4i),
    )


def _interval(s, i, beta, gamma):
    for _ in range(STEPS_PER_OBSERVATION):
        s, i = _rk4_step(s, i, beta, gamma)
    return s, i


def _torch_simulator(interval, device):
    import torch

    def simulate(theta_np: np.ndarray) -> np.ndarray:
        theta = torch.as_tensor(theta_np).to(device)
        beta, gamma = theta[:, 0], theta[:, 1]
        s = torch.full_like(beta, POPULATION - INITIAL_INFECTED)
        i = torch.full_like(beta, INITIAL_INFECTED)
        out = torch.empty((theta.shape[0], OBSERVATIONS), dtype=torch.float32, device=device)
        for k in range(OBSERVATIONS):
            out[:, k] = i / POPULATION
            s, i = interval(s, i, beta, gamma)
        return out.cpu().numpy()

    return simulate


def _candidate(name: str):
    if name == "numpy":
        return numpy_simulator
    if name == "palladium":
        return palladium_simulator
    import torch

    if name == "torch-cpu":
        return _torch_simulator(_interval, "cpu")
    if not torch.backends.mps.is_available():
        return None
    if name == "torch-mps":
        return _torch_simulator(_interval, "mps")
    if name == "torch-compile-step":
        step = torch.compile(_rk4_step, dynamic=False)

        def interval(s, i, beta, gamma):
            for _ in range(STEPS_PER_OBSERVATION):
                s, i = step(s, i, beta, gamma)
            return s, i

        return _torch_simulator(interval, "mps")
    return _torch_simulator(torch.compile(_interval, dynamic=False), "mps")


@mew.parametrize(CASES, ids=IDS, tags="sir-simulators", use_real_time=True, unit="ms")
def bench_sir_simulators(state: mew.State, candidate: str, draws: int) -> None:
    try:
        simulate = _candidate(candidate)
    except ImportError:
        state.skip_with_error("requires torch; run with `uv run --with torch`")
        return
    if simulate is None:
        state.skip_with_error("torch reports no MPS device")
        return
    theta = _theta(draws)
    start = time.perf_counter()
    got = simulate(theta)
    state.set_counter("first_call_ms", (time.perf_counter() - start) * 1000)
    np.testing.assert_allclose(got, _expected(draws), rtol=1e-4, atol=1e-6)
    simulate(theta)
    for _ in state:
        simulate(theta)
