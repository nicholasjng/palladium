"""A fixed-step RK4 Lotka-Volterra ensemble: one Palladium dispatch against JAX.

Run with ``JAX_PLATFORMS=mps,cpu uv run mew run --random-interleaving benchmarks/``.

``palladium`` is a plain ``pl.pallas_call`` on mps, ``palladium-ffi`` the
same kernel through ``metal_call`` from the CPU platform, and ``jax-mps``
and ``jax-cpu`` the idiomatic ``jit(vmap(scan(...)))``. Every path ends in
a downstream JAX reduction, so the kernel is timed inside the XLA graph.
"""

from __future__ import annotations

import functools
import time

import jax
import jax.numpy as jnp
import mew
import numpy as np
from jax.experimental import pallas as pl

import palladium

DT, STEPS = 0.01, 500
SIZES = (10_000, 100_000)
PATHS = {"palladium": "mps", "palladium-ffi": "cpu", "jax-mps": "mps", "jax-cpu": "cpu"}
CASES = [{"path": path, "n": n} for n in SIZES for path in PATHS]
IDS = [f"{case['path']}-n{case['n']}" for case in CASES]


def ensemble(n: int, seed: int = 17) -> tuple[np.ndarray, ...]:
    """Sample a reproducible float32 Lotka-Volterra initial-state and parameter ensemble.

    Parameters
    ----------
    n : int
        Number of ensemble members or examples.
    seed : int, optional
        Deterministic random seed, default 17.

    Returns
    -------
    tuple of numpy.ndarray
        Six vectors in (x0, y0, a, b, c, d) order, each of length n.
    """
    rng = np.random.default_rng(seed)
    bounds = ((0.8, 1.2), (0.8, 1.2), (0.9, 1.3), (0.3, 0.5), (0.08, 0.12), (0.3, 0.5))
    return tuple(rng.uniform(low, high, n).astype(np.float32) for low, high in bounds)


def rk4_step(x, y, a, b, c, d):
    """Advance one or many Lotka-Volterra systems by the fixed benchmark time step.

    Parameters
    ----------
    x, y : scalar or array-like
        Prey and predator populations with broadcast-compatible shapes.
    a, b, c, d : scalar or array-like
        Growth, predation, predator-growth, and decay rates.

    Returns
    -------
    tuple
        Updated prey and predator values after one DT-sized classical RK4 step.
    """

    def rhs(x, y):
        return a * x - b * x * y, c * x * y - d * y

    k1x, k1y = rhs(x, y)
    k2x, k2y = rhs(x + 0.5 * DT * k1x, y + 0.5 * DT * k1y)
    k3x, k3y = rhs(x + 0.5 * DT * k2x, y + 0.5 * DT * k2y)
    k4x, k4y = rhs(x + DT * k3x, y + DT * k3y)
    return (
        x + DT / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        y + DT / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
    )


def lv_kernel(x_ref, y_ref, a_ref, b_ref, c_ref, d_ref, xo_ref, yo_ref):
    """Integrate each program's Lotka-Volterra system for the fixed benchmark horizon.

    Parameters
    ----------
    x_ref, y_ref : Ref
        Initial population blocks.
    a_ref, b_ref, c_ref, d_ref : Ref
        Per-system model parameter blocks.
    xo_ref, yo_ref : Ref
        Final population output blocks.

    Notes
    -----
    STEPS and DT are module constants; no intermediate trajectory is returned.
    """
    params = a_ref[...], b_ref[...], c_ref[...], d_ref[...]
    x, y = jax.lax.fori_loop(
        0, STEPS, lambda _, xy: rk4_step(xy[0], xy[1], *params), (x_ref[...], y_ref[...])
    )
    xo_ref[...] = x
    yo_ref[...] = y


@functools.cache
def make_solver(path: str, n: int):
    """Build a cached jitted ensemble solver with a downstream population reduction.

    Parameters
    ----------
    path : {"palladium", "palladium-ffi", "jax-mps", "jax-cpu"}
        Benchmark implementation; the caller selects the corresponding default device.
    n : int
        Number of ensemble members or examples.

    Returns
    -------
    callable
        Function of the six ensemble arrays returning the sum of both final populations.

    Notes
    -----
    Palladium paths place one trajectory in each grid program; JAX paths use vmap of a scan.
    The cache key includes path and ensemble size; compilation occurs on first use.
    """
    if path.startswith("palladium"):
        spec = pl.BlockSpec((1,), lambda i: (i,))
        out = jax.ShapeDtypeStruct((n,), jnp.float32)
        call = (palladium.metal_call if path == "palladium-ffi" else pl.pallas_call)(
            lv_kernel, grid=(n,), in_specs=[spec] * 6, out_specs=(spec, spec), out_shape=(out, out)
        )
    else:

        def one(x0, y0, *params):
            step = lambda xy, _: (rk4_step(*xy, *params), None)
            return jax.lax.scan(step, (x0, y0), length=STEPS)[0]

        call = jax.vmap(one)

    @jax.jit
    def solve(*args):
        x, y = call(*args)
        return jnp.sum(x) + jnp.sum(y)

    return solve


@mew.parametrize(CASES, ids=IDS, tags="rk4-ensemble", use_real_time=True, unit="ms")
def bench_rk4_ensemble(state: mew.State, path: str, n: int) -> None:
    """Validate and time a synchronized ensemble solve plus downstream JAX reduction.

    Parameters
    ----------
    state : mew.State
        Benchmark iteration controller and destination for timing counters or skip
        reasons.
    path : str
        One of the implementation labels in PATHS.
    n : int
        Number of ensemble members or examples.

    Notes
    -----
    Missing JAX platforms are skipped. First-call setup is reported separately; three warm-
    ups precede timed iterations, with inputs resident on the selected device.
    """
    platform = PATHS[path]
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        state.skip_with_error(f"No {platform} device; set JAX_PLATFORMS=mps,cpu for jax-mps")
        return

    with jax.default_device(device):
        args = tuple(jax.device_put(x, device) for x in ensemble(n))
        solver = make_solver(path, n)
        start = time.perf_counter()
        actual = jax.block_until_ready(solver(*args))
        state.set_counter("first_call_ms", (time.perf_counter() - start) * 1000)
        if path.startswith("palladium"):
            reference = make_solver(f"jax-{platform}", n)(*args)
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-3)
        for _ in range(3):
            jax.block_until_ready(solver(*args))
        for _ in state:
            jax.block_until_ready(solver(*args))
