"""Compare an RK4 Pallas custom call with ``jit(vmap(scan))`` on jax-mps.

Run from the jax-mps environment with
``JAX_PLATFORMS=mps,cpu uv run mew run --random-interleaving benchmarks/``.

The two MPS paths solve exactly the same fixed-step Lotka--Volterra RK4
problem.  The Palladium path is a single Pallas-generated Metal dispatch;
the baseline is the idiomatic JAX ``jit(vmap(scan(...)))`` expression.  Both
return a downstream JAX reduction, demonstrating that the custom call stays
inside the jax-mps graph.
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

DT = 0.01


def ensemble(n: int, seed: int = 17) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    return tuple(
        rng.uniform(low, high, n).astype(np.float32)
        for low, high in (
            (0.8, 1.2),
            (0.8, 1.2),
            (0.9, 1.3),
            (0.3, 0.5),
            (0.08, 0.12),
            (0.3, 0.5),
        )
    )


@functools.cache
def make_pallas_solver(n: int, steps: int):
    def kernel(x_ref, y_ref, a_ref, b_ref, c_ref, d_ref, xo_ref, yo_ref):
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

        x, y = jax.lax.fori_loop(0, steps, step, (x_ref[...], y_ref[...]))
        xo_ref[...] = x
        yo_ref[...] = y

    spec = pl.BlockSpec((1,), lambda i: (i,))
    call = palladium.mps_call_jit(
        kernel,
        grid=(n,),
        in_specs=[spec] * 6,
        out_specs=(spec, spec),
        out_shape=(
            jax.ShapeDtypeStruct((n,), jnp.float32),
            jax.ShapeDtypeStruct((n,), jnp.float32),
        ),
    )

    @jax.jit
    def solve(*args):
        x, y = call(*args)
        # A normal jax-mps operation after the custom call, included in both paths.
        return jnp.sum(x) + jnp.sum(y)

    return solve


@functools.cache
def make_jax_solver(steps: int):
    def one(x0, y0, a, b, c, d):
        def rhs(x, y):
            return a * x - b * x * y, c * x * y - d * y

        def step(carry, _):
            x, y = carry
            k1x, k1y = rhs(x, y)
            k2x, k2y = rhs(x + 0.5 * DT * k1x, y + 0.5 * DT * k1y)
            k3x, k3y = rhs(x + 0.5 * DT * k2x, y + 0.5 * DT * k2y)
            k4x, k4y = rhs(x + DT * k3x, y + DT * k3y)
            return (
                x + DT / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
                y + DT / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
            ), None

        return jax.lax.scan(step, (x0, y0), xs=None, length=steps)[0]

    @jax.jit
    def solve(x, y, a, b, c, d):
        final_x, final_y = jax.vmap(one)(x, y, a, b, c, d)
        return jnp.sum(final_x) + jnp.sum(final_y)

    return solve


STEPS = 500
SIZES = (10_000, 100_000)
CASES = [{"path": path, "n": n} for n in SIZES for path in ("palladium", "jax-mps", "jax-cpu")]
IDS = [f"{case['path']}-n{case['n']}" for case in CASES]


@mew.parametrize(CASES, ids=IDS, tags="rk4-ensemble", use_real_time=True, unit="ms")
def bench_rk4_ensemble(state: mew.State, path: str, n: int) -> None:
    host_args = ensemble(n)
    platform = "cpu" if path == "jax-cpu" else "mps"
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        state.skip_with_error(f"No {platform} device; set JAX_PLATFORMS=mps,cpu for jax-mps")
        return

    with jax.default_device(device):
        args = tuple(jax.device_put(x, device) for x in host_args)
        solver = make_pallas_solver(n, STEPS) if path == "palladium" else make_jax_solver(STEPS)
        start = time.perf_counter()
        actual = jax.block_until_ready(solver(*args))
        state.set_counter("first_call_ms", (time.perf_counter() - start) * 1000)
        if path != "jax-cpu":
            reference = jax.block_until_ready(make_jax_solver(STEPS)(*args))
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(reference), rtol=2e-5, atol=2e-3
            )
        for _ in range(3):
            jax.block_until_ready(solver(*args))
        for _ in state:
            jax.block_until_ready(solver(*args))
