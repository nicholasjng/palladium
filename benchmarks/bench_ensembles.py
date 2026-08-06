"""Ensemble ODE benchmarks: palladium's Metal kernels vs Diffrax on CPU.

The measured story behind examples/01: one thread per Lotka-Volterra
system, the whole RK4 solve fused into a single dispatch, against
jit(vmap(diffeqsolve)), the practical Diffrax deployment on macOS.

Dispatch rows use `use_real_time`: the GPU wait blocks without consuming
CPU time, so Google Benchmark's CPU-time default would misreport them
(same convention as metal-runtime's bench_overhead.py).

Run with `uv run mew run benchmarks/`, filter via `--tag palladium|diffrax`.
"""

import functools

import jax
import jax.numpy as jnp
import mew
import numpy as np
from jax.experimental import pallas as pl

import palladium

DT, STEPS = 0.01, 500
SIZES = [{"n": 10_000}, {"n": 100_000}]
IDS = ["n=10k", "n=100k"]


def ensemble(n: int, seed: int = 17):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(0.8, 1.2, n).astype(np.float32),  # x0 (prey)
        rng.uniform(0.8, 1.2, n).astype(np.float32),  # y0 (predator)
        rng.uniform(0.9, 1.3, n).astype(np.float32),  # a
        rng.uniform(0.3, 0.5, n).astype(np.float32),  # b
        rng.uniform(0.08, 0.12, n).astype(np.float32),  # c
        rng.uniform(0.3, 0.5, n).astype(np.float32),  # d
    )


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


@functools.cache
def palladium_solver(n: int) -> palladium.MetalCallable:
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


@mew.parametrize(SIZES, ids=IDS, tags="palladium", use_real_time=True, unit="ms")
def bench_lv_palladium(state: mew.State, n: int) -> None:
    """Full solve on the GPU: trace/emit/compile primed outside the loop."""
    solve = palladium_solver(n)
    args = ensemble(n)
    solve(*args)  # prime the shape cache and the Metal pipeline
    for _ in state:
        solve(*args)


@mew.parametrize(SIZES, ids=IDS, tags="diffrax", use_real_time=True, unit="ms")
def bench_lv_diffrax(state: mew.State, n: int) -> None:
    """jit(vmap(diffeqsolve)) with Tsit5 + PID on the CPU backend."""
    import diffrax

    def rhs(t, u, p):
        x, y = u
        a_, b_, c_, d_ = p
        return jnp.stack([a_ * x - b_ * x * y, c_ * x * y - d_ * y])

    @jax.jit
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

    x0, y0, a, b, c, d = ensemble(n)
    u0 = jnp.stack([x0, y0], axis=1)
    p = jnp.stack([a, b, c, d], axis=1)
    solve(u0, p).block_until_ready()  # compile outside the clock
    for _ in state:
        solve(u0, p).block_until_ready()
