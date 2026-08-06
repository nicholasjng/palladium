"""Example 1: massive ensembles of small ODEs, palladium vs Diffrax.

One Metal thread integrates one Lotka-Volterra system (RK4, fixed step)
with its own parameters; the grid is the ensemble. The Diffrax baseline is
the same ensemble under jit(vmap(diffeqsolve)) on the CPU backend; on
macOS that IS the practical Diffrax deployment (jax-metal is stale and
cannot run it; see ROADMAP).

The palladium path activates once exercises 1-5 are green:
    uv run pytest tests/ -m exercise
Until then this script still runs and prints the baseline numbers.
"""

import time

import numpy as np

DT, STEPS = 0.01, 500
N = 100_000


def ensemble(n, seed=17):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(0.8, 1.2, n).astype(np.float32),  # x0 (prey)
        rng.uniform(0.8, 1.2, n).astype(np.float32),  # y0 (predator)
        rng.uniform(0.9, 1.3, n).astype(np.float32),  # a
        rng.uniform(0.3, 0.5, n).astype(np.float32),  # b
        rng.uniform(0.08, 0.12, n).astype(np.float32),  # c
        rng.uniform(0.3, 0.5, n).astype(np.float32),  # d
    )


def diffrax_baseline(x0, y0, a, b, c, d):
    import diffrax
    import jax
    import jax.numpy as jnp

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

    u0 = jnp.stack([x0, y0], axis=1)
    p = jnp.stack([a, b, c, d], axis=1)
    solve(u0, p).block_until_ready()  # compile outside the clock
    t0 = time.perf_counter()
    out = solve(u0, p).block_until_ready()
    return np.asarray(out), time.perf_counter() - t0


def palladium_solver(n):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    import palladium

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


def main():
    args = ensemble(N)
    print(f"Lotka-Volterra ensemble: N={N}, RK4, {STEPS} steps of dt={DT}")

    (baseline, t_diffrax) = diffrax_baseline(*args)
    print(f"diffrax (CPU, jit+vmap, Tsit5 adaptive): {t_diffrax:.3f} s")

    try:
        solve = palladium_solver(N)
        solve(*args)  # trace + emit + Metal compile outside the clock
        t0 = time.perf_counter()
        got_x, got_y = solve(*args)
        t_metal = time.perf_counter() - t0
        print(
            f"palladium (Apple GPU, RK4 fixed):        {t_metal:.3f} s "
            f"({t_diffrax / t_metal:.1f}x)"
        )
        err = np.max(np.abs(np.stack([got_x, got_y], axis=1) - baseline))
        print(
            f"max abs deviation from diffrax: {err:.2e} "
            "(fixed- vs adaptive-step, f32; not a bug at ~1e-3)"
        )
    except NotImplementedError as e:
        print(f"palladium path pending: {e}")
        print("finish the exercises: uv run pytest tests/ -m exercise")


if __name__ == "__main__":
    main()
