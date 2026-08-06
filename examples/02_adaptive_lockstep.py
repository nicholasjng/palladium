"""Example 2: the lockstep tax — why per-thread adaptivity is the prize.

`vmap` over an adaptive Diffrax solve forces the whole batch into lockstep:
every trajectory takes (and rejects) the steps its worst neighbour needs.
This script MEASURES that tax on the CPU backend by salting a mild Van der
Pol ensemble with a few stiff members and timing the same batch with and
without them.

It runs today, with no palladium code: it is the motivation. The payoff —
each Metal thread running its own PI-controlled adaptive loop, immune to
its neighbours — is ROADMAP exercise 7, and this script is where it will
plug in.
"""

import time

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

N = 2048
STIFF_FRACTION = 0.02  # 2% of trajectories get a large mu
T1 = 10.0


def solve_batch(mu):
    def rhs(t, u, mu):
        x, v = u
        return jnp.stack([v, mu * (1.0 - x * x) * v - x])

    @jax.jit
    @jax.vmap
    def solve(mu):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            diffrax.Tsit5(),
            t0=0.0,
            t1=T1,
            dt0=0.01,
            y0=jnp.array([2.0, 0.0]),
            args=mu,
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
            max_steps=1_000_000,
        )
        return sol.ys[-1], sol.stats["num_steps"]

    out, steps = solve(mu)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    out, steps = solve(mu)
    jax.block_until_ready(out)
    return time.perf_counter() - t0, np.asarray(steps)


def main():
    rng = np.random.default_rng(3)
    mild = rng.uniform(0.5, 2.0, N).astype(np.float32)

    mixed = mild.copy()
    k = int(N * STIFF_FRACTION)
    mixed[rng.choice(N, k, replace=False)] = rng.uniform(300.0, 500.0, k)

    t_mild, steps_mild = solve_batch(jnp.asarray(mild))
    t_mixed, steps_mixed = solve_batch(jnp.asarray(mixed))

    print(f"Van der Pol ensemble, N={N}, vmap(Tsit5+PID) on CPU")
    print(
        f"  all mild (mu ~ U[0.5, 2]):          {t_mild:.3f} s, "
        f"median steps {int(np.median(steps_mild))}"
    )
    print(
        f"  {k} stiff members (mu ~ U[300, 500]): {t_mixed:.3f} s, "
        f"median steps {int(np.median(steps_mixed))}"
    )
    print(
        f"  lockstep tax: {t_mixed / t_mild:.1f}x wall clock for "
        f"{100 * STIFF_FRACTION:.0f}% of the batch"
    )
    print()
    print("Every lane pays for the stiff lanes' steps: vmap synchronizes the")
    print("step-size controller across the batch. A per-thread adaptive loop")
    print("in a Metal kernel pays only where stiffness actually is —")
    print("ROADMAP exercise 7 builds it on top of the exercise-4 loop rule.")


if __name__ == "__main__":
    main()
