"""Example 2: the lockstep tax, or why per-thread adaptivity is the prize.

Docs: divergent while-loops, docs/supported-subset.md (control flow).
vmap over an adaptive Diffrax solve forces the whole batch into lockstep:
every trajectory takes (and rejects) the steps its worst neighbour needs.
Measures that tax on the CPU by salting a mild Van der Pol ensemble with
a few stiff members, then runs the same ensembles through palladium's
per-thread adaptive kernel (Bogacki-Shampine 3(2), FSAL, PI controller;
`pcoeff=0.4, icoeff=0.3`, diffrax's own suggestion for "moderate
difficulty" problems) to show the tax doesn't apply there. Full recipe
in tests/test_08_adaptive.py.
"""

import time

import diffrax
import jax
import jax.numpy as jnp
import metal_runtime as mr
import numpy as np
from jax.experimental import pallas as pl

import palladium

N = 2048
STIFF_FRACTION = 0.02  # 2% of trajectories get a large mu
T1 = 10.0
X0, V0, DT0 = 2.0, 0.0, 0.01
MAX_STEPS = 6000
RTOL, ATOL = 1e-5, 1e-7
PCOEFF, ICOEFF, ERROR_ORDER = 0.4, 0.3, 3
SAFETY, FACTORMIN, FACTORMAX = 0.9, 0.2, 10.0


def solve_batch(mu, solver=None):
    solver = solver or diffrax.Tsit5()

    def rhs(t, u, mu):
        x, v = u
        return jnp.stack([v, mu * (1.0 - x * x) * v - x])

    @jax.jit
    @jax.vmap
    def solve(mu):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            solver,
            t0=0.0,
            t1=T1,
            dt0=DT0,
            y0=jnp.array([X0, V0]),
            args=mu,
            stepsize_controller=diffrax.PIDController(rtol=RTOL, atol=ATOL),
            max_steps=1_000_000,
        )
        # num_steps = accepted + rejected attempts; num_accepted_steps is the
        # one comparable to our `real` counter (accepted only, pre-T1).
        return sol.ys[-1], sol.stats["num_steps"], sol.stats["num_accepted_steps"]

    out, steps, accepted = solve(mu)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    out, steps, accepted = solve(mu)
    jax.block_until_ready(out)
    return time.perf_counter() - t0, np.asarray(steps), np.asarray(accepted)


def palladium_solver(n, pcoeff=PCOEFF, icoeff=ICOEFF):
    def vdp_adaptive_kernel(
        x0_ref,
        v0_ref,
        mu_ref,
        to_ref,
        xo_ref,
        vo_ref,
        steps_ref,
        rejected_ref,
        real_ref,
    ):
        mu = mu_ref[...]

        def rhs(x, v):
            return v, mu * (1.0 - x**2) * v - x

        def step(_, carry):
            t, dt, x, v, k1x, k1v, prev_inv_err, steps, rejected, real = carry
            was_real = t < T1  # before the clamp: only true pre-completion attempts
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

            coeff1 = (icoeff + pcoeff) / ERROR_ORDER
            coeff2 = -pcoeff / ERROR_ORDER
            inv_err = 1 / err_norm
            factor = SAFETY * inv_err**coeff1 * prev_inv_err**coeff2
            prev_inv_err = jnp.where(accept, inv_err, prev_inv_err)

            dt = dt * jnp.clip(factor, FACTORMIN, FACTORMAX)
            steps = steps + jnp.where(accept, 1.0, 0.0)
            rejected = rejected + jnp.where(accept, 0.0, 1.0)
            # was_real & accept would stage a boolean `and`, which has no
            # emitter rule; nest the wheres instead (both branches numeric).
            real = real + jnp.where(was_real, jnp.where(accept, 1.0, 0.0), 0.0)
            return (
                t2,
                dt,
                x2,
                v2,
                next_k1x,
                next_k1v,
                prev_inv_err,
                steps,
                rejected,
                real,
            )

        t0 = jnp.zeros_like(x0_ref[...])
        dt0 = jnp.full_like(x0_ref[...], DT0)
        steps0 = jnp.zeros_like(x0_ref[...])
        rejected0 = jnp.zeros_like(x0_ref[...])
        real0 = jnp.zeros_like(x0_ref[...])
        k1x_0, k1v_0 = rhs(x0_ref[...], v0_ref[...])
        inv_err0 = jnp.ones_like(x0_ref[...])
        t, _, x, v, _, _, _, steps, rejected, real = jax.lax.fori_loop(
            0,
            MAX_STEPS,
            step,
            (
                t0,
                dt0,
                x0_ref[...],
                v0_ref[...],
                k1x_0,
                k1v_0,
                inv_err0,
                steps0,
                rejected0,
                real0,
            ),
        )
        to_ref[...] = t
        xo_ref[...] = x
        vo_ref[...] = v
        steps_ref[...] = steps
        rejected_ref[...] = rejected
        real_ref[...] = real

    spec_1 = pl.BlockSpec((1,), lambda i: (i,))
    return palladium.metal_call(
        vdp_adaptive_kernel,
        grid=(n,),
        in_specs=[spec_1] * 3,
        out_specs=(spec_1,) * 6,
        out_shape=tuple(jax.ShapeDtypeStruct((n,), jnp.float32) for _ in range(6)),
        math_mode=mr.MathMode.RELAXED,
    )


def main():
    rng = np.random.default_rng(3)
    mild = rng.uniform(0.5, 2.0, N).astype(np.float32)

    mixed = mild.copy()
    k = int(N * STIFF_FRACTION)
    mixed[rng.choice(N, k, replace=False)] = rng.uniform(300.0, 500.0, k)

    t_mild, steps_mild, _ = solve_batch(jnp.asarray(mild))
    t_mixed, steps_mixed, _ = solve_batch(jnp.asarray(mixed))

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
    print("in a Metal kernel pays only where stiffness actually is:")
    print()

    solve = palladium_solver(N)
    x0 = np.full(N, X0, dtype=np.float32)
    v0 = np.full(N, V0, dtype=np.float32)

    solve(x0, v0, mild)  # trace + emit + Metal compile outside the clock
    t0 = time.perf_counter()
    _, _, _, steps_mild_gpu, _, real_mild_gpu = solve(x0, v0, mild)
    t_mild_gpu = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, _, _, steps_mixed_gpu, rejected_mixed_gpu, real_mixed_gpu = solve(x0, v0, mixed)
    t_mixed_gpu = time.perf_counter() - t0

    print("palladium (Apple GPU, per-thread BS3(2) + FSAL + PI controller):")
    print(
        f"  all mild:                             {t_mild_gpu:.3f} s, "
        f"median steps {int(np.median(steps_mild_gpu))}"
    )
    print(
        f"  {k} stiff members:                    {t_mixed_gpu:.3f} s, "
        f"median steps {int(np.median(steps_mixed_gpu))}"
    )
    print(
        f"  lockstep tax on GPU: {t_mixed_gpu / t_mild_gpu:.2f}x "
        f"(vs {t_mixed / t_mild:.1f}x for vmap(Tsit5+PID) on CPU)"
    )
    print(
        f"  step-count std across the stiff ensemble: "
        f"{np.std(steps_mixed_gpu):.1f}, adaptivity happening per-thread, "
        "observably"
    )
    print(
        f"  rejected attempts (mixed ensemble): mean "
        f"{np.mean(rejected_mixed_gpu):.1f}, max {np.max(rejected_mixed_gpu):.0f} "
        "(`steps` above conflates real progress with post-T1 stall padding; "
        "`rejected` doesn't)"
    )

    print()
    print("Controller-quality check: same kernel, PI vs. I-only (pcoeff=0),")
    print("on the mixed ensemble. Fewer rejections means less wasted work:")
    i_only = palladium_solver(N, pcoeff=0.0, icoeff=1.0)
    _, _, _, _, rejected_i, _ = i_only(x0, v0, mixed)
    print(
        f"  I-only (pcoeff=0.0): rejected mean {np.mean(rejected_i):.1f}, "
        f"max {np.max(rejected_i):.0f}"
    )
    print(
        f"  PI (pcoeff={PCOEFF}):     rejected mean "
        f"{np.mean(rejected_mixed_gpu):.1f}, max {np.max(rejected_mixed_gpu):.0f}"
    )

    print()
    print("Order-matched step-count check: Bosh3 (our method) vs. Bosh3 on")
    print("diffrax, same rtol/atol. Both sides count *accepted* steps only")
    print("(diffrax's num_accepted_steps; our `real`, which also excludes")
    print("post-T1 stall padding) for a genuine apples-to-apples:")
    t_mild_b3, _, acc_mild_b3 = solve_batch(jnp.asarray(mild), diffrax.Bosh3())
    t_mixed_b3, _, acc_mixed_b3 = solve_batch(jnp.asarray(mixed), diffrax.Bosh3())
    print(
        f"  diffrax   (Bosh3+I, CPU):  mild median {int(np.median(acc_mild_b3))} "
        f"steps, {t_mild_b3:.3f} s | mixed median "
        f"{int(np.median(acc_mixed_b3))} steps, {t_mixed_b3:.3f} s"
    )
    print(
        f"  palladium (Bosh3+PI, GPU): mild median "
        f"{int(np.median(real_mild_gpu))} steps, {t_mild_gpu:.3f} s | mixed "
        f"median {int(np.median(real_mixed_gpu))} steps, {t_mixed_gpu:.3f} s"
    )
    print(
        "  diffrax's PIDController defaults to pcoeff=0 (I-only, same family"
        " as ours): the step-count gap left is safety-margin/coefficient/"
        "implementation detail, not method order or controller class."
    )


if __name__ == "__main__":
    main()
