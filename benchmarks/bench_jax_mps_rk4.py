"""Compare an RK4 Pallas custom call with ``jit(vmap(scan))`` on jax-mps.

Run from the jax-mps checkout so its plugin is selected before JAX starts::

    JAX_PLATFORMS=mps,cpu env -u VIRTUAL_ENV \
      uv run --project ../jax-mps python ../palladium/benchmarks/bench_jax_mps_rk4.py

The two MPS paths solve exactly the same fixed-step Lotka--Volterra RK4
problem.  The Palladium path is a single Pallas-generated Metal dispatch;
the baseline is the idiomatic JAX ``jit(vmap(scan(...)))`` expression.  Both
return a downstream JAX reduction, demonstrating that the custom call stays
inside the jax-mps graph.
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
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


def invoke_and_wait(fn, args):
    return jax.block_until_ready(fn(*args))


def first_call_ms(fn, args) -> tuple[float, jax.Array]:
    start = time.perf_counter()
    result = invoke_and_wait(fn, args)
    return (time.perf_counter() - start) * 1_000, result


def interleaved_ms(left, right, args, repeats: int) -> tuple[list[float], list[float]]:
    left_samples, right_samples = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        invoke_and_wait(left, args)
        left_samples.append((time.perf_counter() - start) * 1_000)
        start = time.perf_counter()
        invoke_and_wait(right, args)
        right_samples.append((time.perf_counter() - start) * 1_000)
    return left_samples, right_samples


def timed_samples(fn, args, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        invoke_and_wait(fn, args)
        samples.append((time.perf_counter() - start) * 1_000)
    return samples


def report(name: str, samples: list[float]) -> None:
    print(
        f"{name:28} median {statistics.median(samples):8.3f} ms  "
        f"min {min(samples):8.3f} ms  max {max(samples):8.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()

    try:
        mps = jax.devices("mps")[0]
    except RuntimeError as error:
        raise SystemExit(
            "No jax-mps device. Start with JAX_PLATFORMS=mps,cpu."
        ) from error
    cpu = jax.devices("cpu")[0]
    host_args = ensemble(args.n)

    print(f"JAX {jax.__version__}; MPS device: {mps}; CPU device: {cpu}")
    print(f"Lotka--Volterra: N={args.n:,}, RK4 fixed step, {args.steps} steps, dt={DT}")

    with jax.default_device(mps):
        mps_args = tuple(jax.device_put(x, mps) for x in host_args)
        palladium_solver = make_pallas_solver(args.n, args.steps)
        jax_mps_solver = make_jax_solver(args.steps)
        palladium_first, palladium_out = first_call_ms(palladium_solver, mps_args)
        baseline_first, baseline_out = first_call_ms(jax_mps_solver, mps_args)
        np.testing.assert_allclose(
            np.asarray(palladium_out), np.asarray(baseline_out), rtol=2e-5, atol=2e-3
        )
        for _ in range(3):
            invoke_and_wait(palladium_solver, mps_args)
            invoke_and_wait(jax_mps_solver, mps_args)
        palladium_samples, baseline_samples = interleaved_ms(
            palladium_solver, jax_mps_solver, mps_args, args.repeats
        )

    with jax.default_device(cpu):
        cpu_args = tuple(jax.device_put(x, cpu) for x in host_args)
        cpu_solver = make_jax_solver(args.steps)
        cpu_first, _ = first_call_ms(cpu_solver, cpu_args)
        for _ in range(3):
            invoke_and_wait(cpu_solver, cpu_args)
        cpu_samples = timed_samples(cpu_solver, cpu_args, args.repeats)

    print("\nFirst call (trace/lower/compile + execute):")
    print(f"  Palladium custom call on MPS: {palladium_first:.1f} ms")
    print(f"  jit(vmap(scan)) on MPS:       {baseline_first:.1f} ms")
    print(f"  jit(vmap(scan)) on CPU:       {cpu_first:.1f} ms")
    print("\nWarm synchronized executions:")
    report("Palladium custom call (MPS)", palladium_samples)
    report("jit(vmap(scan)) (MPS)", baseline_samples)
    report("jit(vmap(scan)) (CPU)", cpu_samples)
    print(
        f"\nMPS custom-call speedup over MPS baseline: "
        f"{statistics.median(baseline_samples) / statistics.median(palladium_samples):.2f}x"
    )


if __name__ == "__main__":
    main()
