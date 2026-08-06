"""Example 4: method-of-lines PDE — the large-state contrast.

Gray-Scott reaction-diffusion on a 128x128 grid, integrated with explicit
Euler. The baseline (runs today) is jax.jit + lax.scan on CPU. The Metal
version is a stencil kernel: one thread per grid point, halo reads from
neighbours, threadgroup-memory tiling for the reads — which needs *indexed*
ref access (`x_ref[i, j]`), the ROADMAP's stretch exercise 6. Until then
this script prints the baseline and stops.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

SIZE = 128
STEPS = 2000
DU, DV, F, KK, DT = 0.16, 0.08, 0.035, 0.065, 1.0


def laplacian(z):
    return (
        jnp.roll(z, 1, 0)
        + jnp.roll(z, -1, 0)
        + jnp.roll(z, 1, 1)
        + jnp.roll(z, -1, 1)
        - 4.0 * z
    )


@jax.jit
def run_baseline(u, v):
    def step(carry, _):
        u, v = carry
        uvv = u * v * v
        u = u + DT * (DU * laplacian(u) - uvv + F * (1.0 - u))
        v = v + DT * (DV * laplacian(v) + uvv - (F + KK) * v)
        return (u, v), None

    (u, v), _ = jax.lax.scan(step, (u, v), None, length=STEPS)
    return u, v


def initial_state(size, seed=11):
    rng = np.random.default_rng(seed)
    u = np.ones((size, size), np.float32)
    v = np.zeros((size, size), np.float32)
    m = size // 2
    u[m - 8 : m + 8, m - 8 : m + 8] = 0.5
    v[m - 8 : m + 8, m - 8 : m + 8] = 0.25
    u += 0.02 * rng.standard_normal((size, size)).astype(np.float32)
    return jnp.asarray(u), jnp.asarray(v)


def main():
    u0, v0 = initial_state(SIZE)
    run_baseline(u0, v0)[0].block_until_ready()  # compile outside the clock
    t0 = time.perf_counter()
    u, v = run_baseline(u0, v0)
    u.block_until_ready()
    t_cpu = time.perf_counter() - t0
    print(f"Gray-Scott {SIZE}x{SIZE}, {STEPS} Euler steps")
    print(
        f"  jax.jit + scan (CPU): {t_cpu:.3f} s, pattern energy {float(jnp.sum(v)):.1f}"
    )
    print()
    print("Metal stencil version: pending indexed ref access (x_ref[i, j]),")
    print("ROADMAP stretch exercise 6 — one thread per cell, halo reads,")
    print("threadgroup-memory tiling via run(..., threadgroup_memory=[...]).")


if __name__ == "__main__":
    main()
