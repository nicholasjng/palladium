"""Example 4: method-of-lines PDE, the large-state contrast.

Docs: indexed ref access, docs/supported-subset.md (ref indexing).
Gray-Scott reaction-diffusion on a 128x128 grid, integrated with explicit
Euler. The baseline is jax.jit + lax.scan on CPU. The Metal version is a
stencil kernel: one thread per grid point, halo reads from neighbours via
indexed ref access (`x_ref[i, j]`, `test_09_indexed_refs.py`).

A PDE step depends on every thread's neighbours finishing the previous
step first, which a single dispatch cannot guarantee across threadgroups
(no device-wide barrier, only per-threadgroup). So this is one kernel
launch per step with ping-pong buffers, not an in-kernel loop like the
ODE/SDE examples: `metal_runtime.Buffer`/`Batch` used directly rather
than through `metal_call`, so the grid stays device-resident across all
`STEPS` launches instead of round-tripping through NumPy each step.

Threadgroup-memory tiling for the halo reads is a later, separate
optimization; this is plain global-memory indexed access.
"""

import time

import jax
import jax.numpy as jnp
import metal_runtime as mr
import numpy as np
from jax.experimental import pallas as pl

from palladium.dispatch import bind
from palladium.emit import emit_msl
from palladium.trace import trace

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


def gray_scott_kernel(u_ref, v_ref, uo_ref, vo_ref):
    i, j = pl.program_id(0), pl.program_id(1)
    n = SIZE
    ip = jnp.where(i == n - 1, 0, i + 1)
    im = jnp.where(i == 0, n - 1, i - 1)
    jp = jnp.where(j == n - 1, 0, j + 1)
    jm = jnp.where(j == 0, n - 1, j - 1)

    u, v = u_ref[i, j], v_ref[i, j]
    lap_u = u_ref[ip, j] + u_ref[im, j] + u_ref[i, jp] + u_ref[i, jm] - 4.0 * u
    lap_v = v_ref[ip, j] + v_ref[im, j] + v_ref[i, jp] + v_ref[i, jm] - 4.0 * v
    uvv = u * v * v
    uo_ref[i, j] = u + DT * (DU * lap_u - uvv + F * (1.0 - u))
    vo_ref[i, j] = v + DT * (DV * lap_v + uvv - (F + KK) * v)


def run_metal(u0: np.ndarray, v0: np.ndarray):
    full = pl.BlockSpec((SIZE, SIZE), lambda i, j: (0, 0))
    staged = pl.pallas_call(
        gray_scott_kernel,
        grid=(SIZE, SIZE),
        in_specs=[full, full],
        out_specs=(full, full),
        out_shape=(
            jax.ShapeDtypeStruct((SIZE, SIZE), jnp.float32),
            jax.ShapeDtypeStruct((SIZE, SIZE), jnp.float32),
        ),
    )
    spec = trace(staged, jnp.zeros((SIZE, SIZE), jnp.float32), jnp.zeros_like(u0))
    bound = bind(spec, emit_msl(spec))

    bufs = [
        [mr.Buffer(np.ascontiguousarray(u0)), mr.Buffer(np.ascontiguousarray(v0))],
        [
            mr.Buffer.empty([SIZE, SIZE], dtype="float32"),
            mr.Buffer.empty([SIZE, SIZE], dtype="float32"),
        ],
    ]
    t0 = time.perf_counter()
    batch = mr.Batch()
    src = 0
    for _ in range(STEPS):
        dst = 1 - src
        batch.add(bound.kernel, grid=(SIZE, SIZE), buffers=[*bufs[src], *bufs[dst]])
        src = dst
    batch.commit()
    batch.wait()
    t_wall = time.perf_counter() - t0
    return bufs[src][0].to_numpy(), bufs[src][1].to_numpy(), t_wall, batch.gpu_time


def initial_state(size, seed=11):
    rng = np.random.default_rng(seed)
    u = np.ones((size, size), np.float32)
    v = np.zeros((size, size), np.float32)
    m = size // 2
    u[m - 8 : m + 8, m - 8 : m + 8] = 0.5
    v[m - 8 : m + 8, m - 8 : m + 8] = 0.25
    u += 0.02 * rng.standard_normal((size, size)).astype(np.float32)
    return u, v


def main():
    u0, v0 = initial_state(SIZE)
    u0j, v0j = jnp.asarray(u0), jnp.asarray(v0)
    run_baseline(u0j, v0j)[0].block_until_ready()  # compile outside the clock
    t0 = time.perf_counter()
    u, v = run_baseline(u0j, v0j)
    u.block_until_ready()
    t_cpu = time.perf_counter() - t0
    print(f"Gray-Scott {SIZE}x{SIZE}, {STEPS} Euler steps")
    print(
        f"  jax.jit + scan (CPU): {t_cpu:.3f} s, pattern energy {float(jnp.sum(v)):.1f}"
    )

    run_metal(u0, v0)  # trace + emit + Metal compile outside the clock
    _, got_v, t_wall, t_gpu = run_metal(u0, v0)
    print(
        f"  Metal stencil kernel: {t_wall:.3f} s wall ({t_gpu:.3f} s GPU-side), "
        f"pattern energy {float(np.sum(got_v)):.1f}"
    )
    err = np.max(np.abs(got_v - np.asarray(v)))
    print(f"  max abs deviation from CPU baseline: {err:.2e}")


if __name__ == "__main__":
    main()
