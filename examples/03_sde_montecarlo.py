"""Example 3: SDE Monte Carlo on the GPU.

Prices a European call under geometric Brownian motion: N paths x M
Euler-Maruyama steps, one Metal thread per path, RNG generated in-kernel,
validated against the Black-Scholes closed form.

Two versions. `GBM_MSL` is hand-written MSL with a cheap pcg_hash RNG:
the runtime showcase and the performance bar. `pallas_gbm_kernel` is the
same model authored in Pallas, using `jax.random` directly inside the
kernel (Threefry-2x32-20, stretch 8, `test_10_random_bits.py`) instead of
hand-rolled MSL: `random_fold_in` derives an independent key per (path,
step) pair, the same shape `pcg_hash`'s counter does. Threefry is ~20
rounds of add-rotate-xor per draw versus pcg_hash's one; expect the
Pallas version to cost more per step; the point is that it doesn't need
to be hand-written at all, not that it's cheaper.
"""

import time

import jax
import jax.numpy as jnp
import metal_runtime as mr
import numpy as np
from jax.experimental import pallas as pl
from scipy.stats import norm

import palladium

N_PATHS = 1 << 20
N_STEPS = 256
S0, K, T, R, SIGMA = 100.0, 105.0, 1.0, 0.03, 0.2

GBM_MSL = """
#include <metal_stdlib>
using namespace metal;

uint pcg_hash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float uniform01(uint v)
{
    // (hash + 0.5) / 2^32: open interval, safe for log().
    return (float(pcg_hash(v)) + 0.5f) * 2.3283064e-10f;
}

kernel void gbm_paths(
    device float* payoff [[buffer(0)]],
    constant uint& n_steps [[buffer(1)]],
    constant float& s0 [[buffer(2)]],
    constant float& strike [[buffer(3)]],
    constant float& drift_dt [[buffer(4)]],   // (r - sigma^2/2) * dt
    constant float& vol_sqdt [[buffer(5)]],   // sigma * sqrt(dt)
    uint tid [[thread_position_in_grid]])
{
    float s = s0;
    for (uint i = 0; i < n_steps; ++i) {
        uint ctr = tid * n_steps + i;
        float u1 = uniform01(2u * ctr);
        float u2 = uniform01(2u * ctr + 1u);
        float z = sqrt(-2.0f * log(u1)) * cos(6.2831853f * u2);
        s *= exp(drift_dt + vol_sqdt * z);
    }
    payoff[tid] = max(s - strike, 0.0f);
}
"""


def black_scholes_call(s0, k, t, r, sigma):
    d1 = (np.log(s0 / k) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def run_handwritten():
    dt = T / N_STEPS
    kernel = mr.Kernel(GBM_MSL, "gbm_paths")
    payoff = mr.Buffer.zeros([N_PATHS])
    scalars = [
        np.uint32(N_STEPS),
        np.float32(S0),
        np.float32(K),
        np.float32((R - SIGMA**2 / 2) * dt),
        np.float32(SIGMA * np.sqrt(dt)),
    ]
    mr.run(kernel, grid=N_PATHS, buffers=[payoff], scalars=scalars)  # warm-up
    t0 = time.perf_counter()
    mr.run(kernel, grid=N_PATHS, buffers=[payoff], scalars=scalars)
    t_gpu = time.perf_counter() - t0
    return payoff.to_numpy(), t_gpu


def pallas_gbm_kernel(seed_ref, o_ref):
    dt = T / N_STEPS
    drift_dt = (R - SIGMA**2 / 2) * dt
    vol_sqdt = SIGMA * jnp.sqrt(dt)

    tid = pl.program_id(0)
    base = jax.random.wrap_key_data(seed_ref[...], impl="threefry2x32")
    key0 = jax.random.fold_in(base, tid)

    def step(i, s):
        k1 = jax.random.fold_in(key0, 2 * i)
        k2 = jax.random.fold_in(key0, 2 * i + 1)
        # Floored away from 0: jax.random.uniform's range is [0, 1), and
        # log(0) = -inf poisons the whole path (same reason the
        # hand-written kernel above uses an open-interval (hash+0.5)
        # trick instead).
        u1 = jnp.maximum(jax.random.uniform(k1, ()), 1e-7)
        u2 = jax.random.uniform(k2, ())
        z = jnp.sqrt(-2.0 * jnp.log(u1)) * jnp.cos(2.0 * jnp.pi * u2)
        return s * jnp.exp(drift_dt + vol_sqdt * z)

    s = jax.lax.fori_loop(0, N_STEPS, step, S0)
    o_ref[0] = jnp.maximum(s - K, 0.0)


def run_pallas(seed):
    f = palladium.metal_call(
        pallas_gbm_kernel,
        grid=(N_PATHS,),
        in_specs=[pl.BlockSpec((2,), lambda i: (0,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((N_PATHS,), jnp.float32),
    )
    f(seed)  # trace + emit + Metal compile outside the clock
    t0 = time.perf_counter()
    payoff = f(seed)
    t_gpu = time.perf_counter() - t0
    return payoff, t_gpu


def report(label, payoff, t_gpu):
    price = float(np.exp(-R * T) * payoff.mean())
    stderr = float(np.exp(-R * T) * payoff.std(ddof=1) / np.sqrt(N_PATHS))
    exact = black_scholes_call(S0, K, T, R, SIGMA)
    print(
        f"  {label}: {price:.4f} +/- {stderr:.4f}  in {t_gpu:.3f} s "
        f"({N_PATHS * N_STEPS / t_gpu / 1e9:.2f} Gsteps/s), "
        f"{abs(price - exact) / stderr:.1f} sigma from Black-Scholes ({exact:.4f})"
    )
    assert abs(price - exact) < 5 * stderr, f"{label}: MC price off by >5 sigma"


def main():
    print(
        f"GBM European call, {N_PATHS:,} paths x {N_STEPS} steps ({mr.device_name()})"
    )
    p_hand, t_hand = run_handwritten()
    report("hand-written MSL (pcg_hash)", p_hand, t_hand)

    seed = np.asarray(jax.random.key_data(jax.random.key(0)))
    p_pallas, t_pallas = run_pallas(seed)
    report("Pallas-authored  (Threefry)", p_pallas, t_pallas)

    print()
    print(f"  Threefry / pcg_hash slowdown: {t_pallas / t_hand:.1f}x")
    print("  (expected: ~20 add-rotate-xor rounds per draw vs. pcg_hash's one;")
    print("  the payoff is not having hand-written this kernel's MSL at all.)")


if __name__ == "__main__":
    main()
