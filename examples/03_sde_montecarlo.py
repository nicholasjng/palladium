"""Example 3: SDE Monte Carlo on the GPU. Hand-written MSL, runs today.

Prices a European call under geometric Brownian motion: N paths x M
Euler-Maruyama steps, one Metal thread per path, RNG generated in-kernel
(counter-based hash + Box-Muller), validated against the Black-Scholes
closed form. No palladium codegen involved: this is the runtime showcase
and the performance bar the Pallas-authored path should meet.

RNG note: pcg_hash is fine for a pricing demo (each (path, step) pair gets
an independent counter); for production-grade statistics swap in Philox,
which is the same counter-based shape with stronger guarantees.
"""

import time

import metal_runtime as mr
import numpy as np
from scipy.stats import norm

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


def main():
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

    p = payoff.to_numpy()
    price = float(np.exp(-R * T) * p.mean())
    stderr = float(np.exp(-R * T) * p.std(ddof=1) / np.sqrt(N_PATHS))
    exact = black_scholes_call(S0, K, T, R, SIGMA)

    print(
        f"GBM European call, {N_PATHS:,} paths x {N_STEPS} steps ({mr.device_name()})"
    )
    print(
        f"  Monte Carlo:   {price:.4f} +/- {stderr:.4f}  in {t_gpu:.3f} s "
        f"({N_PATHS * N_STEPS / t_gpu / 1e9:.2f} Gsteps/s)"
    )
    print(
        f"  Black-Scholes: {exact:.4f}  "
        f"(|error| = {abs(price - exact) / stderr:.1f} standard errors)"
    )
    assert abs(price - exact) < 4 * stderr, "MC price off by >4 sigma"
    print()
    print("Diffrax comparison: solving the same SDE with diffrax on CPU is")
    print("dominated by Brownian-path bookkeeping; try it with Euler +")
    print("UnsafeBrownianPath at N=10_000 and extrapolate, then write the")
    print("Pallas version once in-kernel RNG lands (ROADMAP exercise 8).")


if __name__ == "__main__":
    main()
