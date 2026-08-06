"""Example 5: does compensated arithmetic (df32) buy real accuracy here.

Stretch 9. Same RK4 Lotka-Volterra capstone as example 1, three ways:
FAST (the default), SAFE (same float32 kernel, reassociation off), and
DF32 (metal-runtime's df32 prelude: float32x2 compensated arithmetic,
hand-written MSL since the Pallas/jnp frontend has no df32 dtype to trace
through). All three measured against a true float64 NumPy RK4 reference,
not against each other: the question this answers is "how far off is
each variant from what actually happened," not "does GPU match CPU."
"""

import time

import jax
import jax.numpy as jnp
import metal_runtime as mr
import metal_runtime.df32 as mrdf32
import numpy as np
from jax.experimental import pallas as pl

import palladium

DT, STEPS = 0.01, 500
N = 100_000

DF32_KERNEL = """
inline df32 operator+(df32 a, df32 b) { return df_add(a, b); }
inline df32 operator-(df32 a, df32 b) { return df_sub(a, b); }
inline df32 operator*(df32 a, df32 b) { return df_mul(a, b); }

struct State { df32 x, y; };

inline State rhs(State s, df32 a, df32 b, df32 c, df32 d) {
    df32 xy = s.x * s.y;
    return State{ a * s.x - b * xy, c * xy - d * s.y };
}

inline State advance(State s, State k, df32 h) {
    return State{ s.x + h * k.x, s.y + h * k.y };
}

kernel void lv_rk4_df32(
    device const float* x0 [[buffer(0)]],
    device const float* y0 [[buffer(1)]],
    device const float* a_ [[buffer(2)]],
    device const float* b_ [[buffer(3)]],
    device const float* c_ [[buffer(4)]],
    device const float* d_ [[buffer(5)]],
    device float* xo [[buffer(6)]],
    device float* yo [[buffer(7)]],
    constant float& dt [[buffer(8)]],
    constant uint& steps [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    df32 a = df_from_float(a_[tid]);
    df32 b = df_from_float(b_[tid]);
    df32 c = df_from_float(c_[tid]);
    df32 d = df_from_float(d_[tid]);
    df32 half_dt = df_from_float(0.5f * dt);
    df32 full_dt = df_from_float(dt);
    df32 sixth_dt = df_from_float(dt / 6.0f);
    df32 two = df_from_float(2.0f);

    State s = { df_from_float(x0[tid]), df_from_float(y0[tid]) };
    for (uint i = 0; i < steps; ++i) {
        State k1 = rhs(s, a, b, c, d);
        State k2 = rhs(advance(s, k1, half_dt), a, b, c, d);
        State k3 = rhs(advance(s, k2, half_dt), a, b, c, d);
        State k4 = rhs(advance(s, k3, full_dt), a, b, c, d);
        State sum = {
            k1.x + two * k2.x + two * k3.x + k4.x,
            k1.y + two * k2.y + two * k3.y + k4.y,
        };
        s = advance(s, sum, sixth_dt);
    }
    xo[tid] = df_to_float(s.x);
    yo[tid] = df_to_float(s.y);
}
"""


def ensemble(n, seed=17):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(0.8, 1.2, n).astype(np.float32),
        rng.uniform(0.8, 1.2, n).astype(np.float32),
        rng.uniform(0.9, 1.3, n).astype(np.float32),
        rng.uniform(0.3, 0.5, n).astype(np.float32),
        rng.uniform(0.08, 0.12, n).astype(np.float32),
        rng.uniform(0.3, 0.5, n).astype(np.float32),
    )


def float64_reference(x0, y0, a, b, c, d):
    x, y = x0.astype(np.float64), y0.astype(np.float64)
    a, b, c, d = (v.astype(np.float64) for v in (a, b, c, d))

    def rhs(x, y):
        return a * x - b * x * y, c * x * y - d * y

    for _ in range(STEPS):
        k1x, k1y = rhs(x, y)
        k2x, k2y = rhs(x + 0.5 * DT * k1x, y + 0.5 * DT * k1y)
        k3x, k3y = rhs(x + 0.5 * DT * k2x, y + 0.5 * DT * k2y)
        k4x, k4y = rhs(x + DT * k3x, y + DT * k3y)
        x = x + DT / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        y = y + DT / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
    return x, y


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


def run_palladium(math_mode, args):
    spec_1 = pl.BlockSpec((1,), lambda i: (i,))
    f = palladium.metal_call(
        lv_kernel,
        math_mode=math_mode,
        grid=(N,),
        in_specs=[spec_1] * 6,
        out_specs=(spec_1, spec_1),
        out_shape=(
            jax.ShapeDtypeStruct((N,), jnp.float32),
            jax.ShapeDtypeStruct((N,), jnp.float32),
        ),
    )
    f(*args)  # trace + emit + Metal compile outside the clock
    t0 = time.perf_counter()
    got_x, got_y = f(*args)
    return got_x, got_y, time.perf_counter() - t0


def run_df32(args):
    x0, y0, a, b, c, d = args
    kernel = mrdf32.kernel(DF32_KERNEL, "lv_rk4_df32")
    bufs = [mr.Buffer(v) for v in (x0, y0, a, b, c, d)]
    xo, yo = mr.Buffer.empty([N]), mr.Buffer.empty([N])
    scalars = [np.float32(DT), np.uint32(STEPS)]

    def launch():
        mr.run(kernel, grid=N, buffers=[*bufs, xo, yo], scalars=scalars)

    launch()  # compile outside the clock
    t0 = time.perf_counter()
    launch()
    return xo.to_numpy(), yo.to_numpy(), time.perf_counter() - t0


def report(label, got_x, got_y, want_x, want_y, t):
    err = max(np.max(np.abs(got_x - want_x)), np.max(np.abs(got_y - want_y)))
    print(f"  {label:<22} {t:.3f} s   max abs error vs float64: {err:.2e}")
    return err


def main():
    args = ensemble(N)
    print(f"Lotka-Volterra ensemble: N={N}, RK4, {STEPS} steps of dt={DT}")
    print("(measuring each variant against a true float64 NumPy reference,")
    print(" not against each other)")

    want_x, want_y = float64_reference(*args)

    got_x, got_y, t = run_palladium(mr.MathMode.FAST, args)
    err_fast = report("float32, FAST:", got_x, got_y, want_x, want_y, t)

    got_x, got_y, t = run_palladium(mr.MathMode.SAFE, args)
    err_safe = report("float32, SAFE:", got_x, got_y, want_x, want_y, t)

    got_x, got_y, t = run_df32(args)
    err_df32 = report("df32, SAFE:", got_x, got_y, want_x, want_y, t)

    print()
    print(
        f"  df32 is {err_fast / err_df32:.0f}x closer to float64 than FAST, "
        f"{err_safe / err_df32:.0f}x closer than SAFE."
    )
    assert err_df32 < err_fast, "df32 should be more accurate than plain FAST float32"


if __name__ == "__main__":
    main()
