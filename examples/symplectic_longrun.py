"""Example 6: long-horizon symplectic integration, and what limits it.

Docs: math modes, docs/performance.md (math mode).
The Kepler two-body problem, q'' = -q/|q|^3, integrated for millions of
steps. Initial conditions are set at perihelion of a unit-semi-major-axis
orbit, so the invariants are known in closed form -- E = -1/2 and
L = sqrt(1 - e^2), for every member of the ensemble. Accuracy therefore
needs no reference integration: the exact answer is a constant.

Two questions, in order.

Part A (CPU, float64): why symplectic. A symplectic integrator conserves
a *modified* Hamiltonian exactly, so its energy error oscillates inside a
bound fixed by the step size and never walks away, at any horizon. A
non-symplectic method of the same order drifts secularly. Velocity-Verlet
vs Heun (both order 2), plus a Yoshida triple-jump composition of Verlet
(order 4, still symplectic) to show the bound falling without the
qualitative behaviour changing.

Part B (GPU): what is left once truncation error is bounded. Nothing in
Part A's symplectic column drifts in exact arithmetic, so whatever drift
survives is round-off: each step commits an O(eps) rounding error to the
state, and those accumulate as a random walk, error ~ eps * sqrt(steps).
Verlet in float32 (FAST and SAFE) against Verlet in df32 (float32x2
compensated arithmetic, hand-written MSL: the Pallas/jnp frontend has no
df32 dtype to trace through). The float32 lines leave the truncation
floor and climb with slope 1/2; the df32 line stays on it. That is the
point of the example: the compensated variant is limited by the
integrator, the plain one by the hardware.
"""

import time

import jax
import jax.numpy as jnp
import metal_runtime as mr
import metal_runtime.df32 as mrdf32
import numpy as np
from jax.experimental import pallas as pl

import palladium

H = 0.002  # ~3140 steps per orbit; the truncation floor is O(H^2)
N = 1024  # ensemble members, one Metal thread each

# Long runs are cut into chunks so no single dispatch trips the macOS GPU
# watchdog (a multi-second kernel is killed as "impacting interactivity").
# Chunking is free of accuracy consequences only because the state round
# trips exactly: float32 state is float32 on the way out, and the df32
# kernel hands back both limbs rather than collapsing to float32, which
# would throw away precisely the compensation being measured.
CHUNK = 50_000
CHUNKS = 100  # 5e6 steps, ~1600 orbits
MARKS = (1, 2, 5, 10, 20, 50, 100)  # chunk indices to print a row for

DF32_KERNEL = """
inline df32 operator+(df32 a, df32 b) { return df_add(a, b); }
inline df32 operator*(df32 a, df32 b) { return df_mul(a, b); }

struct Vec2 { df32 x, y; };

inline Vec2 accel(Vec2 q) {
    df32 r2 = q.x * q.x + q.y * q.y;
    df32 inv_r3 = df_div(df_from_float(-1.0f), r2 * df_sqrt(r2));
    return Vec2{ inv_r3 * q.x, inv_r3 * q.y };
}

inline Vec2 axpy(Vec2 v, Vec2 w, df32 s) {
    return Vec2{ v.x + s * w.x, v.y + s * w.y };
}

kernel void kepler_verlet_df32(
    device const float* qx_hi [[buffer(0)]],
    device const float* qx_lo [[buffer(1)]],
    device const float* qy_hi [[buffer(2)]],
    device const float* qy_lo [[buffer(3)]],
    device const float* px_hi [[buffer(4)]],
    device const float* px_lo [[buffer(5)]],
    device const float* py_hi [[buffer(6)]],
    device const float* py_lo [[buffer(7)]],
    device float* o_qx_hi [[buffer(8)]],
    device float* o_qx_lo [[buffer(9)]],
    device float* o_qy_hi [[buffer(10)]],
    device float* o_qy_lo [[buffer(11)]],
    device float* o_px_hi [[buffer(12)]],
    device float* o_px_lo [[buffer(13)]],
    device float* o_py_hi [[buffer(14)]],
    device float* o_py_lo [[buffer(15)]],
    constant float& h [[buffer(16)]],
    constant uint& steps [[buffer(17)]],
    uint tid [[thread_position_in_grid]])
{
    df32 full_h = df_from_float(h);
    df32 half_h = df_from_float(0.5f * h);

    Vec2 q = { df32{qx_hi[tid], qx_lo[tid]}, df32{qy_hi[tid], qy_lo[tid]} };
    Vec2 p = { df32{px_hi[tid], px_lo[tid]}, df32{py_hi[tid], py_lo[tid]} };

    for (uint i = 0; i < steps; ++i) {   // kick, drift, kick
        p = axpy(p, accel(q), half_h);
        q = axpy(q, p, full_h);
        p = axpy(p, accel(q), half_h);
    }
    o_qx_hi[tid] = q.x.hi;  o_qx_lo[tid] = q.x.lo;
    o_qy_hi[tid] = q.y.hi;  o_qy_lo[tid] = q.y.lo;
    o_px_hi[tid] = p.x.hi;  o_px_lo[tid] = p.x.lo;
    o_py_hi[tid] = p.y.hi;  o_py_lo[tid] = p.y.lo;
}
"""


def ensemble(n, seed=17):
    """Perihelion start on a unit-semi-major-axis Kepler orbit:
    q = (1 - e, 0), p = (0, sqrt((1 + e) / (1 - e))), for which
    E = -1/2 and L = sqrt(1 - e^2) hold exactly, for every e."""
    e = np.linspace(0.0, 0.4, n)
    return (
        (1.0 - e).astype(np.float32),
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.sqrt((1.0 + e) / (1.0 - e)).astype(np.float32),
    )


def energy(qx, qy, px, py):
    """Evaluated in float64 whatever the integration ran in: this
    measures the state's error, not the measurement's."""
    q = np.stack([qx, qy]).astype(np.float64)
    p = np.stack([px, py]).astype(np.float64)
    return 0.5 * np.sum(p * p, axis=0) - 1.0 / np.sqrt(np.sum(q * q, axis=0))


# --- Part A: CPU, float64, integrator vs integrator ----------------------


def accel_np(q):
    r2 = np.sum(q * q, axis=0)
    return -q / (r2 * np.sqrt(r2))


def verlet_step(q, p, h):
    p = p + 0.5 * h * accel_np(q)
    q = q + h * p
    return q, p + 0.5 * h * accel_np(q)


def heun_step(q, p, h):
    """Order 2, explicit, and not symplectic: the control."""
    kq1, kp1 = p, accel_np(q)
    kq2, kp2 = p + h * kp1, accel_np(q + h * kq1)
    return q + 0.5 * h * (kq1 + kq2), p + 0.5 * h * (kp1 + kp2)


def yoshida4_step(q, p, h):
    """Triple-jump composition of Verlet: order 4, still symplectic."""
    cbrt2 = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - cbrt2)
    for w in (w1, -cbrt2 * w1, w1):
        q, p = verlet_step(q, p, w * h)
    return q, p


def envelope(step_fn, h, orbits):
    """Per-orbit max |E - E_exact| for one e=0.3 orbit in float64.

    The envelope, not a point sample: "bounded" is a statement about how
    large the oscillation gets, and a sample at one instant can land
    anywhere inside it.
    """
    q = np.array([[0.7], [0.0]])
    p = np.array([[0.0], [np.sqrt(1.3 / 0.7)]])
    per_orbit = int(2.0 * np.pi / h)
    peaks = []
    for _ in range(orbits):
        peak = 0.0
        for _ in range(per_orbit):
            q, p = step_fn(q, p, h)
            peak = max(peak, abs(energy(*q, *p)[0] + 0.5))
        peaks.append(peak)
    return np.array(peaks)


def part_a():
    print("Part A -- float64 on the CPU, one e=0.3 orbit")
    print("          per-orbit peak |E + 1/2|, first orbit vs last")
    print()
    print(f"  {'method':<26}{'h':>8}{'orbits':>8}{'first':>12}{'last':>12}{'growth':>9}")
    for label, fn, h, orbits in (
        ("velocity-Verlet (sympl.)", verlet_step, 0.008, 200),
        ("Heun (not sympl.)", heun_step, 0.008, 200),
        ("velocity-Verlet (sympl.)", verlet_step, 0.004, 20),
        ("velocity-Verlet (sympl.)", verlet_step, 0.002, 20),
        ("Yoshida-4 (sympl.)", yoshida4_step, 0.008, 20),
    ):
        ev = envelope(fn, h, orbits)
        print(f"  {label:<26}{h:>8}{orbits:>8}{ev[0]:>12.3e}{ev[-1]:>12.3e}{ev[-1] / ev[0]:>8.2f}x")
    print()
    print("  Over the same 200 orbits at the same step size, Verlet's peak is")
    print("  unchanged and Heun's is not: that is the whole distinction, and no")
    print("  amount of added precision removes Heun's drift. Halving h divides")
    print("  Verlet's bound by 4 (order 2); Yoshida-4 buys a much smaller bound")
    print("  at the same h, not a different behaviour in time.")


# --- Part B: GPU, precision vs precision --------------------------------


def verlet_kernel(qx_ref, qy_ref, px_ref, py_ref, qxo, qyo, pxo, pyo):
    def accel(qx, qy):
        r2 = qx * qx + qy * qy
        inv_r3 = -1.0 / (r2 * jnp.sqrt(r2))
        return inv_r3 * qx, inv_r3 * qy

    def step(_, carry):  # kick, drift, kick
        qx, qy, px, py = carry
        ax, ay = accel(qx, qy)
        px, py = px + 0.5 * H * ax, py + 0.5 * H * ay
        qx, qy = qx + H * px, qy + H * py
        ax, ay = accel(qx, qy)
        return qx, qy, px + 0.5 * H * ax, py + 0.5 * H * ay

    carry = (qx_ref[...], qy_ref[...], px_ref[...], py_ref[...])
    qxo[...], qyo[...], pxo[...], pyo[...] = jax.lax.fori_loop(0, CHUNK, step, carry)


def palladium_run(math_mode, state0, on_mark):
    spec_1 = pl.BlockSpec((1,), lambda i: (i,))
    out = jax.ShapeDtypeStruct((N,), jnp.float32)
    f = palladium.metal_call(
        verlet_kernel,
        math_mode=math_mode,
        grid=(N,),
        in_specs=[spec_1] * 4,
        out_specs=(spec_1,) * 4,
        out_shape=(out,) * 4,
    )
    f(*state0)  # trace + emit + Metal compile outside the clock
    state, t0 = state0, time.perf_counter()
    for chunk in range(1, CHUNKS + 1):
        state = f(*state)
        on_mark(chunk, state)
    return time.perf_counter() - t0


def df32_run(state0, on_mark):
    kernel = mrdf32.kernel(DF32_KERNEL, "kepler_verlet_df32")
    # Two limbs per component. The low limbs start at zero: the initial
    # conditions are exactly the float32 ones the other runs use, so the
    # three variants start from bit-identical states.
    zero = np.zeros(N, dtype=np.float32)
    limbs = [limb for v in state0 for limb in (v, zero)]
    bufs = [mr.Buffer(np.ascontiguousarray(v)) for v in limbs]
    outs = [mr.Buffer.empty([N]) for _ in range(8)]
    scalars = [np.float32(H), np.uint32(CHUNK)]

    def launch():
        mr.run(kernel, grid=N, buffers=[*bufs, *outs], scalars=scalars)

    launch()  # compile outside the clock
    t0 = time.perf_counter()
    for chunk in range(1, CHUNKS + 1):
        launch()
        limbs = [b.to_numpy() for b in outs]
        for dst, limb in zip(bufs, limbs):  # feed both limbs back in
            dst.copy_from(limb)
        on_mark(chunk, tuple(limbs[0::2]))
    return time.perf_counter() - t0


def part_b():
    state0 = ensemble(N)
    e0 = energy(*state0)  # the float32 start state's true energy, per member
    print(f"Part B -- velocity-Verlet on the GPU, {N} trajectories, h={H}")
    print(
        f"          {CHUNKS} x {CHUNK:,} steps = {CHUNKS * CHUNK:,} steps, "
        f"~{int(CHUNKS * CHUNK * H / (2 * np.pi))} orbits"
    )
    print("          RMS |E - E_start| over the ensemble, E in float64")
    print(
        f"          (the float32 start states sit {np.abs(e0 + 0.5).max():.1e} "
        "off the exact E = -1/2)"
    )
    print()

    rows = {}

    def collect(name):
        rows[name] = []
        return lambda chunk, state: rows[name].append(
            float(np.sqrt(np.mean((energy(*state) - e0) ** 2)))
        )

    t_fast = palladium_run(mr.MathMode.FAST, state0, collect("f32 FAST"))
    t_safe = palladium_run(mr.MathMode.SAFE, state0, collect("f32 SAFE"))
    t_df32 = df32_run(state0, collect("df32 SAFE"))

    print(f"  {'steps':>12}{'f32 FAST':>13}{'f32 SAFE':>13}{'df32 SAFE':>13}")
    for chunk in MARKS:
        cells = "".join(f"{rows[k][chunk - 1]:>13.2e}" for k in rows)
        print(f"  {chunk * CHUNK:>12,}{cells}")
    print()
    print(f"  wall clock: FAST {t_fast:.1f} s, SAFE {t_safe:.1f} s, df32 {t_df32:.1f} s")
    print()

    # Fit over the last decade only: below it the float32 lines are still
    # sitting on the truncation floor, where round-off is not what is
    # being measured.
    lo = CHUNKS // 10
    n = np.log(np.arange(lo, CHUNKS) + 1.0)
    slopes = {k: float(np.polyfit(n, np.log(v[lo:]), 1)[0]) for k, v in rows.items()}
    print("  slope of log(error) vs log(steps), fitted over the last decade:")
    print("   " + "".join(f"   {k} {s:+.2f}" for k, s in slopes.items()))
    print()
    print("  ~1/2 is the random walk of round-off. ~0 is an error bounded by")
    print("  the integrator, with the hardware no longer in the way -- and the")
    print("  df32 floor is Part A's Verlet bound at this h, reached and held.")
    return rows, slopes


def main():
    print("Kepler two-body, long-horizon energy conservation")
    print("=" * 70)
    print()
    part_a()
    print()
    print("=" * 70)
    print()
    rows, slopes = part_b()

    assert slopes["df32 SAFE"] < slopes["f32 SAFE"], (
        "compensated arithmetic should flatten the round-off walk"
    )
    assert rows["df32 SAFE"][-1] < rows["f32 SAFE"][-1], (
        "df32 should hold a lower error than float32 at the longest horizon"
    )


if __name__ == "__main__":
    main()
