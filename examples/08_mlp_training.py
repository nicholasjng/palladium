"""Example 8: an MLP training step with the matmuls on the Apple GPU.

Docs: builds on examples/07_custom_vjp.py; performance notes in
docs/performance.md.

The division of labor follows the escape-hatch thesis: only the GEMMs
run as Metal kernels (they are what the GPU wins decisively); loss,
activations, bias adds, and the Adam update stay as ordinary jnp inside
the same jax.jit (elementwise work is bandwidth-bound and a wash on
unified memory). One `custom_vjp` on the matmul alone is enough: JAX
differentiates everything around it natively and chains into the two
backward calls.

One kernel shape covers all of training, dictated by the emitter's
contiguous-block discipline (only the leading block dim may be
partial, for inputs and outputs alike). That means:

- the output is blocked over rows only, each instance owning its full
  row width, so a layer's width is capped by the cooperative
  accumulator budget (256 columns at 8 rows per instance);
- the rhs cannot be column-blocked at all, so every product is
  expressed as `a @ b.T` with `b` stored row-major in its output dim.
  The transpose fuses lazily into the dot (an indexing convention, on
  the MMA path a hardware transposed load, never a copy), and weights
  live in `(out, in)` layout, same as PyTorch's `Linear`, which makes
  the forward pass transpose-free:

    forward:  z  = a @ p^T     mmt(x, p)
    backward: da = g @ p       mmt(g, p.T)
              dp = g^T @ a     mmt(g.T, a.T)

The backward's `.T`s are materialized by XLA on the CPU for now
(bandwidth-cheap next to the GEMMs). Both restrictions, plus the
unfused bias, are the motivating measurements for the emitter work
sketched in docs/notes/coop-colvec-and-transposed-lhs.md: 2-D output
tiling is what would lift the width cap and re-enable the (8, 64) MMA
tile at any width; transposed-lhs dots would delete the host
transposes.

The script trains a student MLP against a random teacher, gates the
result against a pure-jnp reference step (same math, `a @ b.T` in
jnp), and reports paired-interleaved step timings.

Honest outcome (M1 Pro, heat-soaked medians; the ratios are paired
within-run and stable, the absolutes are lower bounds): the
correctness gates pass, and the GEMM kernels themselves now beat
XLA-CPU (the staged multi-SIMD-group lowering measures 550-790 GFLOP/s
vs the CPU's 430-480, a 1.3-1.6x win per matmul). The *step* still
trails at ~0.37x (batch 512), ~0.54x (batch 4096), ~0.93x (batch 8192,
hidden 512), because at these shapes the step is overhead-bound, not
GEMM-bound: 9 dispatches of fixed floor+FFI, the backward's host-side
transposes (~0.5 ms each at batch 4096, three per layer), and the CPU
elementwise passes together outweigh the now-fast GEMMs. The next
levers, in measured order: transposed-lhs dots (deletes the host
transposes) and epilogue fusion, both gated in
docs/notes/coop-colvec-and-transposed-lhs.md.
"""

import functools
import itertools
import os
import time

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np

import palladium

BM = 8  # rows per program instance (one SIMD-group)

BATCH = int(os.environ.get("PALLADIUM_MLP_BATCH", "512"))
# in, hidden, hidden, out. The MMA column-tile loop caps a layer's width
# at 512 (the 16KB threadgroup result tile at 8 rows); every matmul here
# rides the MMA path.
_HIDDEN = int(os.environ.get("PALLADIUM_MLP_HIDDEN", "256"))
DIMS = (_HIDDEN, _HIDDEN, _HIDDEN, 64)
STEPS = 60
LR = 1e-3
SEED = 3


@functools.cache
def _mmt(m: int, k: int, n: int):
    """out = a @ b.T with a (m, k) and b (n, k); one (BM, n) full-width
    row stripe of the output per program instance."""

    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...].T)

    return palladium.metal_call_jit(
        kernel,
        grid=(m // BM,),
        in_specs=[
            pl.BlockSpec((BM, k), lambda i: (i, 0)),
            pl.BlockSpec((n, k), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BM, n), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
    )


@jax.custom_vjp
def matmul_t(a, b):
    """a @ b.T on the Apple GPU, differentiable."""
    (m, k), n = a.shape, b.shape[0]
    return _mmt(m, k, n)(a, b)


def _matmul_t_fwd(a, b):
    return matmul_t(a, b), (a, b)


def _matmul_t_bwd(residuals, g):
    a, b = residuals
    (m, k), n = a.shape, b.shape[0]
    da = _mmt(m, n, k)(g, b.T)  # g @ b; the transposes here run on the CPU
    if n >= k:
        db = _mmt(n, m, k)(g.T, a.T)  # g.T @ a
    else:
        # Small output dim: a GEMM's grid extent is its m, so compute
        # db.T with the larger m for occupancy and transpose the small
        # result on the host.
        db = _mmt(k, m, n)(a.T, g.T).T
    return da, db


matmul_t.defvjp(_matmul_t_fwd, _matmul_t_bwd)


def forward(params, x, mmt):
    *hidden, (p_out, b_out) = params
    h = x
    for p, b in hidden:
        h = jnp.tanh(mmt(h, p) + b)  # bias and tanh: cheap jnp, on the CPU
    return mmt(h, p_out) + b_out


def loss_fn(params, x, y, mmt):
    return jnp.mean((forward(params, x, mmt) - y) ** 2)


def _mmt_ref(a, b):
    return a @ b.T


def init_params(key, dims):
    """Weights stored (out, in), PyTorch Linear layout."""
    params = []
    for d_in, d_out in itertools.pairwise(dims):
        key, sub = jax.random.split(key)
        p = jax.random.normal(sub, (d_out, d_in), jnp.float32) / np.sqrt(d_in)
        params.append((p, jnp.zeros((d_out,), jnp.float32)))
    return params


def adam_step(params, grads, state, step, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
    m, v = state
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
    t = step + 1
    scale = jnp.sqrt(1 - b2**t) / (1 - b1**t)
    params = jax.tree.map(
        lambda p, m_, v_: p - lr * scale * m_ / (jnp.sqrt(v_) + eps), params, m, v
    )
    return params, (m, v)


def make_train_step(mmt):
    @jax.jit
    def train_step(params, opt_state, step, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y, mmt=mmt)
        params, opt_state = adam_step(params, grads, opt_state, step)
        return params, opt_state, loss

    return train_step


def main():
    key = jax.random.PRNGKey(SEED)
    key, k_teacher, k_student, k_data = jax.random.split(key, 4)
    teacher = init_params(k_teacher, DIMS)
    x = jax.random.normal(k_data, (BATCH, DIMS[0]), jnp.float32)
    y = forward(teacher, x, mmt=_mmt_ref)

    def run(mmt, label):
        params = init_params(k_student, DIMS)
        opt_state = (
            jax.tree.map(jnp.zeros_like, params),
            jax.tree.map(jnp.zeros_like, params),
        )
        step_fn = make_train_step(mmt)
        losses = []
        for step in range(STEPS):
            params, opt_state, loss = step_fn(params, opt_state, step, x, y)
            losses.append(float(loss))
        print(f"{label:22s} loss {losses[0]:.5f} -> {losses[-1]:.5f}")
        return params, losses, step_fn, opt_state

    params_m, losses_m, step_m, opt_m = run(matmul_t, "metal matmuls")
    params_r, losses_r, step_r, opt_r = run(_mmt_ref, "jnp reference")

    # Gate 1: loss trajectories agree within f32/FAST-math tolerance.
    np.testing.assert_allclose(losses_m, losses_r, rtol=5e-3, atol=5e-5)
    # Gate 2: so do the trained parameters.
    for (pm, bm), (pr, br) in zip(params_m, params_r, strict=True):
        np.testing.assert_allclose(pm, pr, rtol=5e-3, atol=5e-3)
        np.testing.assert_allclose(bm, br, rtol=5e-3, atol=5e-3)
    print("trajectory and trained parameters match the jnp reference")

    # Paired-interleaved step timing (see docs/performance.md for why).
    def time_one(step_fn, params, opt_state):
        t0 = time.perf_counter()
        _p, _o, loss = step_fn(params, opt_state, STEPS, x, y)
        jax.block_until_ready(loss)
        return time.perf_counter() - t0

    for fn, p, o in ((step_m, params_m, opt_m), (step_r, params_r, opt_r)):
        time_one(fn, p, o)  # warm
    tm, tr = [], []
    for _ in range(30):
        tm.append(time_one(step_m, params_m, opt_m))
        tr.append(time_one(step_r, params_r, opt_r))
    tm_med, tr_med = np.median(tm) * 1e3, np.median(tr) * 1e3
    n_layers = len(DIMS) - 1
    print(
        f"step time: metal {tm_med:.2f} ms | jnp-CPU {tr_med:.2f} ms "
        f"| ratio {tr_med / tm_med:.2f}x"
    )
    print(
        f"({3 * n_layers} GPU dispatches/step; the GEMMs now beat the CPU "
        "1.3-1.6x in isolation, and the remaining gap is fixed overhead: "
        "dispatch floor, host-side transposes, CPU elementwise. See the "
        "module docstring and docs/notes/coop-colvec-and-transposed-lhs.md)"
    )


if __name__ == "__main__":
    main()
