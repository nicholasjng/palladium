"""A batched RK4 Lotka-Volterra integrator.

One Metal thread integrates one (prey, predator) system over `STEPS` RK4
steps with its own parameters; the grid is the ensemble.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from kernels import lv_kernel

import palladium


def make_solver(n: int):
    spec_1 = pl.BlockSpec((1,), lambda i: (i,))
    return palladium.metal_call(
        lv_kernel,
        grid=(n,),
        in_specs=[spec_1] * 6,
        out_specs=(spec_1, spec_1),
        out_shape=(
            jax.ShapeDtypeStruct((n,), jnp.float32),
            jax.ShapeDtypeStruct((n,), jnp.float32),
        ),
    )


def _ensemble(rng, n):
    x0 = rng.uniform(0.8, 1.2, n).astype(np.float32)
    y0 = rng.uniform(0.8, 1.2, n).astype(np.float32)
    a = rng.uniform(0.9, 1.3, n).astype(np.float32)
    b = rng.uniform(0.3, 0.5, n).astype(np.float32)
    c = rng.uniform(0.08, 0.12, n).astype(np.float32)
    d = rng.uniform(0.3, 0.5, n).astype(np.float32)
    return x0, y0, a, b, c, d


def test_matches_interpret_oracle(rng):
    """Compare a blocked Lotka-Volterra RK4 ensemble's final states with the Pallas interpreter."""
    n = 512
    f = make_solver(n)
    args = _ensemble(rng, n)
    got_x, got_y = f(*args)
    want_x, want_y = f.interpret(*args)
    # 500 RK4 steps of f32 accumulate real drift between the FAST-math GPU
    # and the CPU, hence the 1e-3 relative bar.
    np.testing.assert_allclose(got_x, np.asarray(want_x), rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(got_y, np.asarray(want_y), rtol=1e-3, atol=1e-5)
