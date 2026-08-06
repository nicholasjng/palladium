"""`_rule_cond` and `_rule_while`: data-dependent branching and loops.

The gridded tests give every program instance its own branch or trip
count; divergence across threads is native to the one-thread-per-instance
model. The interpret oracle is the reference throughout.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium

pytestmark = pytest.mark.exercise


def _per_row(kernel, n_rows, *arrays, n_in):
    """One grid instance per row, (1, 1) blocks."""
    f = palladium.metal_call(
        kernel,
        grid=(n_rows,),
        in_specs=[pl.BlockSpec((1, 1), lambda i: (i, 0)) for _ in range(n_in)],
        out_specs=pl.BlockSpec((1, 1), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((n_rows, 1), jnp.float32),
    )
    got = np.asarray(f(*arrays))
    want = np.asarray(f.interpret(*arrays))
    return got, want


def test_cond_two_branches_diverge_per_thread(rng):
    def kernel(x_ref, o_ref):
        x = x_ref[0, 0]
        o_ref[0, 0] = jax.lax.cond(x > 0.0, lambda v: v * 2.0, lambda v: v - 1.0, x)

    x = rng.standard_normal((512, 1)).astype(np.float32)
    got, want = _per_row(kernel, 512, x, n_in=1)
    np.testing.assert_allclose(got, want, atol=1e-6)
    assert (x > 0).any() and (x <= 0).any(), "test data must exercise both branches"


def test_switch_three_branches_and_index_clamping(rng):
    def kernel(x_ref, i_ref, o_ref):
        o_ref[0, 0] = jax.lax.switch(
            i_ref[0, 0],
            [lambda v: v + 1.0, lambda v: v * 10.0, lambda v: -v],
            x_ref[0, 0],
        )

    x = rng.standard_normal((256, 1)).astype(np.float32)
    # -1 and 3 exercise lax.switch's clamp-to-range semantics.
    idx = rng.integers(-1, 4, size=(256, 1)).astype(np.int32)
    f = palladium.metal_call(
        kernel,
        grid=(256,),
        in_specs=[
            pl.BlockSpec((1, 1), lambda i: (i, 0)),
            pl.BlockSpec((1, 1), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((1, 1), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((256, 1), jnp.float32),
    )
    got = np.asarray(f(x, idx))
    want = np.asarray(f.interpret(x, idx))
    np.testing.assert_allclose(got, want, atol=1e-6)
    assert set(np.unique(idx)) >= {-1, 0, 1, 2, 3}


def test_while_loop_trip_count_diverges_per_thread(rng):
    def kernel(x_ref, o_ref):
        def cond(carry):
            y, _ = carry
            return y >= 1.0

        def body(carry):
            y, n = carry
            return y * 0.7, n + 1

        y, n = jax.lax.while_loop(cond, body, (x_ref[0, 0], jnp.int32(0)))
        o_ref[0, 0] = y + n.astype(jnp.float32)

    x = rng.uniform(1.0, 100.0, size=(512, 1)).astype(np.float32)
    got, want = _per_row(kernel, 512, x, n_in=1)
    np.testing.assert_allclose(got, want, atol=1e-5)
    # trip counts must actually differ across threads for this to test anything
    assert len(np.unique(np.floor(got))) > 3


def test_while_swapped_carries_need_the_snapshot(rng):
    """Body rotates its carries; without the two-phase copy-back one
    would clobber the other."""

    def kernel(a_ref, o_ref):
        def cond(carry):
            *_, n = carry
            return n < 5

        def body(carry):
            a, b, n = carry
            return b, a + b, n + 1

        a0 = a_ref[0, 0]
        a, b, _ = jax.lax.while_loop(cond, body, (a0, a0 + 1.0, jnp.int32(0)))
        o_ref[0, 0] = a + b

    a = rng.standard_normal((128, 1)).astype(np.float32)
    got, want = _per_row(kernel, 128, a, n_in=1)
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_logical_and_xor_not(rng):
    def kernel(x_ref, o_ref):
        x = x_ref[...]
        p = x > 0.0
        q = x > 1.0
        keep = jnp.logical_and(p, jnp.logical_not(q))  # 0 < x <= 1
        flip = jnp.logical_xor(p, q)  # differs from `keep` only at x > 1
        o_ref[...] = jnp.where(jnp.logical_and(keep, flip), x, 0.0)

    x = rng.uniform(-2.0, 2.0, size=1024).astype(np.float32)
    f = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((1024,), jnp.float32)
    )
    got = np.asarray(f(x))
    want = np.asarray(f.interpret(x))
    np.testing.assert_array_equal(got, want)
    band = (x > 0) & (x <= 1)
    np.testing.assert_array_equal(got, np.where(band, x, 0.0))
