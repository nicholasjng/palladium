"""Exercise 4: `_rule_scan` in emit.py. fori_loop becomes a C for-loop.

This is the exercise that makes the whole project worthwhile: once the
time-stepping loop lives *inside* the kernel, one dispatch does the whole
solve, and the per-step overhead that dominates XLA-driven integrators is
gone. Everything else was setup.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import palladium

pytestmark = pytest.mark.exercise


def test_euler_logistic(rng):
    """dy/dt = r*y(1 - y), 100 explicit Euler steps, entirely in-kernel.

    The rate r is read from a Ref *before* the loop and used inside it;
    that is how scan consts arise (see the exercise-4 docstring), and every
    real ODE kernel has them (its parameters). Meet them here, not in the
    capstone."""
    dt, steps = 0.01, 100

    def kernel(y0_ref, r_ref, o_ref):
        r = r_ref[...]

        def step(_, y):
            return y + dt * r * y * (1.0 - y)

        o_ref[...] = jax.lax.fori_loop(0, steps, step, y0_ref[...])

    f = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((1024,), jnp.float32)
    )
    y0 = rng.uniform(0.1, 0.9, 1024).astype(np.float32)
    r = rng.uniform(0.5, 1.5, 1024).astype(np.float32)
    got = f(y0, r)
    want = np.asarray(f.interpret(y0, r))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)
    # And the physics: logistic flows toward the y=1 fixed point.
    assert np.all(np.abs(got - 1.0) < np.abs(y0 - 1.0))  # ty: ignore[unsupported-operator]


def test_tuple_carry(rng):
    """fori_loop with a tuple carry -> scan with two carried values."""

    def kernel(x_ref, o1_ref, o2_ref):
        def step(_, carry):
            a, b = carry
            return a + 0.5 * b, b - 0.5 * a

        a, b = jax.lax.fori_loop(0, 50, step, (x_ref[...], x_ref[...]))
        o1_ref[...] = a
        o2_ref[...] = b

    f = palladium.metal_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((256,), jnp.float32),
            jax.ShapeDtypeStruct((256,), jnp.float32),
        ),
    )
    x = rng.standard_normal(256, dtype=np.float32)
    got1, got2 = f(x)
    want1, want2 = f.interpret(x)
    np.testing.assert_allclose(got1, np.asarray(want1), rtol=1e-4)
    np.testing.assert_allclose(got2, np.asarray(want2), rtol=1e-4)


def test_nested_loops(rng):
    """A loop in a loop: substepping, the shape adaptive integrators take."""

    def kernel(y_ref, o_ref):
        def outer(_, y):
            def inner(_, z):
                return z + 0.001 * z

            return jax.lax.fori_loop(0, 10, inner, y)

        o_ref[...] = jax.lax.fori_loop(0, 20, outer, y_ref[...])

    f = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((128,), jnp.float32)
    )
    y = rng.uniform(0.5, 1.5, 128).astype(np.float32)
    np.testing.assert_allclose(f(y), np.asarray(f.interpret(y)), rtol=1e-4)


def test_carry_permutation(rng):
    """A body that returns its carries reordered: `return b, a` stages
    outvars that ARE the carry vars, so a sequential copy-back clobbers;
    scan semantics require all carries to update simultaneously."""

    def kernel(x_ref, y_ref, xo_ref, yo_ref):
        def step(_, c):
            a, b = c
            return b, a

        a, b = jax.lax.fori_loop(0, 3, step, (x_ref[...], y_ref[...]))
        xo_ref[...] = a
        yo_ref[...] = b

    f = palladium.metal_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((64,), jnp.float32),
            jax.ShapeDtypeStruct((64,), jnp.float32),
        ),
    )
    x = rng.standard_normal(64, dtype=np.float32)
    y = rng.standard_normal(64, dtype=np.float32)
    got_a, got_b = f(x, y)
    want_a, want_b = f.interpret(x, y)
    np.testing.assert_array_equal(got_a, np.asarray(want_a))
    np.testing.assert_array_equal(got_b, np.asarray(want_b))
