"""`_rule_scan` (emit/rules.py): fori_loop becomes a C for-loop.

This is the rule that makes the whole project worthwhile: once the
time-stepping loop lives *inside* the kernel, one dispatch does the whole
solve, and the per-step overhead that dominates XLA-driven integrators is
gone. Everything else was setup.
"""

import jax
import jax.numpy as jnp
import numpy as np

import palladium


def test_euler_logistic(rng):
    """dy/dt = r*y(1 - y), 100 explicit Euler steps, entirely in-kernel.

    The rate r is read from a Ref *before* the loop and used inside it;
    that is how scan consts arise (see `_rule_scan`'s docstring), and
    every real ODE kernel has them (its parameters)."""
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


def test_carry_permutation_beside_computed_carry():
    """Shrunk from the emitter fuzzer (2026-08-18): a carry swap next to
    a computed third carry miscompiles on the Metal shader compiler when
    the loop body holds three or more thread-local array temporaries.
    The optimizer forwards a permuted carry's read across the write it
    must precede: with init (0, 1, 0) and body
    (c1, c0, tanh(c0 + 0*c0)), two iterations returned carry0 == 1
    instead of 0. Verified against hand-written MSL: snapshot arrays,
    fused copy-back loops, and hoisted declarations all still
    miscompile; volatile reads/writes at the hazard endpoints (what
    `_copy_back_carries` emits for permuted carries) do not. Fully
    scalarizing the chain also avoids it, which is expression-AST
    territory, not a copy-back fix.
    """

    def kernel(a_ref, b_ref, c_ref, ao_ref, bo_ref, co_ref):
        def step(_, c):
            return (c[1], c[0], jnp.tanh(c[0] + 0.0 * c[0]))

        init = (a_ref[...], b_ref[...], c_ref[...])
        fa, fb, fc = jax.lax.fori_loop(0, 2, step, init)
        ao_ref[...] = fa
        bo_ref[...] = fb
        co_ref[...] = fc

    n = 16
    f = palladium.metal_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct((n,), jnp.float32) for _ in range(3)),
    )
    a = np.zeros(n, np.float32)
    b = np.ones(n, np.float32)
    c = np.zeros(n, np.float32)
    got = f(a, b, c)
    want = f.interpret(a, b, c)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, np.asarray(w), rtol=1e-6, atol=1e-6)


def test_carry_read_and_alias_conflict(rng):
    """Found by test_fuzz_loops, shrunk by hand. 3 carries, length=2:
    c0 and c2 swap (c0's new value is old c2, c2's new value is old c0),
    c1 becomes tanh(c0). c0 is both the fresh computation's input and one
    half of the swap. Was a strict xfail (a Metal compiler bug, not an
    emitter logic error); fixed by the volatile hazard endpoints in
    `_copy_back_carries`, see test_carry_permutation_beside_computed_carry."""

    def kernel(x0_ref, x1_ref, x2_ref, o0_ref, o1_ref, o2_ref):
        def step(_, carry):
            c0, _c1, c2 = carry
            return c2, jnp.tanh(c0 + 0.0 * c0), c0

        f0, f1, f2 = jax.lax.fori_loop(
            0, 2, step, (x0_ref[...], x1_ref[...], x2_ref[...])
        )
        o0_ref[...] = f0
        o1_ref[...] = f1
        o2_ref[...] = f2

    f = palladium.metal_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct((16,), jnp.float32) for _ in range(3)),
    )
    x0 = np.zeros(16, dtype=np.float32)
    x1 = np.zeros(16, dtype=np.float32)
    x2 = np.ones(16, dtype=np.float32)
    got = f(x0, x1, x2)
    want = f.interpret(x0, x1, x2)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, np.asarray(w))
