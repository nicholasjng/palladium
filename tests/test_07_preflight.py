"""Stretch-7 pre-flight: comparisons, select_n, and inf/nan literals.

These are the emitter features an adaptive step controller stages; each
test isolates one, and the last composes them into a controller-shaped
loop. The inf/nan tests run under MathMode.SAFE on purpose: FAST permits
Metal to assume infinities and NaNs never occur, so sentinel semantics
are only well-defined under SAFE or RELAXED (see metal-runtime's
MathMode docs). The controller itself should prefer a finfo-max sentinel
and keep FAST; these tests pin the literal formatting regardless.
"""

import jax
import jax.numpy as jnp
import metal_runtime as mr
import numpy as np
import pytest

import palladium

F32 = jnp.float32


def _oracle_check(f, *args, rtol=1e-5, atol=1e-6):
    got = f(*args)
    want = np.asarray(f.interpret(*args))
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


def test_comparison_to_float(rng):
    """Stages gt plus a bool -> float convert_element_type."""

    def kernel(x_ref, o_ref):
        o_ref[...] = (x_ref[...] > 0.5).astype(F32)

    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((256,), F32))
    x = rng.standard_normal(256, dtype=np.float32)
    got = f(x)
    np.testing.assert_array_equal(got, (x > 0.5).astype(np.float32))


def test_where_abs_the_hard_way(rng):
    """Predicate-order canary: swapped select_n cases emit -|x|, not |x|."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.where(x > 0.0, x, -x)

    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((256,), F32))
    x = rng.standard_normal(256, dtype=np.float32)
    got = f(x)
    np.testing.assert_array_equal(got, np.abs(x))
    _oracle_check(f, x)


def test_where_asymmetric_branches(rng):
    """Branches with different math, so any case mixup changes values."""

    def kernel(x_ref, y_ref, o_ref):
        x, y = x_ref[...], y_ref[...]
        o_ref[...] = jnp.where(x <= y, x * 2.0, y - 1.0)

    f = palladium.metal_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128,), F32),
    )
    x = rng.standard_normal(128, dtype=np.float32)
    y = rng.standard_normal(128, dtype=np.float32)
    _oracle_check(f, x, y)


def test_select_n_rejects_integer_predicate(rng):
    """lax.select_n with an int index: the predicate-type guard fires
    first, naming the root cause rather than the operand-count symptom."""

    def kernel(i_ref, x_ref, o_ref):
        idx = i_ref[...]
        x = x_ref[...]
        o_ref[...] = jax.lax.select_n(idx, x, x + 1.0, x + 2.0)

    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((8,), F32))
    idx = np.zeros(8, dtype=np.int32)
    x = np.zeros(8, dtype=np.float32)
    with pytest.raises(palladium.emit.EmitError, match="predicate of type bool"):
        f(idx, x)


def test_inf_literal(rng):
    """jnp.inf closed over in the kernel formats as INFINITY."""

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.minimum(x_ref[...], jnp.inf)

    src = palladium.debug_msl(
        kernel,
        jax.ShapeDtypeStruct((64,), F32),
        out_shape=jax.ShapeDtypeStruct((64,), F32),
    )
    assert "INFINITY" in src and "inff" not in src
    f = palladium.metal_call(
        kernel,
        math_mode=mr.MathMode.SAFE,
        out_shape=jax.ShapeDtypeStruct((64,), F32),
    )
    x = rng.standard_normal(64, dtype=np.float32)
    np.testing.assert_array_equal(f(x), x)


def test_negative_inf_sentinel(rng):
    """A running-max seeded with -inf: the standard sentinel pattern."""

    def kernel(x_ref, o_ref):
        def step(i, best):
            return jnp.maximum(best, x_ref[...] - 0.1 * i)

        o_ref[...] = jax.lax.fori_loop(0, 4, step, jnp.full((64,), -jnp.inf, F32))

    f = palladium.metal_call(
        kernel,
        math_mode=mr.MathMode.SAFE,
        out_shape=jax.ShapeDtypeStruct((64,), F32),
    )
    x = rng.standard_normal(64, dtype=np.float32)
    _oracle_check(f, x)


def test_nan_literal_propagates(rng):
    """NAN formats, compiles, and both sides agree on where the NaNs are."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.where(x > 0.0, x + jnp.nan, x)

    f = palladium.metal_call(
        kernel,
        math_mode=mr.MathMode.SAFE,
        out_shape=jax.ShapeDtypeStruct((64,), F32),
    )
    x = rng.standard_normal(64, dtype=np.float32)
    got = f(x)
    want = np.asarray(f.interpret(x))
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)])


def test_conditional_accumulation_in_loop(rng):
    """The controller shape in miniature: per-element accept/reject inside
    a fori_loop, composing comparisons, select_n, consts, and carries."""

    def kernel(y0_ref, r_ref, o_ref):
        r = r_ref[...]

        def step(_, y):
            grown = y + r
            return jnp.where(grown <= 1.0, grown, y)

        o_ref[...] = jax.lax.fori_loop(0, 20, step, y0_ref[...])

    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((256,), F32))
    y0 = rng.uniform(0.0, 0.5, 256).astype(np.float32)
    r = rng.uniform(0.01, 0.2, 256).astype(np.float32)
    got = f(y0, r)
    _oracle_check(f, y0, r)
    assert np.all(got <= 1.0 + 1e-6), "accept/reject failed to cap growth"  # ty: ignore[unsupported-operator]
