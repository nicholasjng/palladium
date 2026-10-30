"""jax.ffi integration.

`metal_call` dispatches through a registered jax.ffi target, so the
result composes inside `jax.jit` next to ordinary `jnp` ops. The module
skips when the native handler is not built.
"""

import jax
import jax.numpy as jnp
import metal_runtime as mr
import numpy as np
import pytest

import palladium
from palladium._runtime import library_path

try:
    library_path()
    _ffi_available = True
except FileNotFoundError:
    _ffi_available = False

pytestmark = pytest.mark.skipif(
    not _ffi_available, reason="palladium's native jax.ffi handler isn't built"
)


def _add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def test_eager_call_matches_interpret(rng):
    """Compare an eager two-input FFI addition with the Pallas interpreter."""
    x = rng.standard_normal((8, 8), dtype=np.float32)
    y = rng.standard_normal((8, 8), dtype=np.float32)
    call = palladium.metal_call(_add_kernel, out_shape=jax.ShapeDtypeStruct((8, 8), jnp.float32))
    got = np.asarray(call(x, y))
    want = np.asarray(call.interpret(x, y))
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_repeated_calls_reuse_the_cached_kernel(rng):
    """Check repeated FFI calls for correct results and reuse of the shape-specialization cache
    entry."""
    call = palladium.metal_call(_add_kernel, out_shape=jax.ShapeDtypeStruct((4, 4), jnp.float32))

    @jax.jit
    def composed(a, b):
        return jnp.sum(call(a, b) ** 2)

    x1 = rng.standard_normal((4, 4), dtype=np.float32)
    x2 = rng.standard_normal((4, 4), dtype=np.float32)
    y = np.ones((4, 4), dtype=np.float32)

    assert len(call._cache) == 0
    r1 = float(composed(x1, y))
    assert len(call._cache) == 1
    r2 = float(composed(x2, y))
    assert len(call._cache) == 1  # same shape/dtype: cache hit, not a second entry

    assert r1 == pytest.approx(float(np.sum((x1 + y) ** 2)), rel=1e-4)
    assert r2 == pytest.approx(float(np.sum((x2 + y) ** 2)), rel=1e-4)


def test_multi_output_kernel(rng):
    """Check sum/difference FFI outputs and a jitted computation consuming both against NumPy."""

    def kernel(x_ref, y_ref, sum_ref, diff_ref):
        sum_ref[...] = x_ref[...] + y_ref[...]
        diff_ref[...] = x_ref[...] - y_ref[...]

    x = rng.standard_normal((6, 6), dtype=np.float32)
    y = rng.standard_normal((6, 6), dtype=np.float32)
    call = palladium.metal_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((6, 6), jnp.float32),
            jax.ShapeDtypeStruct((6, 6), jnp.float32),
        ),
    )

    outs = call(x, y)
    assert isinstance(outs, tuple)  # like pallas_call, not the FFI primitive's list
    s, d = outs
    np.testing.assert_allclose(s, x + y, atol=1e-6)
    np.testing.assert_allclose(d, x - y, atol=1e-6)

    @jax.jit
    def composed(a, b):
        s, d = call(a, b)
        return jnp.sum(s) + jnp.sum(d**2)

    got = float(composed(x, y))
    want = float(np.sum(x + y) + np.sum((x - y) ** 2))
    assert got == pytest.approx(want, rel=1e-4)


def test_math_mode_safe_is_actually_requested(rng):
    """Check NaN propagation to verify that SAFE math reaches native Metal compilation."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.where(x > 0.0, x + jnp.nan, x)

    f = palladium.metal_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((64,), jnp.float32),
        compiler_params=palladium.CompilerParams(math_mode=mr.MathMode.SAFE),
    )
    x = rng.standard_normal(64, dtype=np.float32)
    got = f(x)
    want = np.asarray(f.interpret(x))
    assert np.array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)])


def test_vmap_matches_per_element_calls_and_interpret(rng):
    """Compare vmapped FFI results with individual calls bit for bit and with the interpreter
    numerically."""

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.tanh(x_ref[...]) * 2.0

    kwargs = {"out_shape": jax.ShapeDtypeStruct((16,), jnp.float32)}
    f = palladium.metal_call(kernel, **kwargs)
    xs = rng.standard_normal((37, 16)).astype(np.float32)

    got = np.asarray(jax.vmap(f)(xs))
    # GPU vs GPU: the batched loop must be bit-identical to per-element
    # dispatch of the same compiled kernel.
    want = np.stack([np.asarray(f(x)) for x in xs])
    np.testing.assert_array_equal(got, want)
    # vs the interpret oracle, at FAST-math tolerance (tanh approximation).
    oracle = np.stack([np.asarray(f.interpret(x)) for x in xs])
    np.testing.assert_allclose(got, oracle, rtol=1e-4, atol=1e-5)
    # under jit, and unvmapped calls still take the plain path
    got_jit = np.asarray(jax.jit(jax.vmap(f))(xs))
    np.testing.assert_array_equal(got_jit, want)
    np.testing.assert_array_equal(np.asarray(f(xs[0])), want[0])


def test_vmap_broadcasts_unbatched_operands(rng):
    """Check that a shared unbatched operand broadcasts correctly across a vmapped FFI batch."""

    def kernel(x_ref, w_ref, o_ref):
        o_ref[...] = x_ref[...] * w_ref[...] + 1.0

    f = palladium.metal_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16,), jnp.float32),
    )
    xs = rng.standard_normal((5, 16)).astype(np.float32)
    w = rng.standard_normal(16).astype(np.float32)

    got = np.asarray(jax.vmap(f, in_axes=(0, None))(xs, w))
    want = np.stack([np.asarray(f.interpret(x, w)) for x in xs])
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_vmap_multi_output(rng):
    """Check that vmapped FFI dispatch preserves every output's batch shape and numerical result."""

    def kernel(x_ref, s_ref, d_ref):
        s_ref[...] = x_ref[...] + x_ref[...]
        d_ref[...] = x_ref[...] * x_ref[...]

    f = palladium.metal_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((8,), jnp.float32),
            jax.ShapeDtypeStruct((8,), jnp.float32),
        ),
    )
    xs = rng.standard_normal((6, 8)).astype(np.float32)
    got_s, got_d = jax.vmap(f)(xs)
    np.testing.assert_allclose(np.asarray(got_s), xs + xs, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(got_d), xs * xs, rtol=1e-6)


def test_nested_vmap_batches_outer_levels_sequentially(rng):
    """Check nested vmaps of an FFI scaling kernel against NumPy across both batch dimensions."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 3.0

    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))
    xs = rng.standard_normal((3, 5, 8)).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(jax.vmap(jax.vmap(f))(xs)), xs * 3.0)
