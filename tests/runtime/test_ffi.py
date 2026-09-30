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
from palladium.ffi import _library_path

try:
    _library_path()
    _ffi_available = True
except FileNotFoundError:
    _ffi_available = False

pytestmark = pytest.mark.skipif(
    not _ffi_available, reason="palladium's native jax.ffi handler isn't built"
)


def _add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def test_eager_call_matches_interpret(rng):
    x = rng.standard_normal((8, 8), dtype=np.float32)
    y = rng.standard_normal((8, 8), dtype=np.float32)
    call = palladium.metal_call(_add_kernel, out_shape=jax.ShapeDtypeStruct((8, 8), jnp.float32))
    got = np.asarray(call(x, y))
    want = np.asarray(call.interpret(x, y))
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_composes_inside_jax_jit_with_ordinary_jnp_ops(rng):
    """A kernel result feeds jnp.sum under jax.jit with no tracer conflict."""
    x = rng.standard_normal((8, 8), dtype=np.float32)
    y = rng.standard_normal((8, 8), dtype=np.float32)
    call = palladium.metal_call(_add_kernel, out_shape=jax.ShapeDtypeStruct((8, 8), jnp.float32))

    @jax.jit
    def composed(a, b):
        return jnp.sum(call(a, b) ** 2)

    got = float(composed(x, y))
    want = float(np.sum((x + y) ** 2))
    assert got == pytest.approx(want, rel=1e-4)


def test_repeated_calls_reuse_the_cached_kernel(rng):
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
    """Multiple outputs take a different path from the single-output case on both the Python and C++ sides."""

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

    s, d = call(x, y)
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
    """`x + nan` only reliably produces NaN under SAFE, so a passing run shows math_mode reaches mr_compile_library."""

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
    """jax.vmap handles the whole batch in one FFI call and matches per-element calls bit for bit."""

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
    """An unbatched operand rides along at stride 0 rather than being materialized per element."""

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
    """An enclosing vmap batches the ffi_call itself, one dispatch per outer element."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 3.0

    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))
    xs = rng.standard_normal((3, 5, 8)).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(jax.vmap(jax.vmap(f))(xs)), xs * 3.0)


def test_custom_vjp_pairs_forward_and_backward_kernels(rng):
    """Forward and fused backward kernels paired with jax.custom_vjp match jax.grad of the plain jnp expression."""
    m, k, n = 8, 16, 8

    def fwd_kernel(x_ref, w_ref, y_ref):
        y_ref[...] = jnp.tanh(jnp.dot(x_ref[...], w_ref[...]))

    def bwd_kernel(x_ref, w_ref, y_ref, g_ref, dx_ref, dw_ref):
        t = g_ref[...] * (1.0 - y_ref[...] * y_ref[...])
        dx_ref[...] = jnp.dot(t, w_ref[...].T)
        dw_ref[...] = jnp.dot(x_ref[...].T, t)

    fwd = palladium.metal_call(fwd_kernel, out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32))
    bwd = palladium.metal_call(
        bwd_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((m, k), jnp.float32),
            jax.ShapeDtypeStruct((k, n), jnp.float32),
        ),
    )

    @jax.custom_vjp
    def dense(x, w):
        return fwd(x, w)

    def dense_fwd(x, w):
        y = fwd(x, w)
        return y, (x, w, y)

    def dense_bwd(res, g):
        x, w, y = res
        return bwd(x, w, y, g)

    dense.defvjp(dense_fwd, dense_bwd)

    x = jnp.asarray(rng.standard_normal((m, k)), dtype=jnp.float32)
    w = jnp.asarray(rng.standard_normal((k, n)) / np.sqrt(k), dtype=jnp.float32)
    grads = jax.jit(jax.grad(lambda x, w: jnp.sum(dense(x, w) ** 2), argnums=(0, 1)))
    ref = jax.jit(jax.grad(lambda x, w: jnp.sum(jnp.tanh(x @ w) ** 2), argnums=(0, 1)))
    dx, dw = grads(x, w)
    dx_ref, dw_ref = ref(x, w)
    np.testing.assert_allclose(np.asarray(dx), np.asarray(dx_ref), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(dw), np.asarray(dw_ref), rtol=1e-4, atol=1e-4)


def test_multiple_outputs_come_back_as_a_tuple_like_pallas_call():
    def two(x_ref, a_ref, b_ref):
        a_ref[...] = x_ref[...] + 1.0
        b_ref[...] = x_ref[...] * 2.0

    shape = jax.ShapeDtypeStruct((8,), jnp.float32)
    call = palladium.metal_call(two, out_shape=(shape, shape))
    x = jnp.ones(8, jnp.float32)
    outs = jax.jit(call)(x)
    assert isinstance(outs, tuple)
    np.testing.assert_array_equal(np.asarray(outs[0]), 2.0 * np.ones(8))
    np.testing.assert_array_equal(np.asarray(outs[1]), 2.0 * np.ones(8))
