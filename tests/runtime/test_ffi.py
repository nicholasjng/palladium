"""jax.ffi integration (`native/ffi/palladium_ffi.cpp`).

`metal_call_jit` emits MSL the same way `metal_call` does, but dispatches
through a registered jax.ffi target, so the result composes inside
`jax.jit` next to ordinary `jnp` ops. The native handler ships inside the
`palladium` package (root `CMakeLists.txt`, scikit-build-core) and is
built automatically by `uv sync`/`pip install`; this skips cleanly only
if that didn't happen (e.g. an unbuilt from-source checkout).
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
    call = palladium.metal_call_jit(
        _add_kernel, out_shape=jax.ShapeDtypeStruct((8, 8), jnp.float32)
    )
    got = np.asarray(call(x, y))
    want = np.asarray(call.interpret(x, y))
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_composes_inside_jax_jit_with_ordinary_jnp_ops(rng):
    """The actual point: a bare palladium kernel result feeding into
    jnp.sum under jax.jit, no np.asarray/tracer conflict."""
    x = rng.standard_normal((8, 8), dtype=np.float32)
    y = rng.standard_normal((8, 8), dtype=np.float32)
    call = palladium.metal_call_jit(
        _add_kernel, out_shape=jax.ShapeDtypeStruct((8, 8), jnp.float32)
    )

    @jax.jit
    def composed(a, b):
        return jnp.sum(call(a, b) ** 2)

    got = float(composed(x, y))
    want = float(np.sum((x + y) ** 2))
    assert got == pytest.approx(want, rel=1e-4)


def test_repeated_calls_reuse_the_cached_kernel(rng):
    """Second call with the same shape/dtype must not retrace or
    recompile; correctness across two distinct inputs is what proves the
    cached MSL/spec weren't stale for the second one."""
    call = palladium.metal_call_jit(
        _add_kernel, out_shape=jax.ShapeDtypeStruct((4, 4), jnp.float32)
    )

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
    """RemainingRets with more than one buffer: a genuinely different code
    path from the single-output case (list vs bare ShapeDtypeStruct on
    the Python side, n_inputs offset into `wrapped` on the C++ side)."""

    def kernel(x_ref, y_ref, sum_ref, diff_ref):
        sum_ref[...] = x_ref[...] + y_ref[...]
        diff_ref[...] = x_ref[...] - y_ref[...]

    x = rng.standard_normal((6, 6), dtype=np.float32)
    y = rng.standard_normal((6, 6), dtype=np.float32)
    call = palladium.metal_call_jit(
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
    """`x + nan` only reliably produces NaN under SAFE: FAST permits
    Metal to assume NaN never occurs (the preflight check's
    `test_nan_literal_propagates` establishes this for the base
    metal_call path). Confirmed load-bearing by sabotage: swapping the
    SAFE/FAST ordinals in ffi.py's _MATH_MODE_ORDINALS makes this fail,
    so a passing run means math_mode is actually reaching
    mr_compile_library, not silently ignored."""

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.where(x > 0.0, x + jnp.nan, x)

    f = palladium.metal_call_jit(
        kernel,
        math_mode=mr.MathMode.SAFE,
        out_shape=jax.ShapeDtypeStruct((64,), jnp.float32),
    )
    x = rng.standard_normal(64, dtype=np.float32)
    got = f(x)
    want = np.asarray(f.interpret(x))
    assert np.array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)])


def test_vmap_sequential_matches_per_element(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.tanh(x_ref[...]) * 2.0

    f = palladium.metal_call_jit(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16,), jnp.float32),
        vmap_method="sequential",
    )
    xs = rng.standard_normal((4, 16)).astype(np.float32)
    got = np.asarray(jax.vmap(f)(xs))
    want = np.stack([np.asarray(f(x)) for x in xs])
    np.testing.assert_allclose(got, want, rtol=1e-6)
    # and under jit
    got_jit = np.asarray(jax.jit(jax.vmap(f))(xs))
    np.testing.assert_allclose(got_jit, want, rtol=1e-6)


def test_vmap_pipelined_matches_interpret_and_sequential(rng):
    """'pipelined' handles the whole batch in one FFI call (the native
    handler loops with several dispatches in flight); results must be
    element-for-element identical to the sequential path and the oracle."""

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.tanh(x_ref[...]) * 2.0

    kwargs = {"out_shape": jax.ShapeDtypeStruct((16,), jnp.float32)}
    f = palladium.metal_call_jit(kernel, **kwargs, vmap_method="pipelined")
    f_seq = palladium.metal_call_jit(kernel, **kwargs, vmap_method="sequential")
    # A batch deeper than the handler's in-flight ring (8), so slot reuse
    # and backpressure inside the native loop are exercised.
    xs = rng.standard_normal((37, 16)).astype(np.float32)

    got = np.asarray(jax.vmap(f)(xs))
    # GPU vs GPU: the pipelined loop must be bit-identical to per-element
    # sequential dispatch of the same compiled kernel.
    np.testing.assert_array_equal(got, np.asarray(jax.vmap(f_seq)(xs)))
    want = np.stack([np.asarray(f(x)) for x in xs])
    np.testing.assert_array_equal(got, want)
    # vs the interpret oracle, at FAST-math tolerance (tanh approximation).
    oracle = np.stack([np.asarray(f.interpret(x)) for x in xs])
    np.testing.assert_allclose(got, oracle, rtol=1e-4, atol=1e-5)
    # under jit, and unvmapped calls still take the plain path
    got_jit = np.asarray(jax.jit(jax.vmap(f))(xs))
    np.testing.assert_array_equal(got_jit, want)
    np.testing.assert_array_equal(np.asarray(f(xs[0])), want[0])


def test_vmap_pipelined_broadcasts_unbatched_operands(rng):
    """in_axes=(0, None): the shared operand rides along at stride 0, read
    by every batch element instead of being materialized per element."""

    def kernel(x_ref, w_ref, o_ref):
        o_ref[...] = x_ref[...] * w_ref[...] + 1.0

    f = palladium.metal_call_jit(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16,), jnp.float32),
        vmap_method="pipelined",
    )
    xs = rng.standard_normal((5, 16)).astype(np.float32)
    w = rng.standard_normal(16).astype(np.float32)

    got = np.asarray(jax.vmap(f, in_axes=(0, None))(xs, w))
    want = np.stack([np.asarray(f.interpret(x, w)) for x in xs])
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_vmap_pipelined_multi_output(rng):
    def kernel(x_ref, s_ref, d_ref):
        s_ref[...] = x_ref[...] + x_ref[...]
        d_ref[...] = x_ref[...] * x_ref[...]

    f = palladium.metal_call_jit(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((8,), jnp.float32),
            jax.ShapeDtypeStruct((8,), jnp.float32),
        ),
        vmap_method="pipelined",
    )
    xs = rng.standard_normal((6, 8)).astype(np.float32)
    got_s, got_d = jax.vmap(f)(xs)
    np.testing.assert_allclose(np.asarray(got_s), xs + xs, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(got_d), xs * xs, rtol=1e-6)


def test_vmap_works_by_default(rng):
    """The default is 'pipelined': jax.vmap composes with no opt-in. Was
    None, which sent users to jax.ffi's own NotImplementedError — and that
    message recommends expand_dims/broadcast_all, the two methods
    metal_call_jit rejects as silently wrong here."""

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.tanh(x_ref[...]) * 2.0

    kwargs = {"out_shape": jax.ShapeDtypeStruct((16,), jnp.float32)}
    f = palladium.metal_call_jit(kernel, **kwargs)
    xs = rng.standard_normal((12, 16)).astype(np.float32)

    got = np.asarray(jax.vmap(f)(xs))
    explicit = palladium.metal_call_jit(kernel, **kwargs, vmap_method="pipelined")
    np.testing.assert_array_equal(got, np.asarray(jax.vmap(explicit)(xs)))
    np.testing.assert_array_equal(got, np.stack([np.asarray(f(x)) for x in xs]))


def test_nested_vmap_batches_outer_levels_sequentially(rng):
    """One pipelined FFI call is one vmap level; an enclosing vmap batches
    the ffi_call itself, one dispatch per outer element."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 3.0

    f = palladium.metal_call_jit(kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))
    xs = rng.standard_normal((3, 5, 8)).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(jax.vmap(jax.vmap(f))(xs)), xs * 3.0)


def test_vmap_method_none_refuses_batching_in_palladium_terms():
    """Opting out still reports through palladium: the custom_vmap rule
    raises before jax.ffi can suggest the unsafe whole-batch methods."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    f = palladium.metal_call_jit(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        vmap_method=None,
    )
    x = np.ones((8,), dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(f(x)), x)  # unvmapped still works
    with pytest.raises(ValueError, match="vmap_method='pipelined'"):
        jax.vmap(f)(np.ones((4, 8), dtype=np.float32))


def test_whole_batch_vmap_methods_rejected():
    # expand_dims/broadcast_all re-invoke the target once with batched
    # buffers, but the launch grid is baked per unbatched shape; accepting
    # them would return wrong results silently.
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    with pytest.raises(ValueError, match="pipelined"):
        palladium.metal_call_jit(
            kernel,
            out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
            vmap_method="expand_dims",
        )


def test_custom_vjp_pairs_forward_and_backward_kernels(rng):
    """The custom-VJP recipe: y = tanh(x @ W) forward and
    a fused backward kernel, paired with jax.custom_vjp; gradients must
    match jax.grad of the plain jnp expression."""
    m, k, n = 8, 16, 8

    def fwd_kernel(x_ref, w_ref, y_ref):
        y_ref[...] = jnp.tanh(jnp.dot(x_ref[...], w_ref[...]))

    def bwd_kernel(x_ref, w_ref, y_ref, g_ref, dx_ref, dw_ref):
        t = g_ref[...] * (1.0 - y_ref[...] * y_ref[...])
        dx_ref[...] = jnp.dot(t, w_ref[...].T)
        dw_ref[...] = jnp.dot(x_ref[...].T, t)

    fwd = palladium.metal_call_jit(fwd_kernel, out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32))
    bwd = palladium.metal_call_jit(
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
