"""lax.scan with scanned xs and stacked ys: the dense-output stepper
idiom. Every lowering is diffed against the interpret oracle; the
streaming-store fusion is additionally pinned by MSL text assertions."""

import jax
import jax.numpy as jnp
import numpy as np

import palladium

F32 = jnp.float32


def _shaped(*shape):
    return jax.ShapeDtypeStruct(shape, F32)


def _run_and_compare(kernel, out_shape, *arrays, tol=1e-5):
    call = palladium.metal_call(kernel, out_shape=out_shape)
    got = call(*arrays)
    want = call.interpret(*arrays)
    if isinstance(got, tuple):
        for g, w in zip(got, want, strict=True):
            np.testing.assert_allclose(g, np.asarray(w), rtol=tol, atol=tol)
    else:
        np.testing.assert_allclose(got, np.asarray(want), rtol=tol, atol=tol)
    return call


def test_scanned_xs_reduction(rng):
    def kernel(ts_ref, o_ref):
        def step(acc, t):
            return acc + t * t, None

        total, _ = jax.lax.scan(step, 0.0, ts_ref[...])
        o_ref[0] = total

    ts = rng.standard_normal(32).astype(np.float32)
    _run_and_compare(kernel, _shaped(1), ts)


def test_stacked_ys_without_xs(rng):
    def kernel(y0_ref, o_ref):
        def step(y, _):
            y = y * 1.5 + 1.0
            return y, y

        _, ys = jax.lax.scan(step, y0_ref[0], None, length=8)
        o_ref[...] = ys

    y0 = rng.standard_normal(1).astype(np.float32)
    _run_and_compare(kernel, _shaped(8), y0)


def test_dense_output_scan_streams_to_the_ref(rng):
    """The headline idiom: scan over save times, stack the states, write
    the trajectory. The stacked ys must stream straight to the output
    ref: no thread-local trajectory array, no final copy loop."""

    def kernel(y0_ref, ts_ref, o_ref):
        def step(y, t):
            y_next = y + 0.1 * t * y
            return y_next, y_next

        _, ys = jax.lax.scan(step, y0_ref[...], ts_ref[...])
        o_ref[...] = ys

    y0 = rng.standard_normal(4).astype(np.float32)
    ts = rng.standard_normal(16).astype(np.float32)
    _run_and_compare(kernel, _shaped(16, 4), y0, ts)

    msl = palladium.debug_msl(kernel, _shaped(4), _shaped(16), out_shape=_shaped(16, 4))
    assert "[64]" not in msl  # no 16x4 thread-local trajectory


def test_dense_output_survives_the_stack_ceiling(rng):
    """A trajectory too large for the per-thread stack: only the
    streaming path can lower and run this."""
    n_save, dim = 4096, 8

    def kernel(y0_ref, o_ref):
        def step(y, _):
            y = y * 0.999 + 0.001
            return y, y

        _, ys = jax.lax.scan(step, y0_ref[...], None, length=n_save)
        o_ref[...] = ys

    y0 = rng.standard_normal(dim).astype(np.float32)
    call = palladium.metal_call(kernel, out_shape=_shaped(n_save, dim))
    got = call(y0)
    want = np.asarray(call.interpret(y0))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    msl = palladium.debug_msl(kernel, _shaped(dim), out_shape=_shaped(n_save, dim))
    assert f"[{n_save * dim}]" not in msl


def test_ys_with_other_consumers_falls_back_to_thread_local(rng):
    def kernel(y0_ref, o_ref):
        def step(y, _):
            y = y * 1.1
            return y, y

        _, ys = jax.lax.scan(step, y0_ref[0], None, length=8)
        o_ref[...] = ys * 2.0  # consumer is mul, not the swap itself

    y0 = rng.standard_normal(1).astype(np.float32)
    _run_and_compare(kernel, _shaped(8), y0)
    msl = palladium.debug_msl(kernel, _shaped(1), out_shape=_shaped(8))
    assert "[8]" in msl  # the stacked thread-local exists on this path


def test_multiple_stacked_ys(rng):
    def kernel(ts_ref, sums_ref, prods_ref):
        def step(c, t):
            s, p = c
            s, p = s + t, p * (1.0 + t * t)
            return (s, p), (s, p)

        _, (sums, prods) = jax.lax.scan(step, (0.0, 1.0), ts_ref[...])
        sums_ref[...] = sums
        prods_ref[...] = prods

    ts = (0.1 * rng.standard_normal(16)).astype(np.float32)
    _run_and_compare(kernel, (_shaped(16), _shaped(16)), ts)


def test_reverse_scan_with_xs_and_ys(rng):
    def kernel(ts_ref, o_ref):
        def step(acc, t):
            acc = acc * 0.5 + t
            return acc, acc

        _, ys = jax.lax.scan(step, 0.0, ts_ref[...], reverse=True)
        o_ref[...] = ys

    ts = rng.standard_normal(16).astype(np.float32)
    _run_and_compare(kernel, _shaped(16), ts)


def test_matrix_valued_xs_slices(rng):
    def kernel(ms_ref, v_ref, o_ref):
        def step(v, m):
            return jnp.dot(m, v), None

        v_final, _ = jax.lax.scan(step, v_ref[...], ms_ref[...])
        o_ref[...] = v_final

    ms = (0.1 * rng.standard_normal((6, 4, 4))).astype(np.float32)
    v = rng.standard_normal(4).astype(np.float32)
    _run_and_compare(kernel, _shaped(4), ms, v)


def test_aliased_output_is_not_a_streaming_target(rng):
    """input_output_aliases shares the buffer with the input ref, whose
    reads this level cannot see; the ys must take the thread-local path
    and stay correct."""

    def kernel(x_ref, o_ref):
        def step(acc, x):
            acc = acc + x
            return acc, acc

        _, ys = jax.lax.scan(step, 0.0, x_ref[...])
        o_ref[...] = ys

    f = palladium.metal_call(
        kernel,
        out_shape=_shaped(16),
        input_output_aliases={0: 0},
    )
    x = rng.standard_normal(16).astype(np.float32)
    np.testing.assert_allclose(f(x), np.asarray(f.interpret(x)), rtol=1e-5)


def test_grid_blocked_dense_output(rng):
    """One trajectory per program instance: the ensemble dense-output
    shape, with the ys streaming into each instance's output block."""
    import jax.experimental.pallas as pl

    n_traj, n_save = 8, 16

    def kernel(y0_ref, ts_ref, o_ref):
        def step(y, t):
            y_next = y * (1.0 + 0.1 * t)
            return y_next, y_next

        _, ys = jax.lax.scan(step, y0_ref[0], ts_ref[...])
        o_ref[...] = ys

    call = palladium.metal_call(
        kernel,
        grid=(n_traj,),
        in_specs=[
            pl.BlockSpec((1,), lambda i: i),
            pl.BlockSpec((n_save,), lambda i: 0),
        ],
        out_specs=pl.BlockSpec((None, n_save), lambda i: (i, 0)),
        out_shape=_shaped(n_traj, n_save),
    )
    y0 = rng.standard_normal(n_traj).astype(np.float32)
    ts = (0.1 * rng.standard_normal(n_save)).astype(np.float32)
    got = call(y0, ts)
    want = np.asarray(call.interpret(y0, ts))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_full_block_xs_reads_the_ref_without_a_stack_copy(rng):
    """A save-time grid consumed only as scan xs must not be copied to
    the per-thread stack first; at this size only the aliased read can
    lower and run."""
    n_ts = 16384  # 64KB as f32: alone past the per-thread stack

    def kernel(ts_ref, o_ref):
        def step(acc, t):
            return acc * 0.5 + t, None

        total, _ = jax.lax.scan(step, 0.0, ts_ref[...])
        o_ref[0] = total

    call = palladium.metal_call(kernel, out_shape=_shaped(1))
    ts = rng.standard_normal(n_ts).astype(np.float32)
    got = call(ts)
    want = np.asarray(call.interpret(ts))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    msl = palladium.debug_msl(kernel, _shaped(n_ts), out_shape=_shaped(1))
    assert f"[{n_ts}]" not in msl


def test_scan_const_blocks_keep_their_thread_local_copy(rng):
    """A block closed over by the scan body (a const) is re-read every
    iteration: the cache copy must stay."""

    def kernel(w_ref, ts_ref, o_ref):
        w = w_ref[...]  # const inside the body: copied

        def step(acc, t):
            return acc + t * jnp.sum(w), None

        total, _ = jax.lax.scan(step, 0.0, ts_ref[...])
        o_ref[0] = total

    ts = rng.standard_normal(8).astype(np.float32)
    w = rng.standard_normal(4).astype(np.float32)
    _run_and_compare(kernel, _shaped(1), w, ts)
    msl = palladium.debug_msl(kernel, _shaped(4), _shaped(8), out_shape=_shaped(1))
    assert "[4]" in msl  # w's thread-local cache copy
    assert "[8]" not in msl  # ts aliased: consumed only as xs


def test_block_used_as_both_xs_and_const_still_copies(rng):
    """One value feeding the same scan in two positions disqualifies the
    alias; correctness first."""

    def kernel(ts_ref, o_ref):
        ts = ts_ref[...]

        def step(acc, t):
            return acc * 0.5 + t + 0.01 * jnp.sum(ts), None  # ts: const AND xs

        total, _ = jax.lax.scan(step, 0.0, ts)
        o_ref[0] = total

    ts = rng.standard_normal(8).astype(np.float32)
    _run_and_compare(kernel, _shaped(1), ts)
    msl = palladium.debug_msl(kernel, _shaped(8), out_shape=_shaped(1))
    assert "[8]" in msl
