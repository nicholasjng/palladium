"""Threadgroup-shared scratch, barriers, and cooperative reductions.

Part A's scratch is `thread`-space: private to one program instance.
This is the shared tier -- `palladium.threadgroup_memory` requests a
compile-time-sized array in Metal's `threadgroup` address space, visible
to every thread in the group, with `barrier()` ordering the accesses and
`thread_index()`/`threads_per_threadgroup()` addressing them.

Why the oracle here is NumPy and not `interpret`: Pallas's interpret
mode runs program instances sequentially with no notion of a
threadgroup, so it models every instance as a group of one
(`thread_index() == 0`, `threads_per_threadgroup() == 1`). A kernel that
genuinely reduces across threads therefore computes something different
under interpret, and diffing against it would be meaningless. The
cooperative tests below use an explicit NumPy reference;
`test_interpret_models_a_threadgroup_of_one` pins the interpret
semantics so the divergence stays deliberate rather than surprising.

`test_barrier_is_emitted_where_the_author_put_it` is not redundant with
the numerical tests, and must not be deleted as such. Removing the
barrier emission entirely still produces *correct* results here at every
threadgroup size tried, up to 1024 (measured 2026-08-25): the race is
latent, not observable, so no numerical assertion catches it. The text
assertion is the only guard that does.

The load-bearing case is `test_block_sum_partial_tail_group`. Metal
dispatches non-uniform threadgroups, so a grid that is not a multiple of
the threadgroup size ends in a smaller group;
`threads_per_threadgroup()` reports that group's true size where a
compile-time constant would fold in slots no thread ever wrote.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium
from palladium.errors import EmitError
from palladium.threadgroup import (
    barrier,
    thread_index,
    threadgroup_memory,
    threads_per_threadgroup,
)

TG = 32


def _block_sum_kernel(x_ref, o_ref, s_ref):
    t = thread_index()
    s_ref[t] = x_ref[0]
    barrier()
    n = threads_per_threadgroup()
    acc = jax.lax.fori_loop(0, n, lambda i, a: a + s_ref[i], jnp.float32(0.0))
    o_ref[0] = acc


def _block_sum_call(n, threadgroup=TG, extent=TG):
    return palladium.metal_call(
        _block_sum_kernel,
        grid=(n,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
        scratch_shapes=[threadgroup_memory((extent,), jnp.float32)],
        threadgroup=threadgroup,
    )


def _expected_block_sums(x, tg):
    """Per-thread broadcast of each group's sum, tail group included."""
    out = np.empty_like(x)
    for start in range(0, len(x), tg):
        out[start : start + tg] = x[start : start + tg].sum()
    return out


def _msl(**kwargs):
    return palladium.debug_msl(
        _block_sum_kernel,
        jax.ShapeDtypeStruct((TG,), jnp.float32),
        grid=(TG,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((TG,), jnp.float32),
        scratch_shapes=[threadgroup_memory((TG,), jnp.float32)],
        **kwargs,
    )


# --- emission ------------------------------------------------------------


def test_scratch_declared_in_threadgroup_space():
    """A threadgroup_memory request declares `threadgroup T name[N];`.

    Same compile-time-sized local array as Part A's thread scratch, only
    the address-space qualifier differs.
    """
    body = [ln.strip() for ln in _msl().splitlines()]
    assert f"threadgroup float scratch0[{TG}];" in body
    assert not any(ln.startswith("thread float scratch") for ln in body)


def test_barrier_is_emitted_where_the_author_put_it():
    """barrier() lowers verbatim; placement is never inferred."""
    body = [ln.strip() for ln in _msl().splitlines()]
    assert "threadgroup_barrier(mem_flags::mem_threadgroup);" in body


def test_cooperative_builtins_are_in_the_signature():
    head = _msl().split("{")[0]
    assert "[[thread_position_in_threadgroup]]" in head
    assert "[[threads_per_threadgroup]]" in head


def test_thread_position_builtins_are_all_vector_typed():
    """Metal rejects a signature mixing scalar and vector position builtins.

    `_pid` is uint3, so `_tid`/`_tpt` must be too, or the Metal compiler
    fails with "expecting input declarations with either all scalar types
    or all vector types".
    """
    head = _msl().split("{")[0]
    for attr in (
        "thread_position_in_grid",
        "thread_position_in_threadgroup",
        "threads_per_threadgroup",
    ):
        line = next(ln for ln in head.splitlines() if attr in ln)
        assert line.strip().startswith("uint3 "), line


def test_plain_kernels_keep_their_signature():
    """Kernels without threadgroup scratch gain no new parameters.

    The cooperative builtins are added conditionally precisely so that
    every existing kernel (and every MSL snapshot) stays byte-identical.
    """

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    msl = palladium.debug_msl(
        kernel,
        jax.ShapeDtypeStruct((8,), jnp.float32),
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
    )
    assert "thread_position_in_threadgroup" not in msl
    assert "threads_per_threadgroup" not in msl


def test_thread_scratch_is_unaffected():
    """Part A's thread-space scratch still declares as `thread`."""

    def kernel(x_ref, o_ref, s_ref):
        s_ref[...] = x_ref[...] * 2.0
        o_ref[...] = s_ref[...]

    msl = palladium.debug_msl(
        kernel,
        jax.ShapeDtypeStruct((8,), jnp.float32),
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        scratch_shapes=[pl.MemorySpace.ANY((8,), jnp.float32)],
    )
    assert "thread float scratch0[8];" in msl
    assert "threadgroup float scratch0" not in msl


# --- tracing -------------------------------------------------------------


def test_barrier_survives_dce():
    """The barrier's JAX effect is what keeps it in the jaxpr.

    A zero-output primitive with no declared effect is dead code and JAX
    removes it before palladium ever sees the kernel jaxpr. A silently
    dropped barrier is a data race, so this guards the effect
    registration directly.
    """
    staged = pl.pallas_call(
        _block_sum_kernel,
        grid=(TG,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((TG,), jnp.float32),
        scratch_shapes=[threadgroup_memory((TG,), jnp.float32)],
    )
    jaxpr = jax.make_jaxpr(staged)(jnp.zeros(TG, jnp.float32))
    names = [str(e.primitive) for e in jaxpr.eqns[0].params["jaxpr"].eqns]
    assert "palladium_barrier" in names


def test_spec_tags_the_address_space():
    staged = pl.pallas_call(
        _block_sum_kernel,
        grid=(TG,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((TG,), jnp.float32),
        scratch_shapes=[threadgroup_memory((TG,), jnp.float32)],
    )
    spec = palladium.trace(staged, jax.ShapeDtypeStruct((TG,), jnp.float32))
    assert [s.space for s in spec.scratch] == ["threadgroup"]
    assert spec.uses_threadgroup


def test_thread_scratch_does_not_set_uses_threadgroup():
    def kernel(x_ref, o_ref, s_ref):
        s_ref[...] = x_ref[...]
        o_ref[...] = s_ref[...]

    staged = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        scratch_shapes=[pl.MemorySpace.ANY((8,), jnp.float32)],
    )
    spec = palladium.trace(staged, jax.ShapeDtypeStruct((8,), jnp.float32))
    assert [s.space for s in spec.scratch] == ["thread"]
    assert not spec.uses_threadgroup


# --- execution -----------------------------------------------------------


def test_block_sum_matches_numpy():
    """Cooperative reduction, grid an exact multiple of the threadgroup."""
    n = TG * 8
    x = np.arange(n, dtype=np.float32)
    np.testing.assert_allclose(
        _block_sum_call(n)(x), _expected_block_sums(x, TG), rtol=1e-6
    )


@pytest.mark.parametrize("n", [100, 33, TG + 1, TG * 3 - 1])
def test_block_sum_partial_tail_group(n):
    """The case a compile-time loop bound would get wrong.

    Metal's non-uniform dispatch leaves a smaller final threadgroup;
    threads_per_threadgroup() reports its true size, so the reduction
    stops before reading slots no thread wrote.
    """
    x = np.arange(n, dtype=np.float32)
    np.testing.assert_allclose(
        _block_sum_call(n)(x), _expected_block_sums(x, TG), rtol=1e-6
    )


def test_reduction_is_actually_cooperative():
    """A group's threads must see each other's writes, not just their own.

    If the barrier or the shared allocation silently degraded to
    per-thread storage, every output would equal its own input; this
    input makes that failure numerically obvious.
    """
    n = TG * 2
    x = np.ones(n, dtype=np.float32)
    got = _block_sum_call(n)(x)
    assert np.allclose(got, TG), got[:4]


def test_smaller_threadgroup_than_extent():
    """Dispatching fewer threads than the declared extent is fine.

    The contract is one-sided: the group must not exceed the array, but
    an over-sized array is only wasted threadgroup memory.
    """
    n = 64
    x = np.arange(n, dtype=np.float32)
    f = _block_sum_call(n, threadgroup=16, extent=TG)
    np.testing.assert_allclose(f(x), _expected_block_sums(x, 16), rtol=1e-6)


# --- contracts -----------------------------------------------------------


def test_implicit_threadgroup_is_rejected():
    """A cooperative kernel must be dispatched with an explicit size.

    threadgroup=None lets the runtime pick, commonly far above the
    declared extent, and thread_index() past that extent writes out of
    bounds with no error -- a silent corruption, so it fails loudly.
    """
    n = 64
    f = _block_sum_call(n, threadgroup=None)
    with pytest.raises(EmitError, match="explicit threadgroup"):
        f(np.arange(n, dtype=np.float32))


def test_ffi_path_requires_an_explicit_threadgroup():
    """metal_call_jit enforces the same contract as bind().

    Without a threadgroup= the jax.ffi path launches with a
    runtime-chosen size, which is the same silent out-of-bounds case the
    eager path guards against.
    """
    n = 64
    f = palladium.metal_call_jit(
        _block_sum_kernel,
        grid=(n,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
        scratch_shapes=[threadgroup_memory((TG,), jnp.float32)],
    )
    with pytest.raises(EmitError, match="explicit"):
        f(jnp.arange(n, dtype=jnp.float32))


def _block_sum_jit(n, threadgroup=TG, extent=TG):
    return palladium.metal_call_jit(
        _block_sum_kernel,
        grid=(n,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
        scratch_shapes=[threadgroup_memory((extent,), jnp.float32)],
        threadgroup=threadgroup,
    )


@pytest.mark.parametrize("n", [TG * 4, 100])
def test_ffi_path_runs_cooperative_kernels(n):
    """Cooperative kernels dispatch through jax.ffi, tail group included.

    Nothing about the FFI path conflicts with threadgroup execution: the
    size is a launch parameter, the MSL is identical, and the native
    handler has always bound threadgroup_x/y/z.
    """
    x = np.arange(n, dtype=np.float32)
    got = np.asarray(_block_sum_jit(n)(jnp.asarray(x)))
    np.testing.assert_allclose(got, _expected_block_sums(x, TG), rtol=1e-6)


def test_ffi_path_composes_under_jit():
    """The whole point of the FFI path: it nests inside jax.jit."""
    n = 100
    x = np.arange(n, dtype=np.float32)
    f = _block_sum_jit(n)
    got = np.asarray(jax.jit(lambda a: f(a) * 2.0 + 1.0)(jnp.asarray(x)))
    np.testing.assert_allclose(got, _expected_block_sums(x, TG) * 2.0 + 1.0, rtol=1e-6)


def test_ffi_explain_reports_the_threadgroup():
    d = _block_sum_jit(64).explain(jax.ShapeDtypeStruct((64,), jnp.float32))
    assert d.threadgroup == (TG,)


def test_interpret_models_a_threadgroup_of_one():
    """Interpret has no threadgroups; it models each instance as a group of 1.

    Pinned deliberately: a cooperative kernel's interpret result is *not*
    its GPU result, which is why the tests above use a NumPy reference
    instead of the usual interpret oracle.
    """
    n = 8
    staged = pl.pallas_call(
        _block_sum_kernel,
        grid=(n,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
        scratch_shapes=[threadgroup_memory((TG,), jnp.float32)],
        interpret=True,
    )
    x = np.arange(n, dtype=np.float32)
    # group-of-one: each instance reduces only its own slot back to itself
    np.testing.assert_allclose(np.asarray(staged(x)), x, rtol=1e-6)


def test_cooperative_effects_pass_the_gpu_effect_gate():
    """The cooperative effect is GPU-native, not a host-side effect.

    `trace()` rejects any jaxpr effect that is neither a Ref state
    effect nor a `GpuNativeEffect`, which is what host callbacks and
    debug prints trip. The cooperative primitives declare an effect only
    so DCE cannot delete them, so they must classify as GPU-native or
    every threadgroup kernel would be refused at trace time.
    """
    from palladium import effects
    from palladium.threadgroup import _ThreadgroupEffect

    assert issubclass(_ThreadgroupEffect, effects.GpuNativeEffect)

    staged = pl.pallas_call(
        _block_sum_kernel,
        grid=(TG,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((TG,), jnp.float32),
        scratch_shapes=[threadgroup_memory((TG,), jnp.float32)],
    )
    spec = palladium.trace(staged, jax.ShapeDtypeStruct((TG,), jnp.float32))
    assert effects.foreign_effects(spec.jaxpr) == []


# --- vmap over cooperative kernels ---------------------------------------


def _batched_block_sums(xb, tg):
    """Per-thread broadcast of each group's sum, for every batch row."""
    out = np.empty_like(xb)
    for start in range(0, xb.shape[-1], tg):
        block = xb[..., start : start + tg]
        out[..., start : start + tg] = block.sum(axis=-1, keepdims=True)
    return out


@pytest.mark.parametrize(
    "vmap_method", ["sequential", "sequential_unrolled", "pipelined"]
)
@pytest.mark.parametrize("n", [TG * 4, 100])
def test_vmap_over_a_cooperative_kernel(vmap_method, n):
    """Every vmap method must preserve threadgroup semantics.

    The concern was that 'pipelined' handles a whole batch in one FFI
    call and might therefore widen the grid, moving threadgroup
    boundaries. It does not: the native handler issues one dispatch per
    batch element with the same grid and threadgroup, varying only the
    buffer offsets, so each element sees exactly the geometry it would
    unbatched. n=100 is not a multiple of TG, so every element also
    exercises the partial tail group.
    """
    batch = 4
    xb = np.stack([np.arange(n, dtype=np.float32) * (j + 1) for j in range(batch)])
    f = palladium.metal_call_jit(
        _block_sum_kernel,
        grid=(n,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
        scratch_shapes=[threadgroup_memory((TG,), jnp.float32)],
        threadgroup=TG,
        vmap_method=vmap_method,
    )
    got = np.asarray(jax.jit(jax.vmap(f))(jnp.asarray(xb)))
    np.testing.assert_allclose(got, _batched_block_sums(xb, TG), rtol=1e-6)


def test_vmap_methods_agree_with_each_other_on_a_cooperative_kernel():
    """Cross-check the three paths rather than only the NumPy reference:
    a shared misunderstanding of the batching contract would show up as
    disagreement between them even if one matched by luck."""
    n, batch = 100, 3
    xb = np.stack([np.arange(n, dtype=np.float32) * (j + 1) for j in range(batch)])

    def run(method):
        f = palladium.metal_call_jit(
            _block_sum_kernel,
            grid=(n,),
            in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
            out_specs=pl.BlockSpec((1,), lambda i: (i,)),
            out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
            scratch_shapes=[threadgroup_memory((TG,), jnp.float32)],
            threadgroup=TG,
            vmap_method=method,
        )
        return np.asarray(jax.jit(jax.vmap(f))(jnp.asarray(xb)))

    base = run("sequential")
    for method in ("sequential_unrolled", "pipelined"):
        np.testing.assert_allclose(run(method), base, rtol=1e-6)


def test_vmap_over_a_cooperative_kernel_still_needs_a_threadgroup():
    """Batching must not become a way around the explicit-size contract."""
    n = 64
    f = palladium.metal_call_jit(
        _block_sum_kernel,
        grid=(n,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
        scratch_shapes=[threadgroup_memory((TG,), jnp.float32)],
        vmap_method="pipelined",
    )
    xb = jnp.zeros((2, n), jnp.float32)
    with pytest.raises(EmitError, match="explicit"):
        jax.jit(jax.vmap(f))(xb)
