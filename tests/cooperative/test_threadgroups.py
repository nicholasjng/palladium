"""Threadgroup-shared scratch, barriers, and cooperative reductions.

The reference is NumPy, not `interpret`: Pallas's interpret mode runs
program instances sequentially and models each as a group of one, so a
kernel that reduces across threads computes something different there.

Removing the barrier emission still produces correct results at every
threadgroup size up to 1024 (measured), so the race is latent and only the
text assertion in `test_barrier_is_emitted_where_the_author_put_it` guards it.

Metal dispatches non-uniform threadgroups, so a grid that is not a multiple
of the threadgroup size ends in a smaller group; `threads_per_threadgroup()`
reports that group's true size.
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
        compiler_params=palladium.CompilerParams(threadgroup=threadgroup),
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


def test_barrier_is_emitted_where_the_author_put_it():
    """barrier() lowers verbatim; placement is never inferred."""
    body = [ln.strip() for ln in _msl().splitlines()]
    assert "threadgroup_barrier(mem_flags::mem_threadgroup);" in body


# --- tracing -------------------------------------------------------------


def test_barrier_survives_dce():
    """The barrier's JAX effect keeps it in the jaxpr; a zero-output primitive without a declared effect is removed as dead code."""
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
    np.testing.assert_allclose(_block_sum_call(n)(x), _expected_block_sums(x, TG), rtol=1e-6)


@pytest.mark.parametrize("n", [100, TG + 1, TG * 3 - 1])
def test_block_sum_partial_tail_group(n):
    """Metal's non-uniform dispatch leaves a smaller final threadgroup, and threads_per_threadgroup() reports its true size so the reduction stops before reading slots no thread wrote."""
    x = np.arange(n, dtype=np.float32)
    np.testing.assert_allclose(_block_sum_call(n)(x), _expected_block_sums(x, TG), rtol=1e-6)


def test_reduction_is_actually_cooperative():
    """A group's threads see each other's writes; with all-ones input, per-thread storage would return 1 instead of TG."""
    n = TG * 2
    x = np.ones(n, dtype=np.float32)
    got = _block_sum_call(n)(x)
    assert np.allclose(got, TG), got[:4]


def test_smaller_threadgroup_than_extent():
    """Dispatching fewer threads than the declared extent is allowed; only the reverse is an error."""
    n = 64
    x = np.arange(n, dtype=np.float32)
    f = _block_sum_call(n, threadgroup=16, extent=TG)
    np.testing.assert_allclose(f(x), _expected_block_sums(x, 16), rtol=1e-6)


# --- contracts -----------------------------------------------------------


def test_implicit_threadgroup_is_rejected():
    """A cooperative kernel must be dispatched with an explicit threadgroup size, since a runtime-chosen one can exceed the declared extent and write out of bounds silently."""
    n = 64
    f = _block_sum_call(n, threadgroup=None)
    with pytest.raises(EmitError, match="explicit threadgroup"):
        f(np.arange(n, dtype=np.float32))


def test_ffi_path_requires_an_explicit_threadgroup():
    """metal_call enforces the same explicit-threadgroup contract as bind()."""
    n = 64
    f = palladium.metal_call(
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
    return palladium.metal_call(
        _block_sum_kernel,
        grid=(n,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
        scratch_shapes=[threadgroup_memory((extent,), jnp.float32)],
        compiler_params=palladium.CompilerParams(threadgroup=threadgroup),
    )


@pytest.mark.parametrize("n", [TG * 4, 100])
def test_ffi_path_runs_cooperative_kernels(n):
    """Cooperative kernels dispatch through jax.ffi, tail group included."""
    x = np.arange(n, dtype=np.float32)
    got = np.asarray(_block_sum_jit(n)(jnp.asarray(x)))
    np.testing.assert_allclose(got, _expected_block_sums(x, TG), rtol=1e-6)


def test_ffi_path_composes_under_jit():
    """The FFI path nests inside jax.jit."""
    n = 100
    x = np.arange(n, dtype=np.float32)
    f = _block_sum_jit(n)
    got = np.asarray(jax.jit(lambda a: f(a) * 2.0 + 1.0)(jnp.asarray(x)))
    np.testing.assert_allclose(got, _expected_block_sums(x, TG) * 2.0 + 1.0, rtol=1e-6)


def test_ffi_explain_reports_the_threadgroup():
    d = _block_sum_jit(64).explain(jax.ShapeDtypeStruct((64,), jnp.float32))
    assert d.threadgroup == (TG, 1, 1)


def test_interpret_models_a_threadgroup_of_one():
    """Interpret models each instance as a group of one, so a cooperative kernel's interpret result is not its GPU result."""
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
    # Each instance reduces only its own slot back to itself.
    np.testing.assert_allclose(np.asarray(staged(x)), x, rtol=1e-6)


def test_cooperative_effects_pass_the_gpu_effect_gate():
    """The cooperative effect classifies as GpuNativeEffect, so trace()'s foreign-effect gate does not refuse threadgroup kernels."""
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


@pytest.mark.parametrize("n", [TG * 4, 100])
def test_vmap_over_a_cooperative_kernel(n):
    """vmap preserves threadgroup geometry: the native handler issues one dispatch per batch element with the same grid and threadgroup, varying only buffer offsets.

    n=100 is not a multiple of TG, so it also exercises the partial tail group.
    """
    batch = 4
    xb = np.stack([np.arange(n, dtype=np.float32) * (j + 1) for j in range(batch)])
    f = palladium.metal_call(
        _block_sum_kernel,
        grid=(n,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
        scratch_shapes=[threadgroup_memory((TG,), jnp.float32)],
        compiler_params=palladium.CompilerParams(threadgroup=TG),
    )
    got = np.asarray(jax.jit(jax.vmap(f))(jnp.asarray(xb)))
    np.testing.assert_allclose(got, _batched_block_sums(xb, TG), rtol=1e-6)


def test_vmap_over_a_cooperative_kernel_still_needs_a_threadgroup():
    """Batching must not become a way around the explicit-size contract."""
    n = 64
    f = palladium.metal_call(
        _block_sum_kernel,
        grid=(n,),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
        scratch_shapes=[threadgroup_memory((TG,), jnp.float32)],
    )
    xb = jnp.zeros((2, n), jnp.float32)
    with pytest.raises(EmitError, match="explicit"):
        jax.jit(jax.vmap(f))(xb)
