"""Pallas scratch refs use uninitialized, per-instance storage.

Tests write each element before reading it.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

import palladium
from palladium.trace import ScratchInfo, trace


def _f32(shape):
    return pl.MemorySpace.ANY(shape, jnp.float32)


def test_spec_captures_scratch():
    """trace() lifts grid_mapping.scratch_avals into KernelSpec.scratch."""

    def kernel(x_ref, o_ref, s_ref):
        s_ref[...] = x_ref[...]
        o_ref[...] = s_ref[...]

    staged = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        scratch_shapes=[_f32((8,))],
    )
    spec = trace(staged, jax.ShapeDtypeStruct((8,), jnp.float32))
    assert spec.scratch == (ScratchInfo(shape=(8,), dtype=np.dtype(np.float32)),)
    # Scratch is not an operand: inputs/outputs stay untouched by it.
    assert len(spec.inputs) == 1 and len(spec.outputs) == 1


def test_scratch_roundtrip(rng):
    def kernel(x_ref, o_ref, s_ref):
        s_ref[...] = x_ref[...] * 2.0
        o_ref[...] = s_ref[...] + 1.0

    kwargs = {
        "out_shape": jax.ShapeDtypeStruct((8,), jnp.float32),
        "scratch_shapes": [_f32((8,))],
    }
    f = palladium.metal_call(kernel, **kwargs)
    x = rng.standard_normal(8, dtype=np.float32)
    np.testing.assert_allclose(f(x), np.asarray(f.interpret(x)), rtol=1e-6)


def test_scalar_scratch(rng):
    """Shape-() scratch stays indexable, matching the operand path's axis-1 scalar refs."""

    def kernel(x_ref, o_ref, s_ref):
        s_ref[...] = x_ref[0] + x_ref[1]
        o_ref[...] = jnp.broadcast_to(s_ref[...], (4,))

    kwargs = {
        "out_shape": jax.ShapeDtypeStruct((4,), jnp.float32),
        "scratch_shapes": [pl.MemorySpace.ANY((), jnp.float32)],
    }
    f = palladium.metal_call(kernel, **kwargs)
    x = rng.standard_normal(4, dtype=np.float32)
    np.testing.assert_allclose(f(x), np.asarray(f.interpret(x)), rtol=1e-6)


def test_multiple_scratch_buffers_mixed_dtypes(rng):
    """Scratch entries bind in jaxpr order, each with its own ctype."""

    def kernel(x_ref, o_ref, sf_ref, si_ref):
        sf_ref[...] = x_ref[...] * 3.0
        si_ref[...] = (x_ref[...] > 0.0).astype(jnp.int32)
        o_ref[...] = sf_ref[...] + si_ref[...].astype(jnp.float32)

    kwargs = {
        "out_shape": jax.ShapeDtypeStruct((16,), jnp.float32),
        "scratch_shapes": [_f32((16,)), pl.MemorySpace.ANY((16,), jnp.int32)],
    }
    f = palladium.metal_call(kernel, **kwargs)
    x = rng.standard_normal(16, dtype=np.float32)
    np.testing.assert_allclose(f(x), np.asarray(f.interpret(x)), rtol=1e-6)


def test_scratch_is_private_per_program_instance(rng):
    """Each grid point gets its own scratch; a shared allocation would show up as another row's value."""

    def kernel(x_ref, o_ref, s_ref):
        i = pl.program_id(0)
        s_ref[...] = x_ref[...] + i.astype(jnp.float32)
        o_ref[...] = s_ref[...]

    kwargs = {
        "grid": (4,),
        "in_specs": [pl.BlockSpec((1, 8), lambda i: (i, 0))],
        "out_specs": pl.BlockSpec((1, 8), lambda i: (i, 0)),
        "out_shape": jax.ShapeDtypeStruct((4, 8), jnp.float32),
        "scratch_shapes": [_f32((1, 8))],
    }
    f = palladium.metal_call(kernel, **kwargs)
    x = rng.standard_normal((4, 8), dtype=np.float32)
    expected = x + np.arange(4, dtype=np.float32)[:, None]
    np.testing.assert_allclose(f(x), expected, rtol=1e-6)


def test_scratch_accumulator_across_loop(rng):
    """A fori_loop body accumulates into a scratch Ref instead of threading the value through the carry."""

    def kernel(x_ref, o_ref, s_ref):
        s_ref[...] = jnp.zeros((4,), jnp.float32)

        def body(k, _):
            s_ref[...] = s_ref[...] + x_ref[k, :]
            return 0

        jax.lax.fori_loop(0, 6, body, 0)
        o_ref[...] = s_ref[...]

    kwargs = {
        "out_shape": jax.ShapeDtypeStruct((4,), jnp.float32),
        "scratch_shapes": [_f32((4,))],
    }
    f = palladium.metal_call(kernel, **kwargs)
    x = rng.standard_normal((6, 4), dtype=np.float32)
    np.testing.assert_allclose(f(x), x.sum(axis=0), rtol=1e-5)
