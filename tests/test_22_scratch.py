"""Per-invocation scratch operands (`scratch_shapes=`).

Pallas appends scratch Refs to the kernel jaxpr's invars past
`*ins, *outs`, backed by no caller array. They live in Metal's `thread`
address space: private to the one thread running that program instance,
allocated and discarded within a single kernel invocation. No
`[[buffer(k)]]` binding, so `dispatch.py` never sees them.

Storage is deliberately left uninitialized, matching upstream Pallas's
default (no `poison_buffers` support). Every test here therefore writes
a scratch element before reading it; a test that read scratch first
would be asserting on garbage.

Cross-thread visibility is out of scope for this tier -- that is
threadgroup-space scratch, and it needs barriers (Part B of
docs/notes/scratch-and-threadgroup-plan.md).
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


def test_scratch_storage_is_declared():
    """The emitted MSL must declare every scratch name it references.

    Directly guards the failure mode where a scratch CVal is bound into
    `ref_vals` but its declaration never reaches the Cursor: the kernel
    body then references an undeclared identifier and only fails later,
    inside the Metal compiler.
    """

    def kernel(x_ref, o_ref, s_ref):
        s_ref[...] = x_ref[...] * 2.0
        o_ref[...] = s_ref[...] + 1.0

    msl = palladium.debug_msl(
        kernel,
        jax.ShapeDtypeStruct((8,), jnp.float32),
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        scratch_shapes=[_f32((8,))],
    )
    body = [ln.strip() for ln in msl.splitlines()]
    used = {
        ln.split("[")[0].strip()
        for ln in body
        if ln.startswith("scratch") and "[" in ln
    }
    assert used, "no scratch reference in the emitted body; test is vacuous"
    for name in used:
        assert any(
            ln.startswith(("float ", "thread float ")) and name + "[" in ln
            for ln in body
        ), f"{name} is referenced but never declared:\n{msl}"


def test_scratch_roundtrip(rng):
    """Write to scratch, read it back: the minimal end-to-end path."""

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
    """Shape-() scratch: a single thread-local, not an array.

    The operand path represents scalar refs as axis-1 arrays because
    operands are pointers. Thread-space scratch is not a pointer, so a
    scalar entry has to stay indexable to match.
    """

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
    """Each grid point gets its own scratch; no bleed between threads.

    Every instance writes its own program_id-derived value into scratch
    and reads it straight back, so a shared or aliased allocation shows
    up as another row's value.
    """

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
    """The motivating use case: scratch as a loop-carried accumulator.

    A fori_loop body that stores into a scratch Ref rather than threading
    the running value through the carry.
    """

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
