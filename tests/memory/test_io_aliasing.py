"""input_output_aliases: in-place kernels, one buffer behind the aliased
input/output pair, on the eager and the jax.ffi paths."""

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
import pytest

import palladium
from palladium.errors import DispatchError, TraceError

F32 = jnp.float32


def _double_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0


def test_eager_aliased_call_matches_interpret(rng):
    f = palladium.metal_call(
        _double_kernel,
        out_shape=jax.ShapeDtypeStruct((64,), F32),
        input_output_aliases={0: 0},
    )
    x = rng.standard_normal(64).astype(np.float32)
    got = f(x)
    np.testing.assert_allclose(got, np.asarray(f.interpret(x)), rtol=1e-6)
    # The caller's array is copied at upload, so it must stay untouched.
    np.testing.assert_array_equal(x, x.copy())


def test_eager_aliased_repeated_and_deferred_launches(rng):
    """Slot reuse rewrites the shared buffer; results returned earlier
    (or waited later) must survive as copies."""
    from palladium.dispatch import bind
    from palladium.emit import emit_msl
    from palladium.trace import trace

    call = pl.pallas_call(
        _double_kernel,
        out_shape=jax.ShapeDtypeStruct((256,), F32),
        input_output_aliases={0: 0},
    )
    spec = trace(call, jax.ShapeDtypeStruct((256,), F32))
    bound = bind(spec, emit_msl(spec), pipeline_depth=2)

    xs = [rng.standard_normal(256).astype(np.float32) for _ in range(7)]
    pendings = [bound.launch(x) for x in xs]
    for x, pending in zip(xs, pendings, strict=True):
        np.testing.assert_allclose(pending.wait(), x * 2.0, rtol=1e-6)
    # In-place contract: the output buffer is the slot's input buffer.
    last = pendings[-1]
    assert any(last.out_bufs[0] is buf for slot in bound._slots for buf in slot.in_bufs)


def test_aliased_kernel_grid_and_second_input(rng):
    """Alias one of two inputs, under a blocked grid."""

    def kernel(x_ref, w_ref, o_ref):
        o_ref[...] = x_ref[...] * w_ref[...]

    f = palladium.metal_call(
        kernel,
        grid=(4,),
        in_specs=[
            pl.BlockSpec((16,), lambda i: i),
            pl.BlockSpec((16,), lambda i: i),
        ],
        out_specs=pl.BlockSpec((16,), lambda i: i),
        out_shape=jax.ShapeDtypeStruct((64,), F32),
        input_output_aliases={0: 0},
    )
    x = rng.standard_normal(64).astype(np.float32)
    w = rng.standard_normal(64).astype(np.float32)
    np.testing.assert_allclose(f(x, w), np.asarray(f.interpret(x, w)), rtol=1e-6)


def test_ffi_aliased_call_matches_interpret_under_jit(rng):
    f = palladium.metal_call_jit(
        _double_kernel,
        out_shape=jax.ShapeDtypeStruct((64,), F32),
        input_output_aliases={0: 0},
    )
    x = rng.standard_normal(64).astype(np.float32)
    got = np.asarray(jax.jit(f)(x))
    np.testing.assert_allclose(got, np.asarray(f.interpret(x)), rtol=1e-6)


def test_ffi_aliased_pipelined_vmap(rng):
    f = palladium.metal_call_jit(
        _double_kernel,
        out_shape=jax.ShapeDtypeStruct((16,), F32),
        input_output_aliases={0: 0},
        vmap_method="pipelined",
    )
    xs = rng.standard_normal((11, 16)).astype(np.float32)
    got = np.asarray(jax.vmap(f)(xs))
    np.testing.assert_allclose(got, xs * 2.0, rtol=1e-6)


def test_read_after_aliased_write_is_rejected(rng):
    """The interpret oracle keeps pre-call input values visible for the
    whole kernel; one shared buffer cannot, so this must be rejected."""

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0
        o_ref[0] = x_ref[1]  # reads the input after the aliased write

    f = palladium.metal_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8,), F32),
        input_output_aliases={0: 0},
    )
    with pytest.raises(TraceError, match="read after"):
        f(rng.standard_normal(8).astype(np.float32))


def test_mismatched_alias_pair_is_rejected(rng):
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[0:8] + x_ref[8:16]

    f = palladium.metal_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8,), F32),  # half the input's size
        input_output_aliases={0: 0},
    )
    with pytest.raises((TraceError, ValueError)):
        f(rng.standard_normal(16).astype(np.float32))


def test_pinned_is_rejected_for_aliased_kernels(rng):
    from palladium.dispatch import bind
    from palladium.emit import emit_msl
    from palladium.trace import trace

    call = pl.pallas_call(
        _double_kernel,
        out_shape=jax.ShapeDtypeStruct((8,), F32),
        input_output_aliases={0: 0},
    )
    spec = trace(call, jax.ShapeDtypeStruct((8,), F32))
    bound = bind(spec, emit_msl(spec))
    with pytest.raises(DispatchError, match="pinned"):
        bound.pinned(rng.standard_normal(8).astype(np.float32))
