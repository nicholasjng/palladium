"""The effects layer: foreign-effect rejection at trace time, and the
read/write ref sets the emitter builds on.

Trace-only, so this module runs without a Metal device.
"""

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import pytest

from palladium import effects
from palladium.errors import TraceError
from palladium.trace import trace

F32 = jnp.float32


def _trace(kernel, in_shapes, out_shape):
    call = pl.pallas_call(kernel, out_shape=out_shape)
    return trace(call, *[jax.ShapeDtypeStruct(s, F32) for s in in_shapes])


def test_pallas_debug_print_is_rejected_at_trace_time():
    """Verify that a Pallas debug-print effect inside a kernel is rejected during tracing."""

    def kernel(x_ref, o_ref):
        pl.debug_print("x0={}", x_ref[0])
        o_ref[...] = x_ref[...]

    with pytest.raises(TraceError, match="cannot perform on the GPU"):
        _trace(kernel, [(8,)], jax.ShapeDtypeStruct((8,), F32))


def test_effects_surface_through_control_flow():
    """Check that reads nested in a loop appear in the enclosing equation's aggregate Ref effects."""

    def kernel(x_ref, o_ref):
        def body(i, acc):
            return acc + x_ref[i]

        o_ref[0] = jax.lax.fori_loop(0, 8, body, 0.0)

    spec = _trace(kernel, [(8,)], jax.ShapeDtypeStruct((1,), F32))
    x_ref, o_ref = spec.jaxpr.invars
    loop = next(e for e in spec.jaxpr.eqns if e.primitive.name == "scan")
    assert effects.eqn_reads_ref(loop, x_ref)
    assert not effects.eqn_writes_ref(loop, o_ref)
