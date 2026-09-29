"""The ``palladium.dispatch`` descriptor ABI.

The descriptor tests run on CPU and pin the ABI jax-mps consumes; the
device tests at the end need the jax-mps plugin and skip without it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium


def _add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def test_descriptor_round_trip_is_stable():
    descriptor = palladium.MpsDispatchDescriptor(
        version=2,
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        prologue="const device float* arg0 = (const device float*)arg0_base;",
        body="arg1[_pid.x] = arg0[_pid.x];",
        grid=(8, 1, 1),
        threadgroup=None,
        math_mode=2,
    )
    assert palladium.MpsDispatchDescriptor.from_json(descriptor.to_json()) == descriptor


def _descriptor_for(call, *shapes, dot_general="auto", threadgroup=None):
    spec = palladium.trace(call, *shapes)
    msl = palladium.emit_msl(spec, dot_general=dot_general)
    return msl, palladium.MpsDispatchDescriptor.from_spec(
        spec, msl, threadgroup=threadgroup, math_mode=2
    )


def _tiled_dot_call():
    def dot(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...])

    return pl.pallas_call(
        dot,
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((16, 16), lambda i, j: (i, 0)),
            pl.BlockSpec((16, 32), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((16, 32), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((32, 64), jnp.float32),
    )


_DOT_SHAPES = (
    jax.ShapeDtypeStruct((32, 16), jnp.float32),
    jax.ShapeDtypeStruct((16, 64), jnp.float32),
)


def test_descriptor_splits_an_independent_kernel_into_header_prologue_body():
    call = pl.pallas_call(_add_kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))
    shape = jax.ShapeDtypeStruct((8,), jnp.float32)
    msl, descriptor = _descriptor_for(call, shape, shape)

    assert descriptor.header.startswith("#include <metal_stdlib>")
    assert "kernel void" not in descriptor.header
    assert "kernel void" not in descriptor.body
    assert descriptor.prologue.splitlines() == [
        "const device float* arg0 = (const device float*)arg0_base;",
        "const device float* arg1 = (const device float*)arg1_base;",
        "device float* arg2 = (device float*)arg2_base;",
        "uint3 _pid = uint3(thread_position_in_grid);",
    ]
    assert descriptor.grid == (1, 1, 1)
    assert descriptor.threadgroup is None
    # Nothing is lost: the pieces reassemble the emitted source.
    assert descriptor.header in msl
    assert descriptor.body in msl


def test_descriptor_binds_buffers_by_index_not_by_emitted_name():
    """The attention lowering names its buffers query/key/value/output; the
    handler only knows arg<N>_base."""
    from palladium.workloads.pallas_flash_attention import attention_kernel, attention_specs

    grid, in_specs, out_specs = attention_specs(1, 128, 2, 16, 16)
    call = pl.pallas_call(
        attention_kernel(tile_q=16, tile_k=16, head_dim=16, causal=False),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=jax.ShapeDtypeStruct((1, 128, 2, 16), jnp.float32),
    )
    shape = jax.ShapeDtypeStruct((1, 128, 2, 16), jnp.float32)
    _, descriptor = _descriptor_for(call, shape, shape, shape, dot_general="tensorops")
    lines = descriptor.prologue.splitlines()
    assert "device float* query = (device float*)arg0_base;" in lines
    assert "device float* key = (device float*)arg1_base;" in lines
    assert "device float* value = (device float*)arg2_base;" in lines
    assert "device float* output = (device float*)arg3_base;" in lines
    assert "uint3 group = uint3(threadgroup_position_in_grid);" in lines
    assert "uint tid = uint(thread_index_in_threadgroup);" in lines
    assert not any(
        f"{name}_base" in descriptor.body for name in ("query", "key", "value", "output")
    )


def test_descriptor_rejects_a_mismatched_cooperative_threadgroup():
    with pytest.raises(ValueError, match="cooperative kernel requires threadgroup"):
        _descriptor_for(_tiled_dot_call(), *_DOT_SHAPES, dot_general="tensorops", threadgroup=(64,))


# --- device tests: need the jax-mps plugin -----------------------------------


def _mps_device():
    try:
        return jax.devices("mps")[0]
    except (RuntimeError, IndexError):
        pytest.skip("requires the jax-mps plugin")


def test_plain_pallas_call_runs_through_palladium_on_mps_when_available():
    """One Pallas dispatch can sit between ordinary MPS JAX operations."""
    device = _mps_device()
    call = pl.pallas_call(_add_kernel, out_shape=jax.ShapeDtypeStruct((16,), jnp.float32))

    @jax.jit
    def composed(x, y):
        return jnp.sum(call(x, y) ** 2)

    with jax.default_device(device):
        got = composed(jnp.arange(16, dtype=jnp.float32), jnp.ones(16, dtype=jnp.float32))
    assert got.device.platform == "mps"
    assert float(got) == pytest.approx(1496.0)


def test_cooperative_tensorops_matmul_runs_under_jit_on_mps_when_available():
    """Descriptor v2 carries the MPP header, the threadgroup-position
    prologue, and the scaled launch."""
    device = _mps_device()

    def dot(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...])

    call = pl.pallas_call(
        dot,
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((16, 16), lambda i, j: (i, 0)),
            pl.BlockSpec((16, 32), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((16, 32), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((32, 64), jnp.float32),
        compiler_params=palladium.CompilerParams(dot_general="tensorops"),
    )

    @jax.jit
    def composed(a, b):
        return call(a, b) + 1.0

    rng = np.random.default_rng(3)
    a_np = rng.standard_normal((32, 16), dtype=np.float32)
    b_np = rng.standard_normal((16, 64), dtype=np.float32)
    with jax.default_device(device):
        got = composed(jnp.asarray(a_np), jnp.asarray(b_np))
    assert got.device.platform == "mps"
    np.testing.assert_allclose(np.asarray(got), a_np @ b_np + 1.0, rtol=1e-4, atol=1e-4)
