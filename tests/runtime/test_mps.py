"""The ``palladium.dispatch`` descriptor ABI and the VJP helpers.

The descriptor tests run on CPU and pin the ABI jax-mps consumes; the VJP
tests run the kernels on the Pallas interpreter; the device tests at the
end need the jax-mps plugin and skip without it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium


def _lv_rk4_step(x, y, a, b, c, d, dt=0.01):
    def rhs(x_, y_):
        return a * x_ - b * x_ * y_, c * x_ * y_ - d * y_

    k1x, k1y = rhs(x, y)
    k2x, k2y = rhs(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
    k3x, k3y = rhs(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
    k4x, k4y = rhs(x + dt * k3x, y + dt * k3y)
    return (
        x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x),
        y + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y),
    )


def _lv_rk4_step_vjp(x, y, a, b, c, d, bar_x, bar_y, dt=0.01):
    """Manual transpose of the *discrete* RK4 update, in reverse order."""

    def rhs(x_, y_):
        return a * x_ - b * x_ * y_, c * x_ * y_ - d * y_

    def transpose_rhs(x_, y_, bar_fx, bar_fy):
        return (
            (a - b * y_) * bar_fx + c * y_ * bar_fy,
            -b * x_ * bar_fx + (c * x_ - d) * bar_fy,
            x_ * bar_fx,
            -x_ * y_ * bar_fx,
            x_ * y_ * bar_fy,
            -y_ * bar_fy,
        )

    k1x, k1y = rhs(x, y)
    x2, y2 = x + 0.5 * dt * k1x, y + 0.5 * dt * k1y
    k2x, k2y = rhs(x2, y2)
    x3, y3 = x + 0.5 * dt * k2x, y + 0.5 * dt * k2y
    k3x, k3y = rhs(x3, y3)
    x4, y4 = x + dt * k3x, y + dt * k3y
    k4x, k4y = rhs(x4, y4)
    del k4x, k4y

    bar_a = bar_b = bar_c = bar_d = x * 0
    bar_k1x, bar_k1y = dt / 6 * bar_x, dt / 6 * bar_y
    bar_k2x, bar_k2y = dt / 3 * bar_x, dt / 3 * bar_y
    bar_k3x, bar_k3y = dt / 3 * bar_x, dt / 3 * bar_y
    bar_k4x, bar_k4y = dt / 6 * bar_x, dt / 6 * bar_y
    bar_x4, bar_y4, da, db, dc, dd = transpose_rhs(x4, y4, bar_k4x, bar_k4y)
    bar_a, bar_b, bar_c, bar_d = bar_a + da, bar_b + db, bar_c + dc, bar_d + dd
    # x4/y4 = x/y + dt*k3; add their adjoints before transposing k3.
    bar_x0, bar_y0 = bar_x + bar_x4, bar_y + bar_y4
    bar_k3x, bar_k3y = bar_k3x + dt * bar_x4, bar_k3y + dt * bar_y4
    bar_x3, bar_y3, da, db, dc, dd = transpose_rhs(x3, y3, bar_k3x, bar_k3y)
    bar_a, bar_b, bar_c, bar_d = bar_a + da, bar_b + db, bar_c + dc, bar_d + dd
    bar_x0, bar_y0 = bar_x0 + bar_x3, bar_y0 + bar_y3
    bar_k2x, bar_k2y = bar_k2x + 0.5 * dt * bar_x3, bar_k2y + 0.5 * dt * bar_y3
    bar_x2, bar_y2, da, db, dc, dd = transpose_rhs(x2, y2, bar_k2x, bar_k2y)
    bar_a, bar_b, bar_c, bar_d = bar_a + da, bar_b + db, bar_c + dc, bar_d + dd
    bar_x0, bar_y0 = bar_x0 + bar_x2, bar_y0 + bar_y2
    bar_k1x, bar_k1y = bar_k1x + 0.5 * dt * bar_x2, bar_k1y + 0.5 * dt * bar_y2
    dx, dy, da, db, dc, dd = transpose_rhs(x, y, bar_k1x, bar_k1y)
    return bar_x0 + dx, bar_y0 + dy, bar_a + da, bar_b + db, bar_c + dc, bar_d + dd


def _add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def _add_vjp_kernel(x_ref, y_ref, cotangent_ref, x_gradient_ref, y_gradient_ref):
    del x_ref, y_ref
    x_gradient_ref[...] = cotangent_ref[...]
    y_gradient_ref[...] = cotangent_ref[...]


def _add_with_auxiliary_kernel(x_ref, y_ref, output_ref, auxiliary_ref):
    output_ref[...] = x_ref[...] + y_ref[...]
    auxiliary_ref[...] = x_ref[...]


def _add_with_auxiliary_vjp_kernel(
    x_ref, y_ref, auxiliary_ref, cotangent_ref, x_gradient_ref, y_gradient_ref
):
    del x_ref, y_ref, auxiliary_ref
    x_gradient_ref[...] = cotangent_ref[...]
    y_gradient_ref[...] = cotangent_ref[...]


def _checkpoint_kernel(x_ref, final_ref, checkpoints_ref):
    x = x_ref[...]
    for checkpoint in range(3):
        checkpoints_ref[:, checkpoint] = x
        x = x + 1
    final_ref[...] = x


def test_descriptor_round_trip_is_stable():
    descriptor = palladium.MpsDispatchDescriptor(
        version=2,
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        prologue="const device float* arg0 = (const device float*)arg0_base;",
        body="arg1[_pid.x] = arg0[_pid.x];",
        function_name="add",
        grid=(8, 1, 1),
        threadgroup=None,
        math_mode=2,
        input_shapes=((8,),),
        input_dtypes=("<f4",),
        output_shapes=((8,),),
        output_dtypes=("<f4",),
        aliases=(),
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


def test_descriptor_scales_the_launch_for_a_cooperative_tensorops_kernel():
    _, descriptor = _descriptor_for(_tiled_dot_call(), *_DOT_SHAPES, dot_general="tensorops")
    width = palladium.diagnostics.simdgroup_width()

    assert "MetalPerformancePrimitives" in descriptor.header
    assert "using namespace mpp;" in descriptor.header
    assert descriptor.threadgroup == (4 * width, 1, 1)
    assert descriptor.grid == (2 * 4 * width, 2, 1)
    lines = descriptor.prologue.splitlines()
    assert "device float* arg0 = (device float*)arg0_base;" in lines
    assert "uint3 _pid = uint3(threadgroup_position_in_grid);" in lines
    assert "matmul2d" in descriptor.body


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


def test_manual_rk4_step_transpose_matches_jax_vjp():
    values = tuple(jnp.asarray(value, jnp.float32) for value in (1.1, 0.9, 1.0, 0.4, 0.1, 0.4))
    cotangents = (jnp.asarray(0.7, jnp.float32), jnp.asarray(-0.3, jnp.float32))
    _, pullback = jax.vjp(_lv_rk4_step, *values)
    expected = pullback(cotangents)
    actual = _lv_rk4_step_vjp(*values, *cotangents)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-6, atol=2e-7)


# --- VJP helpers, run on the interpreter -------------------------------------


def _interpreted(kernel, out_shape):
    return pl.pallas_call(kernel, out_shape=out_shape, interpret=True)


def test_reference_vjp_pairs_a_forward_call_with_a_jax_pullback():
    call = palladium.with_reference_vjp(
        _interpreted(_add_kernel, jax.ShapeDtypeStruct((8,), jnp.float32)),
        lambda x, y: x + y,
    )
    x = jnp.arange(8, dtype=jnp.float32)
    y = jnp.ones(8, dtype=jnp.float32)
    loss = lambda a, b: jnp.sum(call(a, b) ** 2)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jax.grad(loss, argnums=(0, 1)))(x, y)),
        np.asarray((2 * (x + y), 2 * (x + y))),
    )


def test_vjp_accepts_a_pallas_backward_kernel():
    forward = _interpreted(_add_kernel, jax.ShapeDtypeStruct((8,), jnp.float32))
    backward = _interpreted(
        _add_vjp_kernel,
        (jax.ShapeDtypeStruct((8,), jnp.float32), jax.ShapeDtypeStruct((8,), jnp.float32)),
    )
    call = palladium.with_vjp(forward, backward)
    x = jnp.arange(8, dtype=jnp.float32)
    y = jnp.ones(8, dtype=jnp.float32)
    loss = lambda a, b: jnp.sum(call(a, b) ** 2)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jax.grad(loss, argnums=(0, 1)))(x, y)),
        np.asarray((2 * (x + y), 2 * (x + y))),
    )


def test_auxiliary_vjp_saves_trailing_outputs_for_the_backward_kernel():
    forward = _interpreted(
        _add_with_auxiliary_kernel,
        (jax.ShapeDtypeStruct((8,), jnp.float32), jax.ShapeDtypeStruct((8,), jnp.float32)),
    )
    backward = _interpreted(
        _add_with_auxiliary_vjp_kernel,
        (jax.ShapeDtypeStruct((8,), jnp.float32), jax.ShapeDtypeStruct((8,), jnp.float32)),
    )
    call = palladium.with_auxiliary_vjp(forward, backward, output_count=1)
    x = jnp.arange(8, dtype=jnp.float32)
    y = jnp.ones(8, dtype=jnp.float32)
    assert np.asarray(call(x, y)).shape == (8,)
    loss = lambda a, b: jnp.sum(call(a, b) ** 2)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jax.grad(loss, argnums=(0, 1)))(x, y)),
        np.asarray((2 * (x + y), 2 * (x + y))),
    )
    with pytest.raises(ValueError, match="trailing auxiliary output"):
        palladium.with_auxiliary_vjp(forward, backward, output_count=2)(x, y)


def test_checkpoint_rows_come_back_as_trailing_outputs():
    call = _interpreted(
        _checkpoint_kernel,
        (jax.ShapeDtypeStruct((4,), jnp.float32), jax.ShapeDtypeStruct((4, 3), jnp.float32)),
    )
    final, checkpoints = call(jnp.zeros(4, jnp.float32))
    np.testing.assert_array_equal(np.asarray(final), np.full(4, 3.0))
    np.testing.assert_array_equal(np.asarray(checkpoints), np.tile(np.arange(3.0), (4, 1)))


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
