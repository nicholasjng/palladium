"""Palladium's JAX-side jax-mps bridge.

These tests run on CPU deliberately.  They pin the descriptor ABI and verify
that a program containing an MPS call remains portable until jax-mps supplies
the native ``palladium.dispatch`` handler.
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


def _descriptor_for(call, *shapes):
    spec, msl, _ = call._staged._spec_and_msl(tuple(shapes))
    return msl, palladium.MpsDispatchDescriptor.from_spec(
        spec, msl, threadgroup=call._staged._threadgroup, math_mode=2
    )


def test_descriptor_splits_an_independent_kernel_into_header_prologue_body():
    call = palladium.mps_call_jit(_add_kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))
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
    def dot(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...])

    call = palladium.mps_call_jit(
        dot,
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((16, 16), lambda i, j: (i, 0)),
            pl.BlockSpec((16, 32), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((16, 32), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((32, 64), jnp.float32),
        dot_general="tensorops",
    )
    _, descriptor = _descriptor_for(
        call,
        jax.ShapeDtypeStruct((32, 16), jnp.float32),
        jax.ShapeDtypeStruct((16, 64), jnp.float32),
    )
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
    call = palladium.mps_call_jit(
        attention_kernel(tile_q=16, tile_k=16, head_dim=16, causal=False),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=jax.ShapeDtypeStruct((1, 128, 2, 16), jnp.float32),
        dot_general="tensorops",
    )
    shape = jax.ShapeDtypeStruct((1, 128, 2, 16), jnp.float32)
    _, descriptor = _descriptor_for(call, shape, shape, shape)
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
    def dot(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...])

    call = palladium.mps_call_jit(
        dot,
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((16, 16), lambda i, j: (i, 0)),
            pl.BlockSpec((16, 32), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((16, 32), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((32, 64), jnp.float32),
        dot_general="tensorops",
        threadgroup=64,
    )
    with pytest.raises(ValueError, match="cooperative kernel requires threadgroup"):
        _descriptor_for(
            call,
            jax.ShapeDtypeStruct((32, 16), jnp.float32),
            jax.ShapeDtypeStruct((16, 64), jnp.float32),
        )


def test_manual_rk4_step_transpose_matches_jax_vjp():
    values = tuple(jnp.asarray(value, jnp.float32) for value in (1.1, 0.9, 1.0, 0.4, 0.1, 0.4))
    cotangents = (jnp.asarray(0.7, jnp.float32), jnp.asarray(-0.3, jnp.float32))
    _, pullback = jax.vjp(_lv_rk4_step, *values)
    expected = pullback(cotangents)
    actual = _lv_rk4_step_vjp(*values, *cotangents)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-6, atol=2e-7)


def test_mps_call_has_a_portable_cpu_fallback():
    call = palladium.mps_call_jit(_add_kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))
    x = jnp.arange(8, dtype=jnp.float32)
    y = jnp.ones(8, dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(call(x, y)), np.asarray(x + y))
    np.testing.assert_array_equal(
        np.asarray(jax.jit(lambda a, b: call(a, b))(x, y)), np.asarray(x + y)
    )
    np.testing.assert_array_equal(np.asarray(call.verify(x, y)), np.asarray(x + y))


def test_mps_call_requires_batch_in_the_grid():
    with pytest.raises(ValueError, match="batch dimension in the Pallas grid"):
        palladium.mps_call_jit(
            _add_kernel,
            out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
            vmap_method="sequential",
        )


def test_mps_call_validates_the_dot_general_policy():
    with pytest.raises(ValueError, match="dot_general must be"):
        palladium.mps_call_jit(
            _add_kernel,
            out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
            dot_general="simdgroup",
        )


def test_mps_call_reference_vjp_enables_cpu_training_fallback():
    def reference(x, y):
        return x + y

    call = palladium.mps_call_jit(
        _add_kernel,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        vjp_reference=reference,
    )
    x = jnp.arange(8, dtype=jnp.float32)
    y = jnp.ones(8, dtype=jnp.float32)
    loss = lambda a, b: jnp.sum(call(a, b) ** 2)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jax.grad(loss, argnums=(0, 1)))(x, y)),
        np.asarray((2 * (x + y), 2 * (x + y))),
    )


def test_mps_call_accepts_a_pallas_backward_kernel():
    forward = palladium.mps_call_jit(_add_kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))
    backward = palladium.mps_call_jit(
        _add_vjp_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((8,), jnp.float32),
            jax.ShapeDtypeStruct((8,), jnp.float32),
        ),
    )
    call = forward.with_vjp(backward)
    x = jnp.arange(8, dtype=jnp.float32)
    y = jnp.ones(8, dtype=jnp.float32)
    loss = lambda a, b: jnp.sum(call(a, b) ** 2)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jax.grad(loss, argnums=(0, 1)))(x, y)),
        np.asarray((2 * (x + y), 2 * (x + y))),
    )


def test_mps_call_can_save_auxiliaries_for_its_pallas_vjp():
    forward = palladium.mps_call_jit(
        _add_with_auxiliary_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((8,), jnp.float32),
            jax.ShapeDtypeStruct((8,), jnp.float32),
        ),
    )
    backward = palladium.mps_call_jit(
        _add_with_auxiliary_vjp_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((8,), jnp.float32),
            jax.ShapeDtypeStruct((8,), jnp.float32),
        ),
    )
    call = forward.with_auxiliary_vjp(backward, output_count=1)
    x = jnp.arange(8, dtype=jnp.float32)
    y = jnp.ones(8, dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jax.grad(lambda a, b: jnp.sum(call(a, b) ** 2), (0, 1)))(x, y)),
        np.asarray((2 * (x + y), 2 * (x + y))),
    )


def test_mps_call_can_emit_per_trajectory_checkpoint_rows():
    n = 8
    point = pl.BlockSpec((1,), lambda i: (i,))
    checkpoint_row = pl.BlockSpec((1, 3), lambda i: (i, 0))
    call = palladium.mps_call_jit(
        _checkpoint_kernel,
        grid=(n,),
        in_specs=[point],
        out_specs=(point, checkpoint_row),
        out_shape=(
            jax.ShapeDtypeStruct((n,), jnp.float32),
            jax.ShapeDtypeStruct((n, 3), jnp.float32),
        ),
    )
    x = jnp.arange(n, dtype=jnp.float32)
    final, got_checkpoints = jax.jit(call)(x)
    np.testing.assert_array_equal(np.asarray(final), np.asarray(x + 3))
    np.testing.assert_array_equal(
        np.asarray(got_checkpoints), np.asarray(jnp.stack((x, x + 1, x + 2), axis=1))
    )


def test_mps_call_refuses_aliases_until_the_mlx_path_can_honor_them():
    def inplace(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1.0

    call = palladium.mps_call_jit(
        inplace,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        input_output_aliases={0: 0},
    )
    with pytest.raises(ValueError, match="input_output_aliases"):
        call(jnp.zeros(8, jnp.float32))


def test_mps_call_composes_with_jax_mps_when_available():
    """One Pallas dispatch can sit between ordinary MPS JAX operations.

    This is intentionally optional in Palladium's own environment: jax-mps is
    a sibling integration, not a package dependency. The jax-mps development
    environment runs this test as the end-to-end ABI gate.
    """
    try:
        device = jax.devices("mps")[0]
    except (RuntimeError, IndexError):
        pytest.skip("requires the jax-mps plugin")

    call = palladium.mps_call_jit(_add_kernel, out_shape=jax.ShapeDtypeStruct((16,), jnp.float32))

    @jax.jit
    def composed(x, y):
        return jnp.sum(call(x, y) ** 2)

    with jax.default_device(device):
        x = jnp.arange(16, dtype=jnp.float32)
        y = jnp.ones(16, dtype=jnp.float32)
        got = composed(x, y)
    assert got.device.platform == "mps"
    assert float(got) == pytest.approx(1496.0)


def test_cooperative_tensorops_matmul_runs_under_jit_on_mps_when_available():
    """A cooperative kernel through jax-mps: descriptor v2 carries the MPP
    header, the threadgroup-position prologue, and the scaled launch."""
    try:
        device = jax.devices("mps")[0]
    except (RuntimeError, IndexError):
        pytest.skip("requires the jax-mps plugin")

    def dot(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...])

    call = palladium.mps_call_jit(
        dot,
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((16, 16), lambda i, j: (i, 0)),
            pl.BlockSpec((16, 32), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((16, 32), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((32, 64), jnp.float32),
        dot_general="tensorops",
        fallback="error",
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
