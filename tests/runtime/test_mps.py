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
        version=1,
        msl_source="kernel void add() {}",
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
