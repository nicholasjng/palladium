"""Palladium's JAX-side jax-mps bridge.

These tests run on CPU deliberately.  They pin the descriptor ABI and verify
that a program containing an MPS call remains portable until jax-mps supplies
the native ``palladium.dispatch`` handler.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import palladium


def _add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


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


def test_mps_call_has_a_portable_cpu_fallback():
    call = palladium.mps_call_jit(
        _add_kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32)
    )
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

    call = palladium.mps_call_jit(
        _add_kernel, out_shape=jax.ShapeDtypeStruct((16,), jnp.float32)
    )

    @jax.jit
    def composed(x, y):
        return jnp.sum(call(x, y) ** 2)

    with jax.default_device(device):
        x = jnp.arange(16, dtype=jnp.float32)
        y = jnp.ones(16, dtype=jnp.float32)
        got = composed(x, y)
    assert got.device.platform == "mps"
    assert float(got) == pytest.approx(1496.0)
