"""Backend selection and policy for cooperative operations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import palladium


def test_backend_diagnostics_and_strict_policy():
    def copy_kernel(x, out):
        out[...] = x[...] * 2

    arg = jax.ShapeDtypeStruct((4,), np.float32)
    eager = palladium.metal_call(copy_kernel, out_shape=arg)
    ffi = palladium.metal_call_jit(copy_kernel, out_shape=arg)
    mps = palladium.mps_call_jit(copy_kernel, out_shape=arg)
    strict = palladium.mps_call_jit(copy_kernel, out_shape=arg, fallback="error")
    assert eager.explain(arg).execution_path == "metal"
    assert ffi.explain(arg).execution_path == "cpu-ffi-to-metal"
    assert mps.explain(arg, platform="cpu").execution_path == "pallas-interpret:cpu"
    assert mps.explain(arg, platform="mps").execution_path == "mps-custom-call"
    assert "rejected" in strict.explain(arg, platform="cpu").execution_path
    with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.arange(4, dtype=jnp.float32)
        np.testing.assert_array_equal(jax.jit(mps)(x), np.arange(4) * 2)
        with pytest.raises(ValueError, match="MPS backend required"):
            jax.jit(strict)(x)
        with pytest.raises(ValueError, match="MPS backend required"):
            strict(x)


def test_cooperative_mps_fallback_rejected():
    def kernel(out):
        out[0] = palladium.thread_index()

    call = palladium.mps_call_jit(
        kernel, threadgroup=4, out_shape=jax.ShapeDtypeStruct((1,), np.int32)
    )
    with (
        jax.default_device(jax.devices("cpu")[0]),
        pytest.raises(ValueError, match="threadgroups of one"),
    ):
        jax.jit(call)()
