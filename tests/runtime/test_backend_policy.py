"""Backend selection diagnostics."""

import jax
import numpy as np

import palladium


def test_backend_diagnostics_name_their_dispatch_route():
    def copy_kernel(x, out):
        out[...] = x[...] * 2

    arg = jax.ShapeDtypeStruct((4,), np.float32)
    eager = palladium.metal_call(copy_kernel, out_shape=arg)
    ffi = palladium.metal_call_jit(copy_kernel, out_shape=arg)
    assert eager.explain(arg).execution_path == "metal"
    assert ffi.explain(arg).execution_path == "cpu-ffi-to-metal"
