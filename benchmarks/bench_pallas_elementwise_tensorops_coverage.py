"""Device correctness check for standalone cooperative pointwise lowering.

Run on Apple Silicon with an MSL 4 capable GPU using
``JAX_PLATFORMS=mps,cpu uv run mew run benchmarks/bench_pallas_elementwise_tensorops_coverage.py``.
The single case applies ReLU, a scale, a row-broadcast bias and a scalar
bias over a 65x97 array with edge tiles, checked against NumPy before timing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import metal_runtime as mr
import mew
import numpy as np
from jax.experimental import pallas as pl

import palladium
from palladium.dispatch import bind
from palladium.emit.tensorops import ProgramScope, compile_kernel

M, N = 65, 97
TM, TN = 16, 32


def _elementwise(input_ref, row_bias_ref, scalar_bias_ref, output_ref):
    output_ref[...] = (
        jnp.maximum(input_ref[...], 0.0) * 2.0 + row_bias_ref[...] + scalar_bias_ref[()]
    )


@mew.parametrize(
    ({},),
    ids=("row-scalar-broadcast-edge-tiles",),
    tags="pallas-tensorops-elementwise-coverage",
    use_real_time=True,
    unit="ms",
)
def bench_pallas_elementwise_tensorops_coverage(state: mew.State) -> None:
    try:
        mr.device_info()
    except mr.DeviceError as error:
        state.skip_with_error(f"No Metal device: {error}")
        return

    rng = np.random.default_rng(71)
    values = rng.standard_normal((M, N), dtype=np.float32)
    row_bias = rng.standard_normal((N,), dtype=np.float32)
    scalar_bias = np.asarray(0.375, dtype=np.float32)
    values.setflags(write=False)
    row_bias.setflags(write=False)
    scalar_bias.setflags(write=False)
    staged = pl.pallas_call(
        _elementwise,
        grid=((M + TM - 1) // TM, (N + TN - 1) // TN),
        in_specs=[
            pl.BlockSpec((TM, TN), lambda i, j: (i, j)),
            pl.BlockSpec((TN,), lambda i, j: (j,)),
            pl.BlockSpec((), lambda i, j: ()),
        ],
        out_specs=pl.BlockSpec((TM, TN), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    )
    spec = palladium.trace(staged, values, row_bias, scalar_bias)
    compilation = compile_kernel(spec, scope=ProgramScope.THREADGROUP)
    kernel = bind(spec, compilation.source, threadgroup=(128, 1, 1))
    actual = kernel(values, row_bias, scalar_bias)
    expected = np.maximum(values, 0.0) * 2.0 + row_bias[None, :] + scalar_bias
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    state.set_counter("max_abs_error", float(np.max(np.abs(actual - expected))))
    state.set_counter("threadgroup_bytes", compilation.threadgroup_bytes)
    pinned = kernel.pinned(values, row_bias, scalar_bias)
    for _ in state:
        pinned()
