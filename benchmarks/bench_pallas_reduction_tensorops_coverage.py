"""Device correctness checks for standalone tensorops cooperative row reductions.

Run on Apple Silicon with an MSL 4 capable GPU using
``JAX_PLATFORMS=mps,cpu uv run mew run benchmarks/bench_pallas_reduction_tensorops_coverage.py``.
Cases: sum and max over partial row tiles (n = 129) and over wide rows
(n = 1025). Each case checks against JAX before timing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import metal_runtime as mr
import mew
import numpy as np
from jax.experimental import pallas as pl

from palladium.emit.tensorops import SIMDGROUPS, ProgramScope, compile_kernel
from palladium.trace import trace

CASES = (
    {"name": "sum", "operation": jnp.sum, "n": 129},
    {"name": "max", "operation": jnp.max, "n": 129},
    {"name": "wide_sum", "operation": jnp.sum, "n": 1025},
    {"name": "wide_max", "operation": jnp.max, "n": 1025},
)
M, TM = 65, 16


def _reduce_rows(input_ref, output_ref, operation):
    output_ref[...] = operation(input_ref[...], axis=1)


@mew.parametrize(
    CASES,
    ids=("sum-partial-row-tile", "max-partial-row-tile", "sum-wide-row", "max-wide-row"),
    tags="pallas-tensorops-row-reduction-coverage",
    use_real_time=True,
    unit="ms",
)
def bench_pallas_reduction_tensorops_coverage(
    state: mew.State, name: str, operation, n: int
) -> None:
    try:
        mr.device_info()
    except mr.DeviceError as error:
        state.skip_with_error(f"No Metal device: {error}")
        return

    rng = np.random.default_rng(59)
    values = rng.standard_normal((M, n), dtype=np.float32)
    values.setflags(write=False)
    call = pl.pallas_call(
        lambda x_ref, out_ref: _reduce_rows(x_ref, out_ref, operation),
        grid=((M + TM - 1) // TM,),
        in_specs=[pl.BlockSpec((TM, n), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((TM,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((M,), jnp.float32),
    )
    spec = trace(call, jax.ShapeDtypeStruct((M, n), jnp.float32))
    kernel_name = f"palladium_tensorops_row_{name}"
    compilation = compile_kernel(spec, kernel_name, scope=ProgramScope.THREADGROUP)
    kernel = mr.Kernel(compilation.source, kernel_name)
    input_buffer = mr.Buffer(values)
    output_buffer = mr.Buffer.empty([M], dtype="float32")
    threadgroup = SIMDGROUPS * kernel.thread_execution_width
    grid = (threadgroup * ((M + TM - 1) // TM), 1, 1)
    buffers = [input_buffer, output_buffer]

    mr.run(kernel, grid=grid, threadgroup=threadgroup, buffers=buffers)
    expected = np.asarray(jax.device_get(operation(jnp.asarray(values), axis=1)))
    actual = output_buffer.to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    state.set_counter("max_abs_error", float(np.max(np.abs(actual - expected))))
    state.set_counter("threadgroup_bytes", compilation.threadgroup_bytes)
    for _ in state:
        mr.run(kernel, grid=grid, threadgroup=threadgroup, buffers=buffers)
