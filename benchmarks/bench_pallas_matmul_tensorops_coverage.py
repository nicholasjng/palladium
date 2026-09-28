"""Device correctness checks for tensorops matmul K tails, edge tiles, and dtypes.

Run on Apple Silicon with an MSL 4 capable GPU using
``JAX_PLATFORMS=mps,cpu uv run mew run benchmarks/bench_pallas_matmul_tensorops_coverage.py``.
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

TM, TN = 16, 32
CASES = (
    {"m": 30, "n": 45, "k": 144, "dtype": "float32"},
    {"m": 32, "n": 64, "k": 256, "dtype": "float16"},
    {"m": 32, "n": 64, "k": 144, "dtype": "bfloat16"},
)
JAX_DTYPES = {
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}
NUMPY_DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": jnp.bfloat16,
}
TOLERANCES = {
    "float32": (3e-3, 3e-3),
    "float16": (2e-2, 2e-2),
    "bfloat16": (1e-1, 1e-1),
}


def _matmul(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.matmul(a_ref[...], b_ref[...])


@mew.parametrize(
    CASES,
    ids=("k-tail-output-edges", "float16-k-loop", "bfloat16-k-tail"),
    tags="pallas-tensorops-matmul-coverage",
    use_real_time=True,
    unit="ms",
)
def bench_pallas_matmul_tensorops_coverage(
    state: mew.State, m: int, n: int, k: int, dtype: str
) -> None:
    try:
        mr.device_info()
    except mr.DeviceError as error:
        state.skip_with_error(f"No Metal device: {error}")
        return

    rng = np.random.default_rng(47)
    input_dtype = NUMPY_DTYPES[dtype]
    a = np.array(
        rng.standard_normal((m, k), dtype=np.float32),
        dtype=input_dtype,
        order="C",
        copy=True,
    )
    b = np.array(
        rng.standard_normal((k, n), dtype=np.float32),
        dtype=input_dtype,
        order="C",
        copy=True,
    )
    a.setflags(write=False)
    b.setflags(write=False)
    tm_count, tn_count = (m + TM - 1) // TM, (n + TN - 1) // TN
    call = pl.pallas_call(
        _matmul,
        grid=(tm_count, tn_count),
        in_specs=[
            pl.BlockSpec((TM, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, TN), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((TM, TN), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), JAX_DTYPES[dtype]),
    )
    spec = trace(
        call,
        jax.ShapeDtypeStruct((m, k), JAX_DTYPES[dtype]),
        jax.ShapeDtypeStruct((k, n), JAX_DTYPES[dtype]),
    )
    compilation = compile_kernel(
        spec, "palladium_tensorops_matmul_coverage", scope=ProgramScope.THREADGROUP
    )
    kernel = mr.Kernel(compilation.source, "palladium_tensorops_matmul_coverage")
    # metal_runtime's NumPy caster does not accept ml_dtypes.bfloat16 arrays.
    # Preserve their bits in uint16 storage and relabel the buffers, matching
    # the runtime path used by Palladium dispatch. The dtype argument is
    # positional in the installed binding.
    a_native = a.view(np.uint16) if dtype == "bfloat16" else a
    b_native = b.view(np.uint16) if dtype == "bfloat16" else b
    a_buffer = mr.Buffer(a_native, dtype if dtype == "bfloat16" else None)
    b_buffer = mr.Buffer(b_native, dtype if dtype == "bfloat16" else None)
    output = mr.Buffer.empty([m, n], dtype=dtype)
    threadgroup = SIMDGROUPS * kernel.thread_execution_width
    grid = (threadgroup * tm_count, tn_count, 1)
    buffers = [a_buffer, b_buffer, output]
    mr.run(kernel, grid=grid, threadgroup=threadgroup, buffers=buffers)

    expected = np.asarray(
        jax.device_get(jnp.matmul(jnp.asarray(a), jnp.asarray(b))), dtype=np.float32
    )
    if dtype == "bfloat16":
        # NumPy has no native bfloat16 dtype; read the bits as uint16 and
        # reinterpret them using ml_dtypes, as Palladium's dispatch path does.
        import ml_dtypes

        actual = output.to_numpy(dtype="uint16").view(ml_dtypes.bfloat16).astype(np.float32)
    else:
        actual = output.to_numpy().astype(np.float32)
    rtol, atol = TOLERANCES[dtype]
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    state.set_counter("max_abs_error", float(np.max(np.abs(actual - expected))))
    state.set_counter("threadgroup_bytes", compilation.threadgroup_bytes)
    for _ in state:
        mr.run(kernel, grid=grid, threadgroup=threadgroup, buffers=buffers)
