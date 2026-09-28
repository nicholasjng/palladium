"""Compare a fused matmul epilogue with a two-kernel global-memory baseline.

Run on Apple Silicon with an MSL 4 capable GPU using
``JAX_PLATFORMS=mps,cpu uv run mew run benchmarks/bench_pallas_matmul_epilogue_fusion.py``.
The fused path applies ReLU and a scalar bias inside the TensorOps kernel. The
baseline writes the matmul tile to a device buffer, then launches a cooperative
elementwise kernel to read, transform, and write it.
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp
import metal_runtime as mr
import mew
import numpy as np
from jax.experimental import pallas as pl

from palladium.emit.tensorops import SIMDGROUPS, ProgramScope, compile_kernel
from palladium.trace import trace

M, N, K = 512, 512, 64
TM, TN = 16, 32
TILE_ELEMENTS = TM * TN


def _matmul(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.matmul(a_ref[...], b_ref[...])


def _fused(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.maximum(jnp.matmul(a_ref[...], b_ref[...]), 0.0) + 1.0


def _relu(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.maximum(jnp.matmul(a_ref[...], b_ref[...]), 0.0)


def _spec(kernel):
    call = pl.pallas_call(
        kernel,
        grid=(M // TM, N // TN),
        in_specs=[
            pl.BlockSpec((TM, K), lambda i, j: (i, 0)),
            pl.BlockSpec((K, TN), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((TM, TN), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    )
    return trace(
        call,
        jax.ShapeDtypeStruct((M, K), jnp.float32),
        jax.ShapeDtypeStruct((K, N), jnp.float32),
    )


def _epilogue_source(name: str) -> str:
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void {name}(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{{
    const uint THREADS = threads_per_group.x * threads_per_group.y * threads_per_group.z;
    for (uint element = tid; element < {TILE_ELEMENTS}; element += THREADS) {{
        const uint row = element / {TN};
        const uint column = element % {TN};
        const uint offset = (group.x * {TM} + row) * {N} + group.y * {TN} + column;
        dst[offset] = fmax(src[offset], 0.0f) + 1.0f;
    }}
}}
"""


@mew.parametrize(
    ({},),
    ids=("fused-relu-bias",),
    tags="pallas-matmul-epilogue-fusion",
    use_real_time=True,
    unit="ms",
)
def bench_pallas_matmul_epilogue_fusion(state: mew.State) -> None:
    try:
        mr.device_info()
    except mr.DeviceError as error:
        state.skip_with_error(f"No Metal device: {error}")
        return

    rng = np.random.default_rng(31)
    a = rng.standard_normal((M, K), dtype=np.float32)
    b = rng.standard_normal((K, N), dtype=np.float32)
    matmul_spec = _spec(_matmul)
    fused_spec = _spec(_fused)
    relu_spec = _spec(_relu)

    start = time.perf_counter()
    matmul_compilation = compile_kernel(
        matmul_spec, "palladium_split_matmul", scope=ProgramScope.THREADGROUP
    )
    matmul_source = matmul_compilation.source
    epilogue_source = _epilogue_source("palladium_split_epilogue")
    split_codegen_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    fused_compilation = compile_kernel(
        fused_spec, "palladium_fused_matmul_epilogue", scope=ProgramScope.THREADGROUP
    )
    fused_codegen_ms = (time.perf_counter() - start) * 1000
    relu_compilation = compile_kernel(
        relu_spec, "palladium_tensorops_relu_matmul", scope=ProgramScope.THREADGROUP
    )
    start = time.perf_counter()
    matmul_kernel = mr.Kernel(matmul_source, "palladium_split_matmul")
    split_matmul_compile_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    epilogue_kernel = mr.Kernel(epilogue_source, "palladium_split_epilogue")
    split_epilogue_compile_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    fused_kernel = mr.Kernel(fused_compilation.source, "palladium_fused_matmul_epilogue")
    fused_compile_ms = (time.perf_counter() - start) * 1000
    relu_kernel = mr.Kernel(relu_compilation.source, "palladium_tensorops_relu_matmul")

    a_buffer = mr.Buffer(a)
    b_buffer = mr.Buffer(b)
    intermediate = mr.Buffer.empty([M, N], dtype="float32")
    fused_output = mr.Buffer.empty([M, N], dtype="float32")
    relu_output = mr.Buffer.empty([M, N], dtype="float32")
    split_output = mr.Buffer.empty([M, N], dtype="float32")
    threadgroup = SIMDGROUPS * fused_kernel.thread_execution_width
    # Grid dimensions are thread counts; x includes the lanes for each row tile.
    grid = (threadgroup * (M // TM), N // TN, 1)

    def execute_split() -> None:
        mr.run(
            matmul_kernel,
            grid=grid,
            threadgroup=threadgroup,
            buffers=[a_buffer, b_buffer, intermediate],
        )
        mr.run(
            epilogue_kernel,
            grid=grid,
            threadgroup=threadgroup,
            buffers=[intermediate, split_output],
        )

    def execute_fused() -> None:
        mr.run(
            fused_kernel,
            grid=grid,
            threadgroup=threadgroup,
            buffers=[a_buffer, b_buffer, fused_output],
        )

    def execute_tensorops_relu() -> None:
        mr.run(
            relu_kernel,
            grid=grid,
            threadgroup=threadgroup,
            buffers=[a_buffer, b_buffer, relu_output],
        )

    execute_split()
    expected_matmul = a @ b
    np.testing.assert_allclose(
        intermediate.to_numpy(),
        expected_matmul,
        rtol=3e-4,
        atol=3e-4,
        err_msg="split TensorOps matmul intermediate is incorrect",
    )
    expected = np.maximum(expected_matmul, 0.0) + 1.0
    expected_relu = np.maximum(expected_matmul, 0.0)
    np.testing.assert_allclose(
        split_output.to_numpy(),
        expected,
        rtol=3e-4,
        atol=3e-4,
        err_msg="split epilogue output is incorrect",
    )
    execute_tensorops_relu()
    np.testing.assert_allclose(
        relu_output.to_numpy(),
        expected_relu,
        rtol=3e-4,
        atol=3e-4,
        err_msg="tensorops fused ReLU output is incorrect",
    )
    execute_fused()
    np.testing.assert_allclose(
        fused_output.to_numpy(),
        expected,
        rtol=3e-4,
        atol=3e-4,
        err_msg="fused tensorops output is incorrect",
    )

    for _ in range(3):
        execute_split()
        execute_fused()

    split_times: list[float] = []
    fused_times: list[float] = []
    for iteration, _ in enumerate(state):
        order = (
            (execute_split, execute_fused) if iteration % 2 == 0 else (execute_fused, execute_split)
        )
        samples = []
        for execute in order:
            start = time.perf_counter()
            execute()
            samples.append((time.perf_counter() - start) * 1000)
        if iteration % 2 == 0:
            split_times.append(samples[0])
            fused_times.append(samples[1])
        else:
            fused_times.append(samples[0])
            split_times.append(samples[1])

    state.set_counter("split_codegen_ms", split_codegen_ms)
    state.set_counter("fused_codegen_ms", fused_codegen_ms)
    state.set_counter("split_matmul_metal_compile_ms", split_matmul_compile_ms)
    state.set_counter("split_epilogue_metal_compile_ms", split_epilogue_compile_ms)
    state.set_counter("fused_metal_compile_ms", fused_compile_ms)
    state.set_counter("split_median_ms", statistics.median(split_times))
    state.set_counter("fused_median_ms", statistics.median(fused_times))
    state.set_counter(
        "split_over_fused_speedup",
        statistics.median(split_times) / statistics.median(fused_times),
    )
    state.set_counter("split_intermediate_bytes", 2 * M * N * np.dtype(np.float32).itemsize)
    state.set_counter("fused_threadgroup_bytes", fused_compilation.threadgroup_bytes)
