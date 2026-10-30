"""Measure the Apple GPU roofline and where Pallas matmuls land on it.

Run with ``uv run mew run --random-interleaving benchmarks/bench_roofline.py``;
add ``JAX_PLATFORMS=mps,cpu`` with jax-mps installed for the vendor rows.

- ``bench_copy_bandwidth``: device memory bandwidth, one read and one write
  of 256 MiB.
- ``bench_simdgroup_matrix_peak``: the 8x8 matrix instruction on register
  operands. Every product reads another accumulator, so the compiler cannot
  hoist it out of the loop.
- ``bench_pallas_matmul``: a plain tiled ``jnp.dot`` Pallas kernel through the
  TensorOps lowering (``matmul2d``), and for float32 also through
  ``dot_general="simdgroup"``, checked against NumPy before timing.
- ``bench_vendor_matmul``: ``jnp.dot`` on the mps platform, which jax-mps
  runs on MLX's kernels. Wall time per call, so it includes dispatch.

Kernel rows report GPU time from the command buffer. Each row sets a
``tflops`` (or ``gbps``) counter from its median.
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

from palladium.emit.tensorops import SIMDGROUPS

DTYPES = ("float32", "float16")
SIZES = (2048, 4096)
TILES = ((32, 32), (64, 32), (64, 64), (128, 64))
MATMUL_CASES = [
    {"size": size, "dtype": dtype, "tile_m": tm, "tile_n": tn, "policy": policy}
    for size in SIZES
    for dtype in DTYPES
    for tm, tn in TILES
    # The simdgroup kernel covers float32 only; float16 keeps matmul2d.
    for policy in (("tensorops", "simdgroup") if dtype == "float32" else ("tensorops",))
]
MATMUL_IDS = [
    f"{c['policy']}-{c['dtype']}-n{c['size']}-tm{c['tile_m']}-tn{c['tile_n']}" for c in MATMUL_CASES
]
VENDOR_CASES = [{"size": size, "dtype": dtype} for size in SIZES for dtype in DTYPES]
VENDOR_IDS = [f"{c['dtype']}-n{c['size']}" for c in VENDOR_CASES]

_ITERATIONS = 2048
_SIMD_GROUPS_PER_THREADGROUP = 4
_THREADGROUPS = 4096


def _gpu_time(kernel: mr.Kernel, grid, threadgroup, buffers) -> float:
    batch = mr.Batch()
    batch.add(kernel, grid=grid, threadgroup=threadgroup, buffers=buffers)
    batch.wait()
    if batch.gpu_time is None:
        raise RuntimeError("gpu_time unavailable after wait()")
    return batch.gpu_time


def _run_timed(state: mew.State, run, work: float, counter: str, scale: float) -> None:
    run()
    times = []
    for _ in state:
        seconds = run()
        times.append(seconds)
        state.set_iteration_time(seconds)
    state.set_counter(counter, work / statistics.median(times) / scale)


@mew.benchmark(tags="roofline", use_manual_time=True, unit="ms")
def bench_copy_bandwidth(state: mew.State) -> None:
    kernel = mr.Kernel(
        """
#include <metal_stdlib>
using namespace metal;
kernel void copy4(device const float4* a [[buffer(0)]], device float4* b [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) { b[i] = a[i]; }
""",
        "copy4",
    )
    nbytes = 256 << 20
    count = nbytes // 16
    buffers = [mr.Buffer(np.ones(count * 4, np.float32)), mr.Buffer.zeros([count * 4])]
    _run_timed(
        state,
        lambda: _gpu_time(kernel, count, 256, buffers),
        work=2 * nbytes,
        counter="gbps",
        scale=1e9,
    )


@mew.parametrize(
    [{"dtype": dtype} for dtype in DTYPES],
    ids=list(DTYPES),
    tags="roofline",
    use_manual_time=True,
    unit="ms",
)
def bench_simdgroup_matrix_peak(state: mew.State, dtype: str) -> None:
    ctype = {"float32": "float", "float16": "half"}[dtype]
    source = f"""
#include <metal_stdlib>
using namespace metal;
kernel void peak(device const {ctype}* x [[buffer(0)]], device {ctype}* out [[buffer(1)]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint tg [[threadgroup_position_in_grid]]) {{
    simdgroup_matrix<{ctype}, 8, 8> a, b, c0(0.001), c1(0.002), c2(0.003), c3(0.004);
    simdgroup_load(a, x, 8);
    simdgroup_load(b, x + 64, 8);
    for (int i = 0; i < {_ITERATIONS}; ++i) {{
        simdgroup_multiply_accumulate(c0, c1, a, c0);
        simdgroup_multiply_accumulate(c1, c2, b, c1);
        simdgroup_multiply_accumulate(c2, c3, a, c2);
        simdgroup_multiply_accumulate(c3, c0, b, c3);
    }}
    device {ctype}* o = out + (tg * {_SIMD_GROUPS_PER_THREADGROUP} + sg) * 256;
    simdgroup_store(c0, o, 8);
    simdgroup_store(c1, o + 64, 8);
    simdgroup_store(c2, o + 128, 8);
    simdgroup_store(c3, o + 192, 8);
}}"""
    kernel = mr.Kernel(source, "peak")
    threadgroup = _SIMD_GROUPS_PER_THREADGROUP * kernel.thread_execution_width
    simd_groups = _THREADGROUPS * _SIMD_GROUPS_PER_THREADGROUP
    buffers = [
        mr.Buffer(np.full(128, 0.01, dtype)),
        mr.Buffer.zeros([simd_groups * 256], dtype=dtype),
    ]
    _run_timed(
        state,
        lambda: _gpu_time(kernel, _THREADGROUPS * threadgroup, threadgroup, buffers),
        work=simd_groups * _ITERATIONS * 4 * 2 * 8**3,
        counter="tflops",
        scale=1e12,
    )


def _matmul_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=o_ref.dtype)


@mew.parametrize(MATMUL_CASES, ids=MATMUL_IDS, tags="roofline", use_manual_time=True, unit="ms")
def bench_pallas_matmul(
    state: mew.State, size: int, dtype: str, tile_m: int, tile_n: int, policy: str
) -> None:
    import palladium

    rng = np.random.default_rng(0)
    x = rng.standard_normal((size, size)).astype(dtype)
    y = rng.standard_normal((size, size)).astype(dtype)
    shape = jax.ShapeDtypeStruct((size, size), dtype)
    source = palladium.debug_msl(
        _matmul_kernel,
        shape,
        shape,
        grid=(size // tile_m, size // tile_n),
        in_specs=(
            pl.BlockSpec((tile_m, size), lambda i, j: (i, 0)),
            pl.BlockSpec((size, tile_n), lambda i, j: (0, j)),
        ),
        out_specs=pl.BlockSpec((tile_m, tile_n), lambda i, j: (i, j)),
        out_shape=shape,
        compiler_params=palladium.CompilerParams(dot_general=policy),
    )
    kernel = mr.Kernel(source, "palladium_kernel")
    output = mr.Buffer.empty([size, size], dtype=dtype)
    buffers = [mr.Buffer(x), mr.Buffer(y), output]
    threadgroup = SIMDGROUPS * kernel.thread_execution_width
    grid = ((size // tile_m) * threadgroup, size // tile_n, 1)
    _gpu_time(kernel, grid, threadgroup, buffers)
    corner = x[:64].astype(np.float32) @ y[:, :64].astype(np.float32)
    tolerance = 1e-5 if dtype == "float32" else 1e-3
    np.testing.assert_allclose(
        output.to_numpy()[:64, :64].astype(np.float32) / np.abs(corner).max(),
        corner / np.abs(corner).max(),
        atol=tolerance,
    )
    _run_timed(
        state,
        lambda: _gpu_time(kernel, grid, threadgroup, buffers),
        work=2 * size**3,
        counter="tflops",
        scale=1e12,
    )


@mew.parametrize(VENDOR_CASES, ids=VENDOR_IDS, tags="roofline", use_manual_time=True, unit="ms")
def bench_vendor_matmul(state: mew.State, size: int, dtype: str) -> None:
    try:
        device = jax.devices("mps")[0]
    except RuntimeError:
        state.skip_with_error("no mps platform: install jax-mps and set JAX_PLATFORMS=mps,cpu")
        return
    with jax.default_device(device):
        x = jax.random.normal(jax.random.key(0), (size, size), dtype)
        y = jax.random.normal(jax.random.key(1), (size, size), dtype)
        matmul = jax.jit(jnp.dot)

        def run() -> float:
            start = time.perf_counter()
            jax.block_until_ready(matmul(x, y))
            return time.perf_counter() - start

        _run_timed(state, run, work=2 * size**3, counter="tflops", scale=1e12)
