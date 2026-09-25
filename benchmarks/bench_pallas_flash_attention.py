"""Check and benchmark Palladium's Pallas TensorOps attention lowering.

Run on Apple Silicon with MSL 4 support using
``JAX_PLATFORMS=mps,cpu uv run mew run --random-interleaving benchmarks/``.
Each case checks the generated kernel against a NumPy reference before timing
resident Metal buffers. Causal and noncausal forms use four 64-wide heads.
"""

from __future__ import annotations

import functools
import time

import jax
import jax.numpy as jnp
import metal_runtime as mr
import mew
import numpy as np

from palladium.emit.tensorops import SIMDGROUPS
from palladium.workloads.pallas_flash_attention import (
    HEAD_DIM,
    TILE_K,
    TILE_Q,
    attention_kernel,
    attention_specs,
    reference_attention,
)

BATCH = 1
HEADS = 4
SEQUENCE_LENGTHS = (1024, 2048, 4096)
CASES = [
    {"sequence_length": length, "causal": causal}
    for length in SEQUENCE_LENGTHS
    for causal in (False, True)
]
IDS = [
    f"pallas-{'causal' if case['causal'] else 'noncausal'}-s{case['sequence_length']}"
    for case in CASES
]


def _inputs(sequence_length: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(23)
    shape = (BATCH, sequence_length, HEADS, HEAD_DIM)
    return tuple(rng.standard_normal(shape, dtype=np.float32) for _ in range(3))


@functools.cache
def _expected(sequence_length: int, causal: bool) -> np.ndarray:
    return reference_attention(*_inputs(sequence_length), causal)


@mew.parametrize(CASES, ids=IDS, tags="pallas-flash-attention", use_real_time=True, unit="ms")
def bench_pallas_flash_attention(
    state: mew.State,
    sequence_length: int,
    causal: bool,
) -> None:
    try:
        mr.device_info()
    except mr.DeviceError as error:
        state.skip_with_error(f"No Metal device: {error}")
        return

    import palladium

    q, k, v = _inputs(sequence_length)
    grid, in_specs, out_specs = attention_specs(BATCH, sequence_length, HEADS, TILE_Q)
    shape = jax.ShapeDtypeStruct(q.shape, jnp.float32)
    start = time.perf_counter()
    source = palladium.debug_msl(
        attention_kernel(tile_q=TILE_Q, tile_k=TILE_K, causal=causal),
        shape,
        shape,
        shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=shape,
        dot_general="tensorops",
    )
    kernel = mr.Kernel(source, "palladium_kernel")
    compile_ms = (time.perf_counter() - start) * 1000
    output = mr.Buffer.empty(list(q.shape), dtype="float32")
    buffers = [mr.Buffer(q), mr.Buffer(k), mr.Buffer(v), output]
    threadgroup = SIMDGROUPS * kernel.thread_execution_width
    metal_grid = (BATCH * threadgroup, HEADS, sequence_length // TILE_Q)

    def execute() -> None:
        mr.run(kernel, grid=metal_grid, threadgroup=threadgroup, buffers=buffers)

    execute()
    np.testing.assert_allclose(
        output.to_numpy(), _expected(sequence_length, causal), rtol=3e-4, atol=3e-4
    )

    state.set_counter("compile_ms", compile_ms)
    for _ in range(3):
        execute()
    for _ in state:
        execute()
