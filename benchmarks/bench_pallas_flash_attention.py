"""Check and benchmark Palladium's Pallas TensorOps attention lowering.

Run on Apple Silicon with MSL 4 support using
``JAX_PLATFORMS=mps,cpu uv run mew run --random-interleaving benchmarks/``.
Each case checks the generated kernel against a NumPy reference before timing
resident Metal buffers. Cases cover causal modes and 16-, 32-, 48-, or
64-wide heads. A focused noncausal sequence-length-4096 sweep compares query
and key tile sizes for 16-, 32-, and 64-wide heads.
"""

from __future__ import annotations

import functools
import os
import sys
import time

import jax
import jax.numpy as jnp
import metal_runtime as mr
import mew
import numpy as np

from palladium.emit.tensorops import SIMDGROUPS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
from flash_attention import attention_kernel, attention_specs, reference_attention

HEAD_DIM, TILE_Q, TILE_K = 64, 32, 64

BATCH = 1
HEADS = 4
HEAD_DIMS = (16, 32, 48, HEAD_DIM)
SEQUENCE_LENGTHS = (1024, 2048, 4096)
BASE_CASES = [
    {
        "sequence_length": length,
        "causal": causal,
        "head_dim": head_dim,
        "tile_q": TILE_Q,
        "tile_k": TILE_K,
    }
    for length in SEQUENCE_LENGTHS
    for causal in (False, True)
    for head_dim in HEAD_DIMS
]
TILE_CONFIGS = ((16, 16), (16, 32), (16, 64), (16, 128), (32, 32), (32, 128), (64, 32), (64, 64))
SWEEP_CASES = [
    {
        "sequence_length": 4096,
        "causal": False,
        "head_dim": head_dim,
        "tile_q": tile_q,
        "tile_k": tile_k,
    }
    for head_dim in (16, 32, 64)
    for tile_q, tile_k in TILE_CONFIGS
    if (tile_q, tile_k) != (TILE_Q, TILE_K)
]
CASES = BASE_CASES + SWEEP_CASES
IDS = [
    f"{'causal' if case['causal'] else 'noncausal'}-d{case['head_dim']}"
    f"-s{case['sequence_length']}-tq{case['tile_q']}-tk{case['tile_k']}"
    for case in CASES
]


def _inputs(sequence_length: int, head_dim: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(23)
    shape = (BATCH, sequence_length, HEADS, head_dim)
    return tuple(rng.standard_normal(shape, dtype=np.float32) for _ in range(3))


@functools.cache
def _expected(sequence_length: int, causal: bool, head_dim: int) -> np.ndarray:
    return reference_attention(*_inputs(sequence_length, head_dim), causal=causal)


@mew.parametrize(CASES, ids=IDS, tags="pallas-flash-attention", use_real_time=True, unit="ms")
def bench_pallas_flash_attention(
    state: mew.State,
    sequence_length: int,
    causal: bool,
    head_dim: int,
    tile_q: int,
    tile_k: int,
) -> None:
    try:
        mr.device_info()
    except mr.DeviceError as error:
        state.skip_with_error(f"No Metal device: {error}")
        return

    import palladium

    q, k, v = _inputs(sequence_length, head_dim)
    grid, in_specs, out_specs = attention_specs(BATCH, sequence_length, HEADS, tile_q, head_dim)
    shape = jax.ShapeDtypeStruct(q.shape, jnp.float32)
    start = time.perf_counter()
    source = palladium.debug_msl(
        attention_kernel(tile_q=tile_q, tile_k=tile_k, head_dim=head_dim, causal=causal),
        shape,
        shape,
        shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=shape,
        compiler_params=palladium.CompilerParams(dot_general="tensorops"),
    )
    kernel = mr.Kernel(source, "palladium_kernel")
    compile_ms = (time.perf_counter() - start) * 1000
    output = mr.Buffer.empty(list(q.shape), dtype="float32")
    buffers = [mr.Buffer(q), mr.Buffer(k), mr.Buffer(v), output]
    threadgroup = SIMDGROUPS * kernel.thread_execution_width
    metal_grid = (BATCH * threadgroup, HEADS, sequence_length // tile_q)

    def execute() -> None:
        mr.run(kernel, grid=metal_grid, threadgroup=threadgroup, buffers=buffers)

    execute()
    np.testing.assert_allclose(
        output.to_numpy(), _expected(sequence_length, causal, head_dim), rtol=3e-4, atol=3e-4
    )

    state.set_counter("compile_ms", compile_ms)
    for _ in range(3):
        execute()
    for _ in state:
        execute()
