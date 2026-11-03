"""Kernels where MLX is strongest (M0 and M3), compared across backends.

Rows have the same names on every backend, so ``mew compare`` lines them up
(``docs/benchmarking.md``). The backend is whatever the process has:

- JAX's default device: ``metal`` runs Palladium's kernels and the plugin's
  compiler; ``mps`` (from a jax-mps venv) runs MLX's kernels through jax-mps.
- MLX directly, with ``BENCH_BACKEND=mlx`` where jax is installed too
  (``uv run --with mlx``), or in a venv with ``mlx`` and no jax.

CPU is skipped: it is not what these kernels compete with. Every row checks its
result against NumPy before timing, and every call waits for its result, so
rows include each backend's dispatch. Counters: ``flops`` per second for
matmuls and attention; on ``metal``, ``gpu_us``, ``launches`` and
``pool_misses`` per call.

- ``bench_matmul``: square ``a @ b`` in float32, float16 and bfloat16.
- ``bench_attention``: 32 heads of 64 over 2,048 tokens, float32, forward and
  the gradient of all three inputs. Forward on ``metal`` is Palladium's flash
  attention; the backward is plain JAX through the plugin, as no attention
  backward kernel exists yet. MLX uses ``mx.fast.scaled_dot_product_attention``.
- ``bench_embedding``: gathering 8,192 rows of a 32,000 by 1,024 float32 table,
  and scatter-adding them back, as an embedding's gradient does.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from typing import Any

import mew
import numpy as np


def _optional(name: str) -> Any:
    """`name` if this environment has it: an MLX venv has no jax."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


_MLX = os.environ.get("BENCH_BACKEND") == "mlx"
jax, jnp = (None, None) if _MLX else (_optional("jax"), _optional("jax.numpy"))
mx = _optional("mlx.core")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))


def _backend() -> str:
    if jax is not None:
        return jax.devices()[0].platform
    if mx is not None:
        return "mlx"
    raise RuntimeError("mlx is not installed" if _MLX else "neither jax nor mlx is installed")


def _timed(state: mew.State, call, flops: float | None = None) -> None:
    """Time `call`, which must wait for its own result.

    On ``metal``, also report per call the GPU's busy time (``gpu_us``), the
    kernels launched, and the buffer pool's misses (new MTLBuffers): wall
    time well above ``gpu_us`` is host or allocation cost, not kernel time.
    """
    for _ in range(3):
        call()
    metal = _backend() == "metal"
    if metal:
        from palladium import _pjrt

        _pjrt.stats(reset=True)
    for _ in state:
        call()
    if flops:
        state.set_counter("flops", flops, mew.CounterFlags.kIsIterationInvariantRate)
    if metal and state.iterations:
        s, n = _pjrt.stats(reset=True), state.iterations
        state.set_counter("gpu_us", s["gpu_busy_seconds"] / n * 1e6)
        state.set_counter("launches", s["launches"] / n)
        state.set_counter("pool_misses", s["pool_misses"] / n)


def _close(actual, expected, rtol: float) -> None:
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    error = np.abs(actual - expected).max() / np.abs(expected).max()
    if not error <= rtol:
        raise AssertionError(f"normwise error {error:.2e} > {rtol:g}")


MATMUL = [
    {"dtype": dtype, "n": n} for dtype in ("float32", "float16", "bfloat16") for n in (1024, 4096)
]


@mew.parametrize(
    MATMUL, ids=[f"{c['dtype']}-n{c['n']}" for c in MATMUL], use_real_time=True, unit="us"
)
def bench_matmul(state: mew.State, dtype: str, n: int) -> None:
    backend = _backend()
    if backend == "cpu":
        state.skip_with_error("cpu: kernel rows compare GPU backends")
        return
    rng = np.random.default_rng(0)
    a, b = (rng.uniform(-1, 1, (n, n)).astype(np.float32) for _ in range(2))
    if backend == "mlx":
        x, y = (mx.array(v).astype(getattr(mx, dtype)) for v in (a, b))
        a, b = (np.asarray(v.astype(mx.float32)) for v in (x, y))  # the rounded inputs

        def call():
            out = mx.matmul(x, y)
            mx.eval(out)
            return out
    else:
        device = jax.devices()[0]
        x, y = (jax.device_put(jnp.asarray(v, dtype), device) for v in (a, b))
        a, b = (np.asarray(v.astype(jnp.float32)) for v in (x, y))
        run = jax.jit(jnp.matmul)

        def call():
            return jax.block_until_ready(run(x, y))

    rows = slice(0, 64)
    out = call()
    actual = np.asarray(out.astype(mx.float32) if backend == "mlx" else out.astype(jnp.float32))
    _close(actual[rows], a[rows].astype(np.float64) @ b, 2e-5 if dtype == "float32" else 1e-2)
    _timed(state, call, flops=2 * n**3)


BATCH, TOKENS, HEADS, DIM = 1, 2048, 32, 64
ATTENTION = [
    {"direction": direction, "causal": causal}
    for direction in ("forward", "backward")
    for causal in (False, True)
]


def _reference_attention(q, k, v, causal: bool):
    """NumPy attention for one head, (tokens, dim) inputs, in float64."""
    scores = q @ k.T / math.sqrt(q.shape[-1])
    if causal:
        scores = np.where(np.tril(np.ones(scores.shape, bool)), scores, -np.inf)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    return (weights / weights.sum(-1, keepdims=True)) @ v


def _jax_attention(device, direction: str, causal: bool, q, k, v):
    def plain(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, is_causal=causal)

    if direction == "backward":
        return jax.jit(jax.grad(lambda *qkv: plain(*qkv).sum(), argnums=(0, 1, 2)))
    if device.platform != "metal":
        return jax.jit(plain)  # jax-mps routes this to MLX's fused attention
    from flash_attention import attention_kernel, attention_specs
    from jax.experimental import pallas as pl

    import palladium  # only the metal rows need Palladium

    grid, in_specs, out_specs = attention_specs(BATCH, TOKENS, HEADS, 64, DIM)
    shape = jax.ShapeDtypeStruct(q.shape, jnp.float32)
    return jax.jit(
        pl.pallas_call(
            attention_kernel(tile_q=64, tile_k=16, head_dim=DIM, causal=causal),
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            out_shape=shape,
            compiler_params=palladium.CompilerParams(dot_general="simdgroup"),
        )
    )


@mew.parametrize(
    ATTENTION,
    ids=[f"{c['direction']}-{'causal' if c['causal'] else 'full'}" for c in ATTENTION],
    use_real_time=True,
    unit="ms",
)
def bench_attention(state: mew.State, direction: str, causal: bool) -> None:
    backend = _backend()
    if backend == "cpu":
        state.skip_with_error("cpu: kernel rows compare GPU backends")
        return
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((BATCH, TOKENS, HEADS, DIM), np.float32) for _ in range(3))
    if backend == "mlx":
        # MLX takes (batch, heads, tokens, dim).
        qkv = [mx.array(x.transpose(0, 2, 1, 3)) for x in (q, k, v)]
        mask = "causal" if causal else None

        def forward(q, k, v):
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=DIM**-0.5, mask=mask)

        if direction == "forward":
            fn = forward
        else:
            fn = mx.grad(lambda q, k, v: forward(q, k, v).sum(), argnums=(0, 1, 2))

        def call():
            out = fn(*qkv)
            mx.eval(out)
            return out

        out = call()
        first = np.asarray(out if direction == "forward" else out[2])[0, 0]
    else:
        device = jax.devices()[0]
        run = _jax_attention(device, direction, causal, q, k, v)
        args = jax.device_put((q, k, v), device)

        def call():
            return jax.block_until_ready(run(*args))

        out = call()
        first = np.asarray(out if direction == "forward" else out[2])[0, :, 0]
    if direction == "forward":
        _close(first, _reference_attention(q[0, :, 0], k[0, :, 0], v[0, :, 0], causal), 1e-4)
    # The forward pass does two products of tokens x tokens x dim per head,
    # the backward five; a causal mask halves the useful work.
    products = 2 if direction == "forward" else 5
    flops = 2 * products * HEADS * TOKENS**2 * DIM / (2 if causal else 1)
    _timed(state, call, flops=flops)


VOCAB, WIDTH, ROWS = 32000, 1024, 8192


@mew.parametrize(
    [{"op": "gather"}, {"op": "scatter_add"}],
    ids=["gather", "scatter_add"],
    use_real_time=True,
    unit="us",
)
def bench_embedding(state: mew.State, op: str) -> None:
    backend = _backend()
    if backend == "cpu":
        state.skip_with_error("cpu: kernel rows compare GPU backends")
        return
    rng = np.random.default_rng(0)
    table = rng.standard_normal((VOCAB, WIDTH), np.float32)
    index = rng.integers(0, VOCAB, ROWS).astype(np.int32)
    rows = rng.standard_normal((ROWS, WIDTH), np.float32)
    if op == "gather":
        expected = table[index]
    else:
        expected = np.zeros_like(table)
        np.add.at(expected, index, rows)
    if backend == "mlx":
        t, i, r = mx.array(table), mx.array(index), mx.array(rows)

        def fn():
            return t[i] if op == "gather" else mx.zeros_like(t).at[i].add(r)

        def call():
            out = fn()
            mx.eval(out)
            return out
    else:
        device = jax.devices()[0]
        t, i, r = jax.device_put((table, index, rows), device)
        if op == "gather":
            run = jax.jit(lambda t, i, r: jnp.take(t, i, axis=0))
        else:
            run = jax.jit(lambda t, i, r: jnp.zeros_like(t).at[i].add(r))

        def call():
            return jax.block_until_ready(run(t, i, r))

    _close(np.asarray(call()), expected, 1e-5)
    _timed(state, call)
