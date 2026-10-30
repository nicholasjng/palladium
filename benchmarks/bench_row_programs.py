"""Row softmax, layer norm and cumsum as row programs, against the bandwidth roofline.

Run with ``uv run mew run --random-interleaving benchmarks/bench_row_programs.py``.

The kernels are plain Pallas with ``CompilerParams(program="threadgroup")``,
checked against NumPy before timing resident buffers. Rows up to 256 columns
use one SIMD group per row; wider rows a whole threadgroup. Each row sets a
``gbps`` counter: one read and one write of the input size per GPU second,
to compare with ``bench_roofline.py``'s copy bandwidth.
"""

from __future__ import annotations

import statistics

import jax
import jax.numpy as jnp
import mew
import numpy as np
from jax.experimental import pallas as pl

from palladium import _runtime
from palladium.emit.tensorops import declared_threadgroup_width

SHAPES = ((8192, 4096), (32768, 1024), (131072, 256))
TILE = 8
CASES = [
    {"kernel": kernel, "rows": rows, "columns": columns}
    for kernel in ("softmax", "layer_norm", "cumsum")
    for rows, columns in SHAPES
]
IDS = [f"{c['kernel']}-{c['rows']}x{c['columns']}" for c in CASES]


def _softmax(x_ref, o_ref):
    x = x_ref[...]
    e = jnp.exp(x - jnp.max(x, axis=1, keepdims=True))
    o_ref[...] = e / jnp.sum(e, axis=1, keepdims=True)


def _layer_norm(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[...]
    mean = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=1, keepdims=True)
    o_ref[...] = (x - mean) * jax.lax.rsqrt(var + 1e-5) * w_ref[...] + b_ref[...]


def _cumsum(x_ref, o_ref):
    o_ref[...] = jnp.cumsum(x_ref[...], axis=1)


def _reference(kernel: str, x, w, b):
    if kernel == "cumsum":
        return np.cumsum(x, axis=1)
    if kernel == "softmax":
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    mean = x.mean(axis=1, keepdims=True)
    return (x - mean) / np.sqrt(x.var(axis=1, keepdims=True) + 1e-5) * w + b


@mew.parametrize(CASES, ids=IDS, tags="row-programs", use_manual_time=True, unit="ms")
def bench_row_program(state: mew.State, kernel: str, rows: int, columns: int) -> None:
    import palladium

    rng = np.random.default_rng(0)
    x = rng.standard_normal((rows, columns)).astype(np.float32)
    w = rng.standard_normal((columns,)).astype(np.float32)
    b = rng.standard_normal((columns,)).astype(np.float32)
    args = (x, w, b) if kernel == "layer_norm" else (x,)
    tile_spec = pl.BlockSpec((TILE, columns), lambda i: (i, 0))
    vector_spec = pl.BlockSpec((columns,), lambda i: (0,))
    source = palladium.debug_msl(
        {"softmax": _softmax, "layer_norm": _layer_norm, "cumsum": _cumsum}[kernel],
        *(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args),
        grid=(rows // TILE,),
        in_specs=[tile_spec] + [vector_spec] * (len(args) - 1),
        out_specs=tile_spec,
        out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
        compiler_params=palladium.CompilerParams(program="threadgroup"),
    )
    compiled = _runtime.Kernel(source, "palladium_kernel")
    output = np.empty((rows, columns), np.float32)
    buffers = [*args, output]
    threads = declared_threadgroup_width(source)
    grid = ((rows // TILE) * threads, 1, 1)

    def run() -> float:
        return _runtime.run(compiled, buffers, grid, threads)

    run()
    # A cumsum adds up to 4096 unit-scale terms in another order than NumPy.
    atol = 1e-2 if kernel == "cumsum" else 1e-5
    np.testing.assert_allclose(output, _reference(kernel, x, w, b), rtol=1e-4, atol=atol)
    times = []
    for _ in state:
        seconds = run()
        times.append(seconds)
        state.set_iteration_time(seconds)
    state.set_counter("gbps", 2 * x.nbytes / statistics.median(times) / 1e9)
