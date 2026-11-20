"""Single lowerings on the default JAX device, compared across platforms.

Rows carry no platform in their names: run this file once per platform and
let ``mew compare`` line them up (``docs/benchmarking.md``). Every row checks
its result against the CPU backend before timing, and every call waits for
its result. On ``metal`` rows carry the plugin's counters per call
(``plugin_stats.py``).

- ``product-*``: float32 products that do not take the aligned square path of
  ``bench_kernels.py``: few output tiles with a long contraction
  (``few_tiles``: 64x8192 by 8192x64), sizes that are no multiple of a tile
  (``tails``), and a transposed left operand (``transposed``).
- ``reduce-*``: sums of a 1,024 by 4,096 array over rows, columns and all.
- ``loop-gemm_carry``: a 16-trip ``fori_loop`` whose carry is a 512x512
  product.
- ``convolution-*``: a 3x3 convolution of 8 images of 64x64x32 into 64
  features, and the kernel gradient of its stride-2 form.
- ``convert-uint8_image``: 8 uint8 images of 256x256x3 scaled to float32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mew
import numpy as np
import plugin_stats
from jax import lax


def _conv(x, w, stride: int = 1):
    return lax.conv_general_dilated(
        x, w, (stride, stride), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )


def _case(name: str):
    """The function and its host arguments."""
    rng = np.random.default_rng(0)

    def uniform(*shape):
        return rng.uniform(-1, 1, shape).astype(np.float32)

    match name:
        case "product-few_tiles":
            return jnp.matmul, [uniform(64, 8192), uniform(8192, 64)]
        case "product-tails":
            return jnp.matmul, [uniform(1000, 770), uniform(770, 1030)]
        case "product-transposed":
            return (lambda a, b: a.T @ b), [uniform(1024, 1024), uniform(1024, 1024)]
        case "reduce-rows":
            return (lambda x: x.sum(axis=1)), [uniform(1024, 4096)]
        case "reduce-columns":
            return (lambda x: x.sum(axis=0)), [uniform(1024, 4096)]
        case "reduce-all":
            return jnp.sum, [uniform(1024, 4096)]
        case "loop-gemm_carry":
            # Scaled so that sixteen products neither vanish nor overflow.
            w = uniform(512, 512) / np.float32(16)
            return (lambda h, w: lax.fori_loop(0, 16, lambda _, h: h @ w, h)), [
                uniform(512, 512),
                w,
            ]
        case "convolution-forward":
            return _conv, [uniform(8, 64, 64, 32), uniform(3, 3, 32, 64)]
        case "convolution-kernel_gradient":
            loss = lambda w, x: jnp.sum(_conv(x, w, stride=2) ** 2)
            return jax.grad(loss), [uniform(3, 3, 32, 64) / 16, uniform(8, 64, 64, 32)]
        case "convert-uint8_image":
            image = rng.integers(0, 256, (8, 256, 256, 3)).astype(np.uint8)
            return (lambda image: image.astype(jnp.float32) / 255), [image]
    raise ValueError(name)


CASES = [
    "product-few_tiles",
    "product-tails",
    "product-transposed",
    "reduce-rows",
    "reduce-columns",
    "reduce-all",
    "loop-gemm_carry",
    "convolution-forward",
    "convolution-kernel_gradient",
    "convert-uint8_image",
]


@mew.parametrize(
    [{"case": case} for case in CASES], ids=CASES, use_real_time=True, unit="us", tags="ops"
)
def bench_op(state: mew.State, case: str) -> None:
    fn, host = _case(case)
    run = jax.jit(fn)
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        expected = np.asarray(run(*jax.device_put(host, cpu)), np.float64)
    args = jax.device_put(host, jax.devices()[0])
    actual = np.asarray(run(*args), np.float64)
    error = np.abs(actual - expected).max() / np.abs(expected).max()
    if not error <= 1e-4:
        raise AssertionError(f"{case}: normwise error {error:.2e} against the CPU backend")
    for _ in range(3):
        jax.block_until_ready(run(*args))
    plugin_stats.reset()
    for _ in state:
        jax.block_until_ready(run(*args))
    plugin_stats.report(state, state.iterations)
