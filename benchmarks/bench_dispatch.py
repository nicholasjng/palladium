"""What a call costs on the default JAX device: M1's overhead and bandwidth gates.

Run once per platform and compare (``docs/benchmarking.md``); rows carry no
platform in their names.

- ``bench_small_call``: ``tanh(2x + 1) * x`` on 1,024 floats, pipelined as in
  a training loop (one sync per batch of 64 calls): the per-call overhead.
- ``bench_elementwise``: the same expression on 256 MiB, waiting for every
  call: memory bandwidth, reported as bytes read and written per second. On
  an Apple GPU compare it with ``bench_roofline.py::bench_copy_bandwidth``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mew
import numpy as np


def _setup(size: int):
    host = np.full(size, 0.25, np.float32)
    x = jax.device_put(host, jax.devices()[0])
    run = jax.jit(lambda x: jnp.tanh(2 * x + 1) * x)
    np.testing.assert_allclose(np.asarray(run(x)), np.tanh(2 * host + 1) * host, rtol=2e-5)
    return run, x


@mew.benchmark(use_real_time=True, unit="us")
def bench_small_call(state: mew.State) -> None:
    run, x = _setup(1024)
    for count in state.batches(64):
        for _ in range(count):
            y = run(x)
        y.block_until_ready()


@mew.benchmark(use_real_time=True, unit="ms")
def bench_elementwise(state: mew.State) -> None:
    run, x = _setup(256 * (1 << 20) // 4)
    for _ in range(3):
        run(x).block_until_ready()
    for _ in state:
        run(x).block_until_ready()
    state.set_counter("bytes", 2 * x.nbytes, mew.CounterFlags.kIsIterationInvariantRate)
