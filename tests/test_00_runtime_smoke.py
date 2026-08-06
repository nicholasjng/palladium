"""The runtime side works without palladium.

If these fail, the problem is the metal-runtime install (or the
machine), not the emitter; fix that first.
"""

import metal_runtime as mr
import numpy as np

SAXPY = """
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* o [[buffer(2)]],
    constant float& a [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    o[tid] = a * x[tid] + y[tid];
}
"""


def test_handwritten_saxpy(rng):
    n = 4096
    x = rng.standard_normal(n, dtype=np.float32)
    y = rng.standard_normal(n, dtype=np.float32)
    xb, yb = mr.Buffer(x), mr.Buffer(y)
    ob = mr.Buffer.zeros([n])
    kernel = mr.Kernel(SAXPY, "saxpy")
    mr.run(kernel, grid=n, buffers=[xb, yb, ob], scalars=[np.float32(2.5)])
    # atol: Metal contracts a*x + y into fma (single rounding); the NumPy
    # reference rounds twice, so near-zero elements differ in the last ulp.
    np.testing.assert_allclose(ob.to_numpy(), 2.5 * x + y, rtol=1e-6, atol=1e-7)


def test_batch_reports_gpu_time(rng):
    n = 1 << 16
    x = rng.standard_normal(n, dtype=np.float32)
    xb, yb, ob = mr.Buffer(x), mr.Buffer(x), mr.Buffer.zeros([n])
    kernel = mr.Kernel(SAXPY, "saxpy")
    with mr.Batch() as batch:
        for _ in range(8):
            batch.add(kernel, grid=n, buffers=[xb, yb, ob], scalars=[np.float32(1.0)])
    assert batch.gpu_time is None or batch.gpu_time > 0.0
