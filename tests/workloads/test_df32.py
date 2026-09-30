"""The df32 MSL library from examples/df32.py against float64 on the device."""

import metal_runtime as mr
import numpy as np
import pytest
from df32 import PRELUDE, kernel

BINARY = """
kernel void k(device const df32* a [[buffer(0)]], device const df32* b [[buffer(1)]],
              device df32* out [[buffer(2)]], uint tid [[thread_position_in_grid]]) {{
    out[tid] = {name}(a[tid], b[tid]);
}}
"""
UNARY = """
kernel void k(device const df32* a [[buffer(0)]], device df32* out [[buffer(1)]],
              uint tid [[thread_position_in_grid]]) {{
    out[tid] = {name}(a[tid]);
}}
"""


def _pairs(x: np.ndarray) -> np.ndarray:
    hi = x.astype(np.float32)
    return np.stack([hi, (x - hi.astype(np.float64)).astype(np.float32)], axis=-1)


def _run(template: str, name: str, *operands: np.ndarray) -> np.ndarray:
    pairs = [_pairs(x) for x in operands]
    out = mr.Buffer.zeros(pairs[0].shape, "float32")
    mr.run(
        kernel(template.format(name=name), "k"),
        grid=len(operands[0]),
        buffers=[*map(mr.Buffer, pairs), out],
    )
    return out.to_numpy().astype(np.float64)


def _operands(rng, n=10_000):
    # Exponents well inside float32's normal range, so neither the operands
    # nor the compensation terms underflow.
    mant = rng.uniform(1.0, 2.0, n) * rng.choice([-1.0, 1.0], n)
    return np.ldexp(mant, rng.integers(-40, 41, n))


@pytest.mark.parametrize(
    "name, reference",
    [("df_add", np.add), ("df_sub", np.subtract), ("df_mul", np.multiply), ("df_div", np.divide)],
)
def test_binary_ops_match_float64(rng, name, reference):
    a, b = _operands(rng), _operands(rng)
    out = _run(BINARY, name, a, b)
    want = reference(a, b)
    assert np.max(np.abs(out[:, 0] + out[:, 1] - want) / np.abs(want)) <= 2**-40
    # The result stays a normalized pair: |lo| <= ulp(hi) / 2.
    hi, lo = out[:, 0].astype(np.float32), out[:, 1].astype(np.float32)
    assert np.all(np.abs(lo) <= 0.5 * np.abs(np.spacing(hi)))


def test_sqrt_matches_float64_and_handles_zero_and_negatives(rng):
    a = np.abs(_operands(rng))
    out = _run(UNARY, "df_sqrt", a)
    assert np.max(np.abs(out[:, 0] + out[:, 1] - np.sqrt(a)) / np.sqrt(a)) <= 2**-40
    edge = _run(UNARY, "df_sqrt", np.array([0.0, -0.0, -4.0]))
    assert edge[0, 0] == 0.0 and not np.signbit(edge[0, 0])
    assert edge[1, 0] == 0.0 and np.signbit(edge[1, 0])
    assert np.isnan(edge[2, 0])


def test_prelude_requires_safe_math():
    source = PRELUDE + "kernel void noop() {}"
    for mode in (mr.MathMode.FAST, mr.MathMode.RELAXED):
        with pytest.raises(mr.CompileError, match="SAFE"):
            mr.Kernel(source, "noop", math_mode=mode)
    assert kernel("kernel void noop() {}", "noop").math_mode == mr.MathMode.SAFE
