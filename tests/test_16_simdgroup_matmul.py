"""Standalone tiled-`simdgroup_matrix` matmul (docs/simdgroup-matmul-design.md
step 1's benchmark kernel, promoted to a real, tested module). Not routed
through trace/emit/bind: `simdgroup_matmul(m, k, n)` compiles and returns
a NumPy-in/NumPy-out callable directly. Scoped like `_rule_dot_general`
(rank-2, no batch dims) plus one more constraint neither the naive rule
nor Pallas need: m, k, n must each be a multiple of 8
(`simdgroup_float8x8`'s tile size).
"""

import numpy as np
import pytest

import palladium
from palladium.emit import EmitError

pytestmark = pytest.mark.exercise


def test_square_matmul_matches_numpy(rng):
    a = rng.standard_normal((64, 64), dtype=np.float32)
    b = rng.standard_normal((64, 64), dtype=np.float32)
    f = palladium.simdgroup_matmul(64, 64, 64)
    got = f(a, b)
    np.testing.assert_allclose(got, a @ b, rtol=1e-4, atol=1e-4)


def test_rectangular_matmul_matches_numpy(rng):
    a = rng.standard_normal((32, 16), dtype=np.float32)
    b = rng.standard_normal((16, 24), dtype=np.float32)
    f = palladium.simdgroup_matmul(32, 16, 24)
    got = f(a, b)
    np.testing.assert_allclose(got, a @ b, rtol=1e-4, atol=1e-4)


def test_larger_matmul_matches_numpy(rng):
    a = rng.standard_normal((256, 128), dtype=np.float32)
    b = rng.standard_normal((128, 256), dtype=np.float32)
    f = palladium.simdgroup_matmul(256, 128, 256)
    got = f(a, b)
    np.testing.assert_allclose(got, a @ b, rtol=1e-3, atol=1e-3)


def test_non_multiple_of_8_rejected():
    with pytest.raises(EmitError, match="multiple of 8"):
        palladium.simdgroup_matmul(10, 8, 8)


def test_unsupported_dtype_rejected():
    with pytest.raises(EmitError, match="dtype"):
        palladium.simdgroup_matmul(8, 8, 8, dtype="float16")


def test_wrong_input_shape_rejected(rng):
    f = palladium.simdgroup_matmul(16, 16, 16)
    a = rng.standard_normal((8, 16), dtype=np.float32)
    b = rng.standard_normal((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="expected shapes"):
        f(a, b)


def test_compiled_kernel_is_reusable(rng):
    """Same compiled kernel, called with different data each time."""
    f = palladium.simdgroup_matmul(16, 16, 16)
    for _ in range(3):
        a = rng.standard_normal((16, 16), dtype=np.float32)
        b = rng.standard_normal((16, 16), dtype=np.float32)
        np.testing.assert_allclose(f(a, b), a @ b, rtol=1e-4, atol=1e-4)
