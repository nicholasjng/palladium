"""The M1 agreement rule rejects what a fixed absolute tolerance would hide."""

import ml_dtypes
import numpy as np
import pytest
from pjrt_tolerance import assert_normwise_close


def test_tiny_arrays_are_checked_relative_to_their_own_scale():
    # assert_allclose with atol=2e-6, the old rule, accepts an all-zero second moment.
    second_moment = np.array([1.5e-11, 3e-12, 0.0], np.float32)
    with pytest.raises(AssertionError, match="normwise"):
        assert_normwise_close(np.zeros(3, np.float32), second_moment, rtol=1e-4)
    assert_normwise_close(second_moment * (1 + 5e-5), second_moment, rtol=1e-4)


def test_near_zero_entries_are_judged_against_the_array_scale():
    reference = np.array([1.0, -1.2106e-4], np.float32)
    # 0.5% off componentwise, which a componentwise rtol of 1e-4 rejects.
    actual = np.array([1.0, -1.2049e-4], np.float32)
    assert_normwise_close(actual, reference, rtol=1e-4)


def test_nonfinite_values_must_match_in_place():
    reference = np.array([np.nan, np.inf, 1.0], np.float32)
    assert_normwise_close(reference.copy(), reference)
    with pytest.raises(AssertionError):
        assert_normwise_close(np.array([np.nan, -np.inf, 1.0], np.float32), reference)
    with pytest.raises(AssertionError):  # nonfinite where the reference is finite
        assert_normwise_close(np.array([np.nan, np.inf, np.nan], np.float32), reference)


def test_integers_are_exact_and_narrow_floats_use_their_tolerance():
    with pytest.raises(AssertionError):
        assert_normwise_close(np.array([1, 2], np.int32), np.array([1, 3], np.int32))
    half = np.array([100, 1], ml_dtypes.bfloat16)
    assert_normwise_close(np.array([100, 1.5], ml_dtypes.bfloat16), half)  # 0.5% of 100
    with pytest.raises(AssertionError):
        assert_normwise_close(np.array([1.0, 2.0], np.float32), np.array([1.0, 2.001], np.float32))


def test_all_zero_references_require_exact_zeros():
    zeros = np.zeros(3, np.float32)
    assert_normwise_close(zeros, zeros)
    with pytest.raises(AssertionError):
        assert_normwise_close(np.array([0, 1e-30, 0], np.float32), zeros)
    assert_normwise_close(np.empty(0, np.float32), np.empty(0, np.float32))
