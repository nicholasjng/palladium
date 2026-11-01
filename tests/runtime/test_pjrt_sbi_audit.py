"""The numerical audit must expose zero/near-zero errors rather than mask them."""

import numpy as np
import pytest
from pjrt_sbi_audit import errors


def test_required_tolerance_accounts_for_relative_budget_and_zero_reference():
    result = errors([1.0002, 1e-7, -1e-8], [1, 0, -2e-8], rtol=1e-4, atol=2e-6)
    assert result["relative_failures"] == 3
    assert result["mixed_failures"] == 1
    assert result["zero_reference_mismatches"] == 1
    assert result["required_atol"] == pytest.approx(1e-4)
    assert result["max_relative_error_nonzero"] == pytest.approx(0.5)
    assert result["worst"]["index"] == [0]


def test_nonfinite_values_fail_even_when_they_match():
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="nonfinite"):
            errors([value], [value], 1e-4, 2e-6)


def test_scalar_and_empty_results():
    assert errors(1, 1, 1e-4, 2e-6)["required_atol"] == 0
    assert errors([], [], 1e-4, 2e-6)["elements"] == 0
