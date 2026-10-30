import os
import sys

import numpy as np
import pytest

# The CNF and attention workloads live with the examples that run them.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))

# Modules with device-independent checks; any GPU tests within them
# handle their own skips, so pure checks still run without Metal.
_NO_GPU_MODULES = {
    "test_kernel_spec",
    "test_effects",
    "test_write_safety",
    "test_regressions",
    "test_structural_lowerings",
    "test_typed_arithmetic",
    "test_tensorops_dot",
    "test_tensorops_flash_attention",
    "test_cursor",
    "test_views",
}


def pytest_collection_modifyitems(config, items):
    from palladium.device import device_limits

    if not device_limits():  # pragma: no cover - CI without a GPU
        skip = pytest.mark.skip(reason="no Metal device")
        for item in items:
            if item.path.stem not in _NO_GPU_MODULES:
                item.add_marker(skip)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=17)


@pytest.fixture
def metal_device():
    """Skip the test when no Metal device is present."""
    from palladium.device import device_limits

    if not device_limits():
        pytest.skip("no Metal device")
