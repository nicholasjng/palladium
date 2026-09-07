import numpy as np
import pytest

# Modules with device-independent checks; any GPU tests within them
# handle their own skips, so pure checks still run without Metal.
_NO_GPU_MODULES = {
    "test_01_trace",
    "test_18_effects",
    "test_20_write_races",
    "test_msl_snapshots",
    "test_emit_regressions",
    "test_emit_features",
}


def pytest_collection_modifyitems(config, items):
    import metal_runtime as mr

    try:
        mr.device_name()
    except mr.DeviceError as e:  # pragma: no cover - CI without a GPU
        skip = pytest.mark.skip(reason=f"no Metal device: {e}")
        for item in items:
            if item.module.__name__ not in _NO_GPU_MODULES:
                item.add_marker(skip)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=17)
