import numpy as np
import pytest

# Modules that only manipulate text/jaxprs and never touch the GPU; they
# keep running on machines (and CI runners) without a Metal device.
_NO_GPU_MODULES = {"test_01_trace", "test_msl_snapshots"}


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
