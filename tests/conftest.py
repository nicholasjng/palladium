import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def _require_metal_device():
    import metal_runtime as mr

    try:
        mr.device_name()
    except mr.DeviceError as e:  # pragma: no cover - CI without a GPU
        pytest.skip(f"no Metal device: {e}")


@pytest.fixture
def rng():
    return np.random.default_rng(seed=17)
