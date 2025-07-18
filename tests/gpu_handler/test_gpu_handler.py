import pytest
import numpy as np
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from complex_langevin.simulation.state import SimState
from complex_langevin.utils.gpu_handler import GPU_handler
from pytest import MonkeyPatch



@pytest.fixture
def simstate() -> SimState:
    return SimState(int(1e3))


def test_tensor_names_collected_correctly(monkeypatch: MonkeyPatch,simstate):
    monkeypatch.setenv("MY_NUMBA_TARGET", "cuda")

    handler = GPU_handler(simstate)
    expected_names = [
        name for name, attr in simstate.__dict__.items() if isinstance(attr, np.ndarray)
    ]
    assert set(handler.tensor_names) == set(expected_names)


def test_exclude_mechanism(monkeypatch: MonkeyPatch,simstate):
    monkeypatch.setenv("MY_NUMBA_TARGET", "cuda")

    exclude = [tensor_name for tensor_name in simstate.__dict__ if isinstance(getattr(simstate, tensor_name), np.ndarray)]
    if not exclude:
        pytest.skip("No ndarray fields to test exclusion.")
    name_to_exclude = exclude[0]

    handler = GPU_handler(simstate, exclude=[name_to_exclude])
    handler.to_device()
    assert isinstance(getattr(simstate, name_to_exclude), np.ndarray), f"{name_to_exclude} should remain on host"

    for name in handler.tensor_names:
        if name != name_to_exclude:
            assert isinstance(getattr(simstate, name), DeviceNDArray), f"{name} should be on device"


def test_to_device_and_to_host_round_trip(monkeypatch: MonkeyPatch,simstate):
    monkeypatch.setenv("MY_NUMBA_TARGET", "cuda")
    handler = GPU_handler(simstate)

    original_data = {
        name: getattr(simstate, name).copy()
        for name in handler.tensor_names
    }

    handler.to_device()
    for name in handler.tensor_names:
        assert isinstance(getattr(simstate, name), DeviceNDArray), f"{name} not on GPU"

    handler.to_host()
    for name in handler.tensor_names:
        array = getattr(simstate, name)
        assert isinstance(array, np.ndarray), f"{name} not back on host"
        np.testing.assert_array_equal(array, original_data[name])


def test_non_array_fields_untouched(monkeypatch: MonkeyPatch, simstate):
    monkeypatch.setenv("MY_NUMBA_TARGET", "cuda")
    handler = GPU_handler(simstate)

    non_array_data = {
        k: v for k, v in simstate.__dict__.items()
        if not isinstance(v, np.ndarray)
    }

    handler.to_device()
    handler.to_host()

    for k, v in non_array_data.items():
        assert getattr(simstate, k) == v, f"Non-array field {k} should remain unchanged"
