### tests/compiler/test_numba_backend.py
from complex_langevin.compiler.numba_backend import NumbaBackend
import numpy as np
import pytest

@pytest.fixture
def backend():
    return NumbaBackend()


def test_parallel_loop_basic(backend: NumbaBackend):
    num = int(1e2)
    arr = np.zeros(num, dtype=np.int32)

    @backend.kernel
    def kernel(i, arr):
        arr[i] = i * i

    backend.parallel_loop(kernel, num, arr)
    expected = np.array([i * i for i in range(num)], dtype=np.int32)
    np.testing.assert_array_equal(arr, expected)


def test_act_parallel_loop(backend: NumbaBackend):
    num = 10
    arr = np.zeros(num, dtype=np.int32)
    activation = np.array([i % 2 == 1 for i in range(num)])

    @backend.kernel
    def kernel(i, arr):
        arr[i] = i * 2

    backend.act_parallel_loop(kernel, activation, num, arr)

    expected = np.zeros(num, dtype=np.int32)
    for i in range(num):
        if activation[i]:
            expected[i] = i * 2

    np.testing.assert_array_equal(arr, expected)