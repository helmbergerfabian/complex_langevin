### tests/compiler/test_cuda_backend.py
from complex_langevin.compiler.cuda_backend import CudaBackend
import numpy as np
import pytest

@pytest.fixture
def backend():
    return CudaBackend()


def test_parallel_loop_basic(backend: CudaBackend):
    from numba import cuda
    
    num = int(1e4)
    arr = np.zeros(num, dtype=np.int32)
    d_arr = cuda.to_device(arr)

    @backend.kernel
    def kernel(i, arr):
        arr[i] = i * i

    backend.parallel_loop(kernel, num, d_arr)
    expected = np.array([i * i for i in range(num)], dtype=np.int32)
    arr = d_arr.copy_to_host()
    np.testing.assert_array_equal(arr, expected)


def test_act_parallel_loop(backend: CudaBackend):
    from numba import cuda

    num = int(1e4)
    arr = np.zeros(num, dtype=np.int32)
    activation = np.array([i % 2 == 1 for i in range(num)])
    d_arr = cuda.to_device(arr) 
    d_activation = cuda.to_device(activation) 

    @backend.kernel
    def kernel(i, arr):
        arr[i] = i * 2

    backend.act_parallel_loop(kernel, d_activation, num, d_arr)

    expected = np.zeros(num, dtype=np.int32)
    for i in range(num):
        if activation[i]:
            expected[i] = i * 2
    arr = d_arr.copy_to_host()

    np.testing.assert_array_equal(arr, expected)

