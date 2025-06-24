import pytest
from complex_langevin.compiler.python_backend import PythonBackend


@pytest.fixture
def backend():
    return PythonBackend()

# def test_compile(backend: PythonBackend):
#     backend.compile()


def test_parallel_loop_basic(backend: PythonBackend):
    arr = [0] * 10

    @backend.kernel
    def kernel(i, arr):
        arr[i] = i * i

    backend.parallel_loop(kernel, 10, arr)
    assert arr == [i * i for i in range(10)]


def test_act_parallel_loop(backend: PythonBackend):
    arr = [0] * 10
    activation = [False, True, False, True, False, True, False, True, False, True]

    @backend.kernel
    def kernel(i, arr):
        arr[i] = i * 2

    backend.act_parallel_loop(kernel, activation, 10, arr)

    expected = [0 if not activation[i] else i * 2 for i in range(10)]
    assert arr == expected
