from pytest import MonkeyPatch


def test_python_backend_selection(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("MY_NUMBA_TARGET", "python")

    from complex_langevin.compiler.factory import get_backend
    from complex_langevin.compiler.python_backend import PythonBackend

    backend = get_backend()
    assert isinstance(backend, PythonBackend)

def test_python_kernel_decotrator(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("MY_NUMBA_TARGET", "python")

    from complex_langevin.compiler.factory import get_backend

    backend = get_backend()
    assert callable(backend.kernel)



def test_numba_backend_selection(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("MY_NUMBA_TARGET", "numba")

    from complex_langevin.compiler.factory import get_backend
    from complex_langevin.compiler.numba_backend import NumbaBackend

    backend = get_backend()
    assert isinstance(backend, NumbaBackend)

def test_numba_kernel_decotrator(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("MY_NUMBA_TARGET", "numba")

    from complex_langevin.compiler.factory import get_backend

    backend = get_backend()
    assert callable(backend.kernel)


def test_cuda_backend_selection(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("MY_NUMBA_TARGET", "cuda")

    from complex_langevin.compiler.factory import get_backend
    from complex_langevin.compiler.cuda_backend import CudaBackend

    backend = get_backend()
    assert isinstance(backend, CudaBackend)

def test_cuda_kernel_decotrator(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("MY_NUMBA_TARGET", "numba")

    from complex_langevin.compiler.factory import get_backend

    backend = get_backend()
    assert callable(backend.kernel)
