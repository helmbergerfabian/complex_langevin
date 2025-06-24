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
