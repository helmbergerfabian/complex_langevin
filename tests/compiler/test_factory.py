from pytest import MonkeyPatch
from complex_langevin.compiler.factory import get_backend
from complex_langevin.compiler.python_backend import PythonBackend


def test_python_backend_selection(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("MY_NUMBA_TARGET", "python")
    backend = get_backend()
    assert isinstance(backend, PythonBackend)