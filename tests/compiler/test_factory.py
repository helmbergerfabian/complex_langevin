from pytest import MonkeyPatch

from compiler.factory import get_backend
from compiler.python_backend import PythonBackend


def test_python_backend_selection(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("MY_NUMBA_TARGET", "python")
    backend = get_backend()
    assert isinstance(backend, PythonBackend)