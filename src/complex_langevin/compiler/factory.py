### complex_langevin/compiler/factory.py

import os
_backend_instance = None

def get_backend():
    '''Factory function to get the appropriate backend based on the MY_NUMBA_TARGET environment variable.
    Returns:
        SimulationBackend: An instance of the backend class.'''
    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance

    target = os.environ.get('MY_NUMBA_TARGET', 'numba').lower()
    
    if target == 'python':
        from .python_backend import PythonBackend
        _backend_instance = PythonBackend()
    
    elif target == 'numba':
        from .numba_backend import NumbaBackend
        _backend_instance = NumbaBackend()
    
    elif target == 'cuda':
        from .cuda_backend import CudaBackend
        _backend_instance = CudaBackend()
    
    else:
        raise ValueError(f"Unknown target backend: {target}")
    
    return _backend_instance