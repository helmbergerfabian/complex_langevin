### complex_langevin/compiler/factory.py

import os
_backend_instance = None
_target = None

def print_backend_info(target):
    if target == 'numba': print("Using NUMBA backend.")
    elif target == 'cuda': print("Using CUDA backend.")
    elif target == 'python': print("Using PYTHON backend.")

def get_backend():
    '''Factory function to get the appropriate backend based on the MY_NUMBA_TARGET environment variable.
    Returns:
        SimulationBackend: An instance of the backend class.'''
    
    global _target
    global _backend_instance
    target = os.environ.get('MY_NUMBA_TARGET', 'numba').lower()

    # if target was changed or set, declare a new instance
    if target != _target:
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
        
        _target = target
        
    print_backend_info(target)
    return _backend_instance