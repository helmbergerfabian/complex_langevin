### complex_langevin/compiler/factory.py

def get_backend():
    '''Factory function to get the appropriate backend based on the MY_NUMBA_TARGET environment variable.
    Returns:
        SimulationBackend: An instance of the backend class.'''

    import os
    target = os.environ.get('MY_NUMBA_TARGET', 'python').lower()
    
    if target == 'python':
        from .python_backend import PythonBackend
        return PythonBackend()
    
    if target == 'numba':
        from .numba_backend import NumbaBackend
        return NumbaBackend()
    
    if target == 'cuda':
        from .cuda_backend import CudaBackend
        return CudaBackend()
    
    else:
        raise ValueError(f"Unknown target backend: {target}")