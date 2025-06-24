### complex_langevin/compiler/factory.py

def get_backend():
    '''Factory function to get the appropriate backend based on the MY_NUMBA_TARGET environment variable.
    Returns:
        SimulationBackend: An instance of the backend class.'''

    import os
    target = os.environ.get('MY_NUMBA_TARGET', 'numba').lower()
    
    if target == 'python':
        from .python_backend import PythonBackend
        return PythonBackend()

    else:
        raise ValueError(f"Unknown target backend: {target}")