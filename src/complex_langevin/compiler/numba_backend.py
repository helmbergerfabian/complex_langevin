
### complex_langevin/compiler/numba_backend.py
from .base import SimulationBackend
import numba
from numba import prange

compiled_kernels = {}
compiled_act_kernels = {}

class NumbaBackend(SimulationBackend):
    '''Numba backend for executing parallel loops using Numba's JIT compilation.
    This backend compiles kernel functions to machine code for performance.
    '''
    def __init__(self):
        super().__init__()
        self._kernel = numba.njit(nogil=True, fastmath=True)
        print("Using NUMBA backend.")

    @property
    def kernel(self):
        return self._kernel
    
    def parallel_loop(self, kernel_function, iter_max, *args):
        if kernel_function not in compiled_kernels:
            @numba.njit(nogil=True, fastmath=True)
            def numba_func(iter_max, *args):
                for xi in prange(iter_max):
                    kernel_function(xi, *args)

            compiled_kernels[kernel_function] = numba_func
        compiled_kernels[kernel_function](iter_max, *args)

    def act_parallel_loop(self, kernel_function, act_matrix, iter_max, *args):
        if kernel_function not in compiled_act_kernels:

            @numba.njit(parallel=True, nogil=True, fastmath=True)
            def numba_func(iter_max, act_matrix, *args):
                for xi in prange(iter_max):
                    if act_matrix[xi]:
                        kernel_function(xi, *args)

            compiled_act_kernels[kernel_function] = numba_func
        compiled_act_kernels[kernel_function](iter_max, act_matrix, *args)