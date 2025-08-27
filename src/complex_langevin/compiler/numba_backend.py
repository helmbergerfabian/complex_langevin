
### complex_langevin/compiler/numba_backend.py
from .base import SimulationBackend
import numba # type: ignore
from numba import prange # type: ignore

compiled_kernels_parallel = {}
compiled_kernels_serial = {}
compiled_act_kernels = {}

class NumbaBackend(SimulationBackend):
    '''Numba backend for executing parallel loops using Numba's JIT compilation.
    This backend compiles kernel functions to machine code for performance.
    '''
    def __init__(self):
        super().__init__()
        self._kernel = numba.njit(nogil=True, fastmath=True)
        self.use_cuda = False

    @property
    def kernel(self):
        return self._kernel
    
    def parallel_loop(self, kernel_function, iter_max, *args):
        if kernel_function not in compiled_kernels_parallel:
            @numba.njit(parallel=True, nogil=True, fastmath=True)
            def numba_func(iter_max, *args):
                for xi in prange(iter_max):
                    kernel_function(xi, *args)

            compiled_kernels_parallel[kernel_function] = numba_func
        compiled_kernels_parallel[kernel_function](iter_max, *args)

    def serial_loop(self, kernel_function, iter_max, *args):
        if kernel_function not in compiled_kernels_serial:
            @numba.njit(parallel=False, nogil=True, fastmath=True)
            def numba_func(iter_max, *args):
                for xi in prange(iter_max):
                    kernel_function(xi, *args)

            compiled_kernels_serial[kernel_function] = numba_func
        compiled_kernels_serial[kernel_function](iter_max, *args)

    def act_parallel_loop(self, kernel_function, act_matrix, iter_max, *args):
        if kernel_function not in compiled_act_kernels:

            @numba.njit(parallel=True, nogil=True, fastmath=True)
            def numba_func(iter_max, act_matrix, *args):
                for xi in prange(iter_max):
                    if act_matrix[xi]:
                        kernel_function(xi, *args)

            compiled_act_kernels[kernel_function] = numba_func
        compiled_act_kernels[kernel_function](iter_max, act_matrix, *args)