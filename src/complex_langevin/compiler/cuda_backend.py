
### complex_langevin/compiler/cuda_backend.py
import math, numba
import numpy as np
from itertools import count
from numba import cuda
from .base import SimulationBackend

compiled_kernels = {}
compiled_act_kernels = {}


class CudaBackend(SimulationBackend):
    def __init__(self):
        super().__init__()
        self._kernel = numba.jit(nogil=True, fastmath=True)
        self._unique_counter = count()
        self.use_cuda = True
        self.threadsperblock = 256

    @property
    def kernel(self):
        return self._kernel
    
    def _compile_cuda_kernel(self, kernel_function, with_activation=False):
        
        assert hasattr(kernel_function, 'py_func'), "Kernel function must be NUMBA compiled."
        args_count = kernel_function.py_func.__code__.co_argcount - 1

        args_string = ', '.join('c' + str(i) for i in range(args_count))
        kernel_id = next(self._unique_counter)

        prefix = """def {name}_cuda_kernel(iter_max, {extra}{args}):\n"""
        body = """    xi = cuda.grid(1)\n    if xi < iter_max{cond}:\n        _kernel_function_{id}(xi, {args})\n"""
        cond = " and act_matrix[xi]" if with_activation else ""
        extra = "act_matrix, " if with_activation else ""

        try:
            name = kernel_function.py_func.__name__
        except AttributeError:
            name = "no_name"

        code = prefix.format(name=name, extra=extra, args=args_string) + \
               body.format(cond=cond, id=kernel_id, args=args_string)

        namespace = globals().copy()
        namespace['_kernel_function_{}'.format(kernel_id)] = kernel_function
        exec(code, namespace)
        return cuda.jit(namespace[f'{name}_cuda_kernel'])

    def parallel_loop(self, kernel_function, iter_max, *args, stream=None):
        """       
        Launches a CUDA kernel function in parallel over a specified number of iterations.
        This method checks if the given kernel function has already been compiled. If not, it compiles the kernel.
        It then calculates the number of blocks per grid required to cover all iterations, and launches the kernel
        with the specified arguments and CUDA stream.
        Args:
            kernel_function (callable): The CUDA kernel function to execute.
            iter_max (int): The total number of iterations to perform.
            *args: Additional arguments to pass to the kernel function.
            stream (cuda.Stream, optional): CUDA stream to use for kernel execution. Defaults to None.
        Returns:
            None
        """
 
        if kernel_function not in compiled_kernels:
            compiled_kernels[kernel_function] = self._compile_cuda_kernel(kernel_function)
        blockspergrid = math.ceil(iter_max / threadsperblock)
        compiled_kernels[kernel_function][blockspergrid, threadsperblock, stream](iter_max, *args)

    def act_parallel_loop(self, kernel_function, act_matrix, iter_max, *args, stream=None):
        if kernel_function not in compiled_act_kernels:
            compiled_act_kernels[kernel_function] = self._compile_cuda_kernel(kernel_function, with_activation=True)
        blockspergrid = math.ceil(iter_max / threadsperblock)
        compiled_act_kernels[kernel_function][blockspergrid, threadsperblock, stream](iter_max, act_matrix, *args)