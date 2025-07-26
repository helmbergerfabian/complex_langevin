### complex_langevin/compiler/python_backend.py
from .base import SimulationBackend

class PythonBackend(SimulationBackend):
    def __init__(self):
        super().__init__()
        self._kernel = (lambda f: f)
        self.use_cuda = False
        
    @property
    def kernel(self):
        return self._kernel
    
    def parallel_loop(self, kernel_function, iter_max, *args):
        '''Executes a kernel function in a loop for iter_max iterations.
        This method does not use any parallelization and runs sequentially.
        Args:
            kernel_function: The function to execute in the loop.
            iter_max: The number of iterations to run.
            *args: Additional arguments to pass to the kernel function.
        '''
        for xi in range(iter_max):
            kernel_function(xi, *args)

    def act_parallel_loop(self, kernel_function, act_matrix, iter_max, *args):
        '''Executes a kernel function in a loop for iter_max iterations,
        but only for indices where act_matrix is True.
        Args:
            kernel_function: The function to execute in the loop.
            act_matrix: A boolean array indicating which indices to process.
            iter_max: The maximum number of iterations to run.
            *args: Additional arguments to pass to the kernel function.
        '''
        if len(act_matrix) < iter_max:
            raise ValueError("act_matrix length must be at least iter_max")
        
        for xi in range(iter_max):
            if act_matrix[xi]:
                kernel_function(xi, *args)