### complex_langevin/compiler/base.py
from abc import ABC, abstractmethod

class SimulationBackend(ABC):
    """
    Abstract base class for simulation backends.
    This class defines the interface that all simulation backends must implement.
    It provides abstract methods for defining kernel functions and executing parallel loops,
    which are essential for running simulations efficiently on different hardware or frameworks.
    Attributes:
        kernel (callable): A property that should return a function decorator to be used for kernel functions.
    Methods:
        parallel_loop(kernel_function, iter_max, *args):
            Execute the given kernel function in parallel over a specified number of iterations.
        act_parallel_loop(kernel_function, act_matrix, iter_max, *args):
            Execute the given kernel function in parallel, applying it to an action matrix over a specified number of iterations.
    """
    @property
    @abstractmethod
    def kernel(self) -> callable:
        """Return a function decorator to use for kernel functions."""
        pass

    @abstractmethod
    def parallel_loop(self, kernel_function, iter_max, *args) -> None:
        pass

    @abstractmethod
    def act_parallel_loop(self, kernel_function, act_matrix, iter_max, *args) -> None:
        pass