### complex_langevin/compiler/base.py
from abc import ABC, abstractmethod

class SimulationBackend(ABC):

    @property
    @abstractmethod
    def kernel(self):
        """Return a function decorator to use for kernel functions."""
        pass

    # @abstractmethod
    # def compile(self):
    #     pass

    @abstractmethod
    def parallel_loop(self, kernel_function, iter_max, *args):
        pass

    @abstractmethod
    def act_parallel_loop(self, kernel_function, act_matrix, iter_max, *args):
        pass