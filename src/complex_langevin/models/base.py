# complex_langevin/models/base.py
from abc import ABC, abstractmethod
from complex_langevin.utils.cl_types import DriftKernel

class Model(ABC):
    @abstractmethod
    def generate_drift_kernel(self) -> DriftKernel:
        pass

    @abstractmethod
    def generate_action_kernel(self) -> callable:
        pass