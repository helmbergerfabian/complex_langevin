# complex_langevin/models/cl_types.py
from typing import Protocol, runtime_checkable, Any

@runtime_checkable
class DriftKernel(Protocol):
    def __call__(self, idx: int, drift_arr: Any, phi_arr: Any) -> None: 
        ...

@runtime_checkable
class NoiseKernel(Protocol):
    def __call__(self, idx: int, noise_arr: Any, rng: Any) -> None: 
        ...

@runtime_checkable
class EvolveKernel(Protocol):
    def __call__(self, idx: int, phi_arr: Any, drift_arr: Any, noise_arr: Any, dt_arr: Any) -> None: 
        ...