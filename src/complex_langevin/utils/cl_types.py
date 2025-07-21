# complex_langevin/utils/cl_types.py
from typing import Protocol, runtime_checkable, Any
from complex_langevin.config import CL_REAL, CL_COMPLEX

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
    def __call__(self, idx: int, phi_arr: Any, drift_arr: Any, noise_arr: Any, dt_ada_arr: Any, dt_base: CL_REAL, langevin_time: Any) -> None: 
        ...

@runtime_checkable
class dtadaKernel(Protocol):
    def __call__(self, idx: int, dt_ada_arr: Any, drift_arr: Any) -> None: 
        ...


from typing import Union
import complex_langevin.compiler as comp
BackendType = Union[comp.PythonBackend, comp.NumbaBackend, comp.CudaBackend]