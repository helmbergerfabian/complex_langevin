# complex_langevin/compiler/zero_d_scalar_phi4.py
from complex_langevin.models.base import Model
from complex_langevin.utils.cl_types import DriftKernel
from complex_langevin.compiler.factory import get_backend
from complex_langevin.config import CL_REAL, CL_COMPLEX

backend = get_backend()

class ZeroDScalarPhi4(Model):
    def __init__(self, sigma: CL_COMPLEX = None, lamb: CL_REAL = None) -> None:
        if sigma is None:
            sigma = CL_COMPLEX(1+1j)
        if lamb is None:
            lamb = CL_REAL(1)
            
        self.sigma = sigma
        self.lamb = lamb
        self.drift_kernel = self.generate_drift_kernel()

    def generate_drift_kernel(self) -> DriftKernel:
        _sigma = self.sigma
        _lamb = self.lamb

        @backend.kernel
        def _drift_kernel(idx, idx_list, drift_arr, phi_arr) -> None:
            _idx = idx_list[idx]
            phi_idx = phi_arr[_idx]
            drift_arr[_idx] = -(_sigma * phi_idx + _lamb * phi_idx**3)

        return _drift_kernel


    def generate_action_kernel(self) -> callable:
        raise NotImplementedError("generate_action not yet implemented")
        _sigma = self.sigma
        _lamb = self.lamb
        
        @backend.kernel
        def _drift(idx, action_arr, phi_arr) -> None:
            phi_idx = phi_arr[idx]
            val = 0.5*_sigma * phi_idx**2 + 0.25*_lamb * phi_idx**4
            action_arr[idx] = val

        return _drift
