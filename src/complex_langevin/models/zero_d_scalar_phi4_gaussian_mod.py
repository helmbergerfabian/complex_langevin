# complex_langevin/compiler/zero_d_scalar_phi4.py
from complex_langevin.models.base import Model
from complex_langevin.utils.cl_types import DriftKernel
from complex_langevin.compiler.factory import get_backend
from complex_langevin.config import CL_REAL, CL_COMPLEX
import cmath

backend = get_backend()

class ZeroDScalarPhi4GaussianMod(Model):
    def __init__(self, sigma: CL_COMPLEX, lamb: CL_REAL, massmod: CL_COMPLEX, pull: CL_REAL) -> None:
        self.sigma = sigma
        self.lamb = lamb
        self.massmod = massmod
        self.pull = pull

    def generate_drift_kernel(self) -> DriftKernel:

        _sigma = self.sigma
        _lamb = self.lamb
        _massmod = self.massmod
        _pull = self.pull

        @backend.kernel
        def _drift(idx, idx_list, drift_arr, phi_arr):
            _idx = idx_list[idx]

            phi_idx = phi_arr[_idx]

            action = _sigma/2*phi_idx**2+_lamb/4*phi_idx**4
            mod = -_massmod*phi_idx**2/2

            action_mod = action + mod
            drift = (_sigma*phi_idx+_lamb*phi_idx**3)

            if action_mod.real < 0:
                out = drift-_pull*(drift-_massmod*phi_idx)*cmath.exp(action_mod) / (1+_pull*cmath.exp(action_mod))
            else: 
                out = drift-_pull*(drift-_massmod*phi_idx) / (cmath.exp(-action_mod)+_pull)
            drift_arr[_idx] = -out

        return _drift
    

    def generate_action_kernel(self) -> callable:
        raise NotImplementedError("generate_action not yet implemented")