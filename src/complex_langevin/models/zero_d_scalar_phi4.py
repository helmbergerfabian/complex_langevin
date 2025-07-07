
### complex_langevin/compiler/cuda_backend.py
from .base import Model
from complex_langevin.compiler.factory import get_backend
from complex_langevin.config import CL_REAL, CL_COMPLEX, CL_INT

backend = get_backend()

class ZeroDScalarPhi4(Model):
    def __init__(self, sigma: CL_COMPLEX, lamb: CL_REAL):
        self.sigma = sigma
        self.lamb = lamb

    def drift(self):
        _sigma = self.sigma
        _lamb = self.lamb

        @backend.kernel
        def _drift(idx, drift, phi):
            phi = phi[idx]
            val = _sigma * phi + _lamb * phi**3
            drift[idx] = val
        return _drift

    def action(self):
        _sigma = self.sigma
        _lamb = self.lamb
        
        @backend.kernel
        def _drift(idx, action, phi):
            phi = phi[idx]
            val = 0.5*_sigma * phi**2 + 0.25*_lamb * phi**4
            action[idx] = val

        return _drift
