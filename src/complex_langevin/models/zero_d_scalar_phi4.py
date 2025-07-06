
### complex_langevin/compiler/cuda_backend.py
from .base import Model
from complex_langevin.compiler.factory import get_backend
from complex_langevin.config import CL_REAL, CL_COMPLEX, CL_INT

backend = get_backend()

class ZeroDScalarPhi4(Model):
    def __init__(self, sigma: CL_COMPLEX, lamb: CL_REAL):
        self.sigma = sigma
        self.lamb = lamb

    @backend.kernel
    def drift(self, phi):
        return self.sigma * phi + self.lamb * phi**3

    @backend.kernel
    def action(self, phi):
        return 0.5 * self.sigma * phi**2 + 0.25 * self.lamb * phi**4
