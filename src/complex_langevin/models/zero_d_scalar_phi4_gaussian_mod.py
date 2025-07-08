
### complex_langevin/compiler/
from .base import Model
from complex_langevin.compiler.factory import get_backend
from complex_langevin.config import CL_REAL, CL_COMPLEX, CL_INT

backend = get_backend()

class ZeroDScalarPhi4Gaussian(Model):
    def __init__(self, sigma: CL_COMPLEX, lamb: CL_REAL, massmod: CL_COMPLEX, pull: CL_REAL):
        self.sigma = sigma
        self.lamb = lamb
        self.massmod = massmod
        self.pull = pull

    def drift(self):
        import cmath

        _sigma = self.sigma
        _lamb = self.lamb
        _massmod = self.massmod
        _pull = self.pull

        @backend.kernel
        def _drift(idx, drift_arr, phi_arr):
            phi_idx = phi_arr[idx]

            action = _sigma/2*phi_idx**2+_lamb/4*phi_idx**4
            mod = -_massmod*phi_idx**2/2

            action_mod = action + mod
            drift = _sigma*phi_idx+_lamb*phi_idx**3

            if action_mod.real < 0:
                out = drift-_pull*(drift-_massmod*phi_idx)*cmath.exp(action_mod) / (1+_pull*cmath.exp(action_mod))
            else: 
                out = drift-_pull*(drift-_massmod*phi_idx) / (cmath.exp(-action_mod)+_pull)
            drift_arr[idx] = out

        return _drift
    
    # def action(self):
    #     import cmath

    #     _sigma = self.sigma
    #     _lamb = self.lamb
    #     _massmod = self.massmod
    #     _pull = self.pull

    #     @backend.kernel
    #     def _action(idx, action_arr, phi_arr):
    #         phi_idx = phi_arr[idx]

    #         action = _sigma/2*phi_idx**2+_lamb/4*phi_idx**4
    #         # mod = -cmath.log(1+)

    #         action_arr[idx] = 1# action + mod

    #     return _action