
### complex_langevin/simulation/cl_evolution.py
from complex_langevin.models.base import Model 
from complex_langevin.compiler.factory import get_backend
from complex_langevin.config import CL_REAL, CL_COMPLEX
from complex_langevin.simulation.state import SimState

from numba.cuda.random import xoroshiro128p_normal_float32, create_xoroshiro128p_states
backend = get_backend()

import math 
SQRT2 = math.sqrt(2)

import numpy as np
class cl_evolution():
    def __init__(self, model: Model, simstate: SimState):
        self.model = model

        # Initialize kernels
        self.drift_kernel = model.drift()
        self.action_kernel = model.action()
        self.noise_seed = 0
        self.n_seeds = simstate.n_seeds
        self.simstate = simstate
        self.backend = backend
        self.noise_arr = simstate.noise_arr

        if backend.use_cuda:
            n_blocks = math.ceil(self.n_seeds / backend.threadsperblock)
            self.rng = create_xoroshiro128p_states(
                backend.threadsperblock * n_blocks, seed=self.noise_seed
            )
        else:
            self.rng = None

    def generate_noise(self):
        if self.backend.use_cuda:
            @self.backend.kernel
            def _generate_noise(idx, noise_arr, rng):
                r = xoroshiro128p_normal_float32(rng, idx)
                noise_arr[idx] = SQRT2 * r
            return _generate_noise
        
        else:
            import numpy as np
            @self.backend.kernel
            def _generate_noise(idx, noise_arr):
                noise_arr[idx] = SQRT2 * CL_REAL(np.random.normal())
            return _generate_noise