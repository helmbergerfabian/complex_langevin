
### complex_langevin/simulation/cl_evolution.py
from complex_langevin.models.base import Model 
from complex_langevin.compiler.factory import get_backend
from complex_langevin.config import CL_REAL
from complex_langevin.simulation.state import SimState
from complex_langevin.utils.cl_types import NoiseKernel, EvolveKernel, dtadaKernel

from numba.cuda.random import xoroshiro128p_normal_float32, create_xoroshiro128p_states
backend = get_backend()

import math, cmath
SQRT2 = math.sqrt(2)
DS_MAX_LOWER = 1e-12
mean_dS_max = 1

import numpy as np

class cl_evolution():
    def __init__(self, model: Model, simstate: SimState) -> None:
        self.model = model

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

    def generate_noise_kernel(self) -> NoiseKernel:
        if self.backend.use_cuda:

            @self.backend.kernel
            def _noise_kernel(idx, noise_arr, rng) -> None:
                r = xoroshiro128p_normal_float32(rng, idx)
                noise_arr[idx] = SQRT2 * r

            return _noise_kernel
        else:
            @self.backend.kernel
            def _generate_noise(idx, noise_arr, rng) -> None:
                noise_arr[idx] = SQRT2 * CL_REAL(np.random.normal())

            return _generate_noise

    def generate_evolution_kernel(self) -> EvolveKernel:

        @self.backend.kernel
        def _evolve_kernel(idx, phi_arr, drift_arr, noise_arr, dt_ada_arr, dt_base) -> None:
            dt_idx = dt_ada_arr[idx]*dt_base
            phi_arr[idx] += dt_idx * drift_arr[idx] + math.sqrt(dt_idx) * noise_arr[idx]

        return _evolve_kernel
    
    def generate_dt_ada_kernel(self) -> dtadaKernel:

        @self.backend.kernel
        def _dt_ada_kernel(idx, dt_ada_arr, drift_arr) -> None:
            drift_idx = abs(drift_arr[idx])
            dt_ada_arr[idx] = 1
            if drift_idx > DS_MAX_LOWER and mean_dS_max < drift_idx:
                dt_ada_arr[idx] = mean_dS_max / drift_idx 

        return _dt_ada_kernel