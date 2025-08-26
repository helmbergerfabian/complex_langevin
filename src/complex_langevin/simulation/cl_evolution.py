
### complex_langevin/simulation/cl_evolution.py
from complex_langevin.models.base import Model 
from complex_langevin.compiler.factory import get_backend
from complex_langevin.compiler import backend
from complex_langevin.config import CL_REAL, log
from complex_langevin.simulation.state import SimState
from complex_langevin.utils.cl_types import NoiseKernel, EvolveKernel, dtadaKernel
from numpy.random import normal

from numba.cuda.random import xoroshiro128p_normal_float32, create_xoroshiro128p_states
from numba import cuda

import math
SQRT2 = math.sqrt(2)
DS_MAX_LOWER = 1e-12
mean_dS_max = 100


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

        if backend.use_cuda: 
            from complex_langevin.utils.gpu_handler import GPU_handler
            self.handler = GPU_handler(self)
            self.to_device()
        
    def to_device(self):
        self.handler.to_device()
        self.log("Copied evolution arrays to device")

    def to_host(self):
        self.handler.to_host()
        self.log("Copied evolution arrays to host")  

    def generate_noise_kernel(self) -> NoiseKernel:
        if self.backend.use_cuda:

            @self.backend.kernel
            def _noise_kernel(idx, idx_list, noise_arr, rng) -> None:
                _idx = idx_list[idx]
                r = xoroshiro128p_normal_float32(rng, _idx)
                noise_arr[_idx] = SQRT2 * r

            return _noise_kernel
        else:
            @self.backend.kernel
            def _generate_noise(idx, idx_list, noise_arr, rng) -> None:
                _idx = idx_list[idx]
                noise_arr[_idx] = SQRT2 * CL_REAL(normal())

            return _generate_noise

    def generate_evolution_kernel(self) -> EvolveKernel:

        @self.backend.kernel
        def _evolve_kernel(idx, idx_list, phi_arr, drift_arr, noise_arr, dt_ada_arr, dt_base, langevin_time) -> None:
            _idx = idx_list[idx]
            dt_idx = dt_ada_arr[_idx]*dt_base
            phi_arr[_idx] += dt_idx * drift_arr[_idx] + math.sqrt(dt_idx) * noise_arr[_idx]
            langevin_time[_idx] += dt_idx

        return _evolve_kernel
    
    def generate_dt_ada_kernel(self) -> dtadaKernel:
        @self.backend.kernel
        def _dt_ada_kernel(idx, idx_list, dt_ada_arr, drift_arr) -> None:
            _idx = idx_list[idx]
            drift_idx = abs(drift_arr[_idx])
            dt_ada_arr[_idx] = 1
            if drift_idx > DS_MAX_LOWER and mean_dS_max < drift_idx:
                new_val = mean_dS_max / drift_idx 
                dt_ada_arr[_idx] = new_val

        return _dt_ada_kernel

    
    def generate_kill_kernel(self):
        if self.backend.use_cuda: 
            @self.backend.kernel
            def _kill_kernel(idx, idx_list, dt_ada_arr, alive, out_indices, out_count) -> None:
                _idx = idx_list[idx]
                
                if dt_ada_arr[_idx] < 1e-3: 
                    alive[_idx] = False
                else:
                    j = cuda.atomic.add(out_count, 0, 1)
                    out_indices[j] = idx

        else: 
            @self.backend.kernel
            def _kill_kernel(idx, idx_list, dt_ada_arr, alive, out_indices, out_count) -> None:
                _idx = idx_list[idx]
                
                if dt_ada_arr[_idx] < 1e-3: 
                    alive[_idx] = False
                else:
                    out_indices[out_count[0]] = _idx
                    out_count[0] += 1

        return _kill_kernel
    
        

    def log(self, message): log(self, "EVO", message)