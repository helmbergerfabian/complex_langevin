### complex_langevi/simulation/state.py

# from complex_langevin.compiler.factory import get_backend
from complex_langevin.compiler import backend
from complex_langevin.config import CL_COMPLEX, CL_REAL

import numpy as np
import os
target = os.environ.get('MY_NUMBA_TARGET', 'numba').lower()


class SimState:
    """
    Holds simulation state for multiple Langevin trajectories (seeds).
    Supports CPU or GPU (CUDA) via backend abstraction.
    """
    def __init__(self, n_seeds: int, dt_base = None):
        self.n_seeds = n_seeds
        self.dt_base = dt_base or CL_REAL(1e-4)
        
        self.phi_read = np.zeros(n_seeds, dtype=CL_COMPLEX)
        # self.phi_write = self.phi_read.copy()

        self.noise_arr = np.zeros(n_seeds, dtype=CL_REAL)
        self.dt_ada_arr = np.ones(n_seeds, dtype=CL_REAL)
        self.drift_arr = np.zeros(n_seeds, dtype=CL_COMPLEX)
        self.langevin_time = np.zeros(n_seeds, dtype=CL_REAL)

        if backend.use_cuda: 
            from complex_langevin.utils.gpu_handler import GPU_handler
            self.handler = GPU_handler(self)
            self.to_device()
        
    def to_device(self):
        self.handler.to_device()
        print("Copied state arrays to device")

    def to_host(self):
        self.handler.to_host()
        print("Copied state arrays to host")

    # def swap_buffers(self):
    #     self.phi_read, self.phi_write = self.phi_write, self.phi_read
