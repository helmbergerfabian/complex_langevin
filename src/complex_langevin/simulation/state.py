### complex_langevi/simulation/state.py

from complex_langevin.compiler.parallel import parallel_loop, act_parallel_loop
# from complex_langevin.compiler.factory import get_backend
from complex_langevin.config import CL_COMPLEX, CL_REAL

import numpy as np
import os
target = os.environ.get('MY_NUMBA_TARGET', 'numba').lower()


class SimState:
    """
    Holds simulation state for multiple Langevin trajectories (seeds).
    Supports CPU or GPU (CUDA) via backend abstraction.
    """
    def __init__(self, n_seeds: int):
        self.n_seeds = n_seeds
        self.use_cuda = (target == "cuda")

        phi0 = np.zeros(n_seeds, dtype=CL_COMPLEX)
        noise0 = np.zeros(n_seeds, dtype=CL_REAL)
        langevin_time0 = np.zeros(n_seeds, dtype=CL_REAL)

        if self.use_cuda:
            from numba import cuda
            self.phi_read = cuda.to_device(phi0)
            self.phi_write = cuda.to_device(phi0.copy())
            self.noise = cuda.to_device(noise0)
            self.langevin_time = cuda.to_device(langevin_time0)
        else:
            self.phi_read = phi0
            self.phi_write = phi0.copy()
            self.noise = noise0
            self.langevin_time = langevin_time0

    def swap_buffers(self):
        self.phi_read, self.phi_write = self.phi_write, self.phi_read