### complex_langevi/simulation/state.py

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
        # self.use_cuda = (target == "cuda")

        self.phi_read = np.zeros(n_seeds, dtype=CL_COMPLEX)
        self.phi_write = self.phi_read.copy()

        self.noise = np.zeros(n_seeds, dtype=CL_REAL)
        self.langevin_time = np.zeros(n_seeds, dtype=CL_REAL)

    def swap_buffers(self):
        self.phi_read, self.phi_write = self.phi_write, self.phi_read