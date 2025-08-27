### complex_langevi/simulation/state.py

# from complex_langevin.compiler.factory import get_backend
from complex_langevin.compiler import backend
from complex_langevin.config import CL_COMPLEX, CL_REAL, log, CL_INT

from numpy import zeros, ones
import os
target = os.environ.get('MY_NUMBA_TARGET', 'numba').lower()

from time import time
import numpy as np

class SimState:
    """
    Holds simulation state for multiple Langevin trajectories (seeds).
    Supports CPU or GPU (CUDA) via backend abstraction.
    """
    def __init__(self, n_seeds: int = None, dt_base = None):
        self.n_seeds = n_seeds if n_seeds is not None else int(1e4)
        self.dt_base = dt_base if dt_base is not None else CL_REAL(1e-4)
        
        self.phi_read = zeros(self.n_seeds, dtype=CL_COMPLEX)
        # self.phi_write = self.phi_read.copy()

        self.noise_arr = zeros(self.n_seeds, dtype=CL_REAL)
        self.dt_ada_arr = ones(self.n_seeds, dtype=CL_REAL)
        self.drift_arr = zeros(self.n_seeds, dtype=CL_COMPLEX)
        self.langevin_time = zeros(self.n_seeds, dtype=CL_REAL)

        self.global_step = 0
        self.alive = np.full(self.n_seeds, True)
        self.alive_count = np.array([self.n_seeds], dtype = CL_INT)
        self.alive_idx_list = np.arange(self.alive_count, dtype = CL_INT)
        self.zero_buffer = np.array([0], dtype = CL_INT)

        if backend.use_cuda: 
            from complex_langevin.utils.gpu_handler import GPU_handler
            self.handler = GPU_handler(self)
            self.to_device()
        
    def to_device(self):
        self.handler.to_device()
        self.log("Copied state arrays to device")

    def to_host(self):
        self.handler.to_host()
        self.log("Copied state arrays to host")

    def log(self, message): log(self, "STATE", message)