import os
from time import time
os.environ["RUNNER_START_TIME"] = str(time())


from complex_langevin.simulation.state import SimState
from complex_langevin.models.base import Model
from complex_langevin.compiler.factory import get_backend
from complex_langevin.config import CL_REAL
from complex_langevin.simulation.cl_evolution import cl_evolution

from complex_langevin.config import log

import numpy as np

class SimulationRunner:
    """
    Controls the simulation loop for Complex Langevin evolution.
    """

    def __init__(self, state: SimState, model: Model, evolution: cl_evolution = None):
        self.backend = get_backend()
        self.use_cuda = self.backend.use_cuda

        self.state = state
        self.model = model
        self.evolution = evolution or cl_evolution(model, state)

        self.drift_kernel = model.generate_drift_kernel()
        self.noise_kernel = self.evolution.generate_noise_kernel()
        self.evolve_kernel = self.evolution.generate_evolution_kernel()
        self.dt_ada_kernel = self.evolution.generate_dt_ada_kernel()
        self.kill_kernel = self.evolution.generate_kill_kernel()
        self.rng = self.evolution.rng


    def update_noise(self):
        """Generates standard Gaussian noise for each seed."""
        self.backend.parallel_loop(self.noise_kernel, self.state.alive_count[0], self.state.alive_idx_list,
                                   self.state.noise_arr, self.rng)
        # self.log(f"upated noise: {self.state.alive_idx_list}")
    
    def update_drift(self):
        # self.log(f"update_drift: {self.state.alive_idx_list}")
        """Updates the drift term in the simulation state."""
        self.backend.parallel_loop(self.drift_kernel, self.state.alive_count[0], self.state.alive_idx_list,
                                   self.state.drift_arr, self.state.phi_read)

    def evolve(self):
        """Performs one step of the Complex Langevin evolution."""
        # self.log(f"evolve: {self.state.alive_idx_list}")
        self.backend.parallel_loop(self.evolve_kernel, self.state.alive_count[0], self.state.alive_idx_list,
                                   self.state.phi_read, self.state.drift_arr, 
                                   self.state.noise_arr, self.state.dt_ada_arr,
                                   self.state.dt_base, self.state.langevin_time
                                   )
    
    def update_dt_ada(self):
        # self.log(f"update_dt_ada: {self.state.alive_idx_list}")

        self.backend.parallel_loop(self.dt_ada_kernel, self.state.alive_count[0], self.state.alive_idx_list,
                                   self.state.dt_ada_arr, self.state.drift_arr
                                   )

    def kill_trajs(self):
        # self.log(f"kill_trajs: {self.state.alive_idx_list}")
        num_alive_copy = self.state.alive_count[0].copy()
        self.state.alive_count = np.array([0])
        self.backend.serial_loop(self.kill_kernel, num_alive_copy, self.state.alive_idx_list,
                                   self.state.dt_ada_arr, self.state.drift_arr, self.state.alive_idx_list, self.state.alive_count
                                   )

    def step(self):
        self.update_drift()
        self.update_dt_ada()
        self.kill_trajs()

        self.update_noise()
        self.evolve()
        self.state.global_step += 1
        # self.log(f"gloabl step: {self.state.global_step}")
        # self.state.swap_buffers()

    def log(self, message): log(self, "RUN", message)