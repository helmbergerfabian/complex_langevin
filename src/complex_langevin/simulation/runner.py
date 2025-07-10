import numpy as np
import os

from complex_langevin.simulation.state import SimState
from complex_langevin.models.base import Model
from complex_langevin.compiler.factory import get_backend
from complex_langevin.config import CL_REAL
from complex_langevin.simulation.cl_evolution import cl_evolution

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

        self.rng = self.evolution.rng

    def update_noise(self):
        """Generates standard Gaussian noise for each seed."""
        self.backend.parallel_loop(self.noise_kernel, self.state.n_seeds, 
                                   self.state.noise_arr, self.rng)
    
    def update_drift(self):
        """Updates the drift term in the simulation state."""
        self.backend.parallel_loop(self.drift_kernel, self.state.n_seeds, 
                                   self.state.drift_arr, self.state.phi_read)

    def evolve(self):
        """Performs one step of the Complex Langevin evolution."""
        self.backend.parallel_loop(self.evolve_kernel, self.state.n_seeds, 
                                   self.state.phi_write, self.state.drift_arr, 
                                   self.state.noise_arr, self.state.dt_arr)
    
    def step(self):
        self.update_noise()
        self.update_drift()
        self.evolve()
        self.state.swap_buffers()