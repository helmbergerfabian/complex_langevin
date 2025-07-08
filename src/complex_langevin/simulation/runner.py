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

    def __init__(self, state: SimState, model: Model, evolution: cl_evolution):
        self.backend = get_backend()
        self.use_cuda = self.backend.use_cuda

        self.state = state
        self.model = model
        self.evolution = evolution

        self.drift_kernel = model.drift()
        self.noise_kernel = evolution.generate_noise()

    def update_noise(self):
        """Generates standard Gaussian noise for each seed."""
        self.backend.parallel_loop(self.noise_kernel, self.state.n_seeds, self.state.noise_arr)