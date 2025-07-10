# tests/test_simulation_runner_real.py

import numpy as np
import pytest

from complex_langevin.simulation.runner import SimulationRunner
from complex_langevin.simulation.state import SimState
from complex_langevin.simulation.cl_evolution import cl_evolution
from complex_langevin.models.zero_d_scalar_phi4 import ZeroDScalarPhi4

@pytest.mark.parametrize("n_seeds", [4])
def test_simulation_runner_step_modifies_state(n_seeds):
    model = ZeroDScalarPhi4(1 - 1j, 1)
    state = SimState(n_seeds)

    # Save initial values for comparison
    phi_before = state.phi_read.copy()
    drift_before = state.drift_arr.copy()
    noise_before = state.noise_arr.copy()

    runner = SimulationRunner(state=state, model=model)
    runner.step()
    runner.step()

    # Check that fields have changed — simulation has evolved the state
    assert not np.allclose(phi_before, state.phi_read), "phi_read did not change after step()"
    assert not np.allclose(noise_before, state.noise_arr), "Noise array unchanged — noise kernel may not have run"
    assert not np.allclose(drift_before, state.drift_arr), "Drift not updated"
    assert np.all(np.isfinite(state.drift_arr)), "Drift contains non-finite values"
    assert np.all(state.dt_ada_arr > 0), "Adaptive timestep should be positive"
