import numpy as np
import pytest
from complex_langevin.models.zero_d_scalar_phi4 import ZeroDScalarPhi4
from complex_langevin.config import CL_REAL, CL_COMPLEX, CL_PRECISION


@pytest.fixture
def sample_field():
    # Small complex field for simple validation
    return np.array([1.0 + 0.5j, -1.0 - 0.5j], dtype=CL_COMPLEX)


@pytest.mark.parametrize("sigma, lamb", [
    (1.0 + 0.0j, 1.0),           # simple
    (-1.0 + 1.0j, 0.5),          # complex σ
    (0.0 + 0.0j, 1e-3),          # edge case: zero sigma
])
def test_drift_against_manual(sigma, lamb, sample_field):
    model = ZeroDScalarPhi4(sigma=CL_COMPLEX(sigma), lamb=CL_REAL(lamb))

    drift_func = model.generate_drift_kernel()
    drift_array = np.zeros_like(sample_field, dtype=CL_COMPLEX)
    idx = 0
    drift_func(idx, drift_array, sample_field)
    expected = -(sigma * sample_field[idx] + lamb * sample_field[idx]**3)
    
    np.testing.assert_allclose(drift_array[idx], expected, rtol=1e-5, atol=1e-7)


# def test_action_against_manual(sample_field):
#     sigma = CL_COMPLEX(2.0 + 0j)
#     lamb = CL_REAL(0.25)
#     model = ZeroDScalarPhi4(sigma, lamb)

#     action_func = model.action()
#     action_array = np.zeros_like(sample_field, dtype=CL_COMPLEX)
#     idx = 0
#     action_func(idx, action_array, sample_field)
#     expected = sigma * sample_field[idx]**2/2 + lamb * sample_field[idx]**4/4

#     np.testing.assert_allclose(action_array[idx], expected, rtol=1e-5, atol=1e-7)


# def test_precision_dtype(sample_field):
#     sigma = CL_COMPLEX(2.0 + 0j)
#     lamb = CL_REAL(0.25)
#     model = ZeroDScalarPhi4(sigma, lamb)

#     action_func = model.action()
#     action_array = np.zeros_like(sample_field, dtype=CL_COMPLEX)
#     idx = 0
#     action_func(idx, action_array, sample_field)

#     result = action_array[0]
#     assert result.dtype == CL_COMPLEX
