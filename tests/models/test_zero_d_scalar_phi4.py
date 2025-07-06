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

    phi = sample_field
    drift = model.drift(phi)
    
    expected = sigma * phi + lamb * phi**3
    
    np.testing.assert_allclose(drift, expected, rtol=1e-5, atol=1e-7)


def test_action_against_manual(sample_field):
    sigma = CL_COMPLEX(2.0 + 0j)
    lamb = CL_REAL(0.25)

    model = ZeroDScalarPhi4(sigma, lamb)

    phi = sample_field
    action = model.action(phi)

    expected = 0.5 * sigma * phi**2 + 0.25 * lamb * phi**4

    np.testing.assert_allclose(action, expected, rtol=1e-5, atol=1e-7)


def test_precision_dtype(sample_field):
    sigma = CL_COMPLEX(1.0 + 0j)
    lamb = CL_REAL(1.0)
    model = ZeroDScalarPhi4(sigma, lamb)

    result = model.drift(sample_field)
    assert result.dtype == CL_COMPLEX
