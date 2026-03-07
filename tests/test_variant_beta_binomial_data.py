import jax.numpy as jnp
import pytest
from covvfit._variant_beta_binomial.data import DispersionParams, ModelData


def _valid_data_inputs():
    Y = jnp.array([[1, 2], [0, 1]], dtype=jnp.int32)
    N = jnp.array([[3, 3], [2, 2]], dtype=jnp.int32)
    city = jnp.array([0, 1], dtype=jnp.int32)
    time = jnp.array([0.0, 1.0], dtype=float)
    A = jnp.array([[1, 0], [1, 1]], dtype=bool)
    obs_mask = jnp.array([[True, True], [False, True]], dtype=bool)
    X = jnp.array([[0.1], [0.2]], dtype=float)
    return Y, N, city, time, A, obs_mask, X


def test_model_data_valid_shapes() -> None:
    Y, N, city, time, A, obs_mask, X = _valid_data_inputs()
    data = ModelData(Y=Y, N=N, city=city, time=time, A=A, obs_mask=obs_mask, X=X)

    assert data.Y.shape == (2, 2)
    assert data.N.shape == (2, 2)
    assert data.city.shape == (2,)
    assert data.time.shape == (2,)
    assert data.A.shape == (2, 2)
    assert data.obs_mask.shape == (2, 2)
    assert data.X is not None and data.X.shape == (2, 1)


def test_model_data_rejects_non_boolean_mask() -> None:
    Y, N, city, time, A, _, X = _valid_data_inputs()
    obs_mask = jnp.array([[1, 1], [0, 1]], dtype=jnp.int32)
    with pytest.raises(ValueError, match="obs_mask must have boolean dtype"):
        ModelData(Y=Y, N=N, city=city, time=time, A=A, obs_mask=obs_mask, X=X)


def test_model_data_rejects_non_integer_city() -> None:
    Y, N, _, time, A, obs_mask, X = _valid_data_inputs()
    city = jnp.array([0.0, 1.0], dtype=float)
    with pytest.raises(ValueError, match="city must have integer dtype"):
        ModelData(Y=Y, N=N, city=city, time=time, A=A, obs_mask=obs_mask, X=X)


def test_model_data_rejects_non_binary_A() -> None:
    Y, N, city, time, _, obs_mask, X = _valid_data_inputs()
    A = jnp.array([[1.0, 0.5], [1.0, 1.0]], dtype=float)
    with pytest.raises(ValueError, match="A must be binary"):
        ModelData(Y=Y, N=N, city=city, time=time, A=A, obs_mask=obs_mask, X=X)


def test_model_data_rejects_missing_locus_support() -> None:
    Y, N, city, time, _, obs_mask, X = _valid_data_inputs()
    A = jnp.array([[1, 0], [0, 0]], dtype=bool)
    with pytest.raises(ValueError, match="Every locus must be present"):
        ModelData(Y=Y, N=N, city=city, time=time, A=A, obs_mask=obs_mask, X=X)


def test_dispersion_params_shapes() -> None:
    params_1d = DispersionParams(eta_rho=jnp.zeros((4,)))
    params_2d = DispersionParams(eta_rho=jnp.zeros((3, 4)))

    assert params_1d.eta_rho.shape == (4,)
    assert params_2d.eta_rho.shape == (3, 4)


def test_dispersion_params_invalid_rank() -> None:
    with pytest.raises(ValueError, match="eta_rho must have shape"):
        DispersionParams(eta_rho=jnp.zeros((2, 3, 4)))
