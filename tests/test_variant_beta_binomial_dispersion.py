import jax.numpy as jnp
import numpy.testing as npt
from covvfit._variant_beta_binomial.dispersion import (
    rho_to_log_kappa,
    select_rho_per_sample,
    transform_eta_to_rho,
)


def test_transform_eta_to_rho_bounds() -> None:
    eta = jnp.array([-100.0, 0.0, 100.0])
    rho = transform_eta_to_rho(eta, eps_rho=1e-8)

    assert jnp.all(rho > 1e-8)
    assert jnp.all(rho < 1.0 - 1e-8)


def test_transform_eta_to_rho_shape() -> None:
    eta = jnp.zeros((3, 4))
    rho = transform_eta_to_rho(eta)
    assert rho.shape == eta.shape


def test_transform_eta_to_rho_monotonic() -> None:
    eta = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    rho = transform_eta_to_rho(eta)

    assert jnp.all(jnp.diff(rho) > 0)


def test_rho_to_log_kappa_finite_extremes() -> None:
    eta = jnp.array([-100.0, 0.0, 100.0])
    rho = transform_eta_to_rho(eta)
    log_kappa = rho_to_log_kappa(rho)

    assert jnp.isfinite(log_kappa).all()


def test_select_rho_per_sample_city_specific() -> None:
    rho = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    city = jnp.array([2, 0, 1, 2], dtype=jnp.int32)

    selected = select_rho_per_sample(rho=rho, city=city)
    expected = jnp.array([[0.5, 0.6], [0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    npt.assert_allclose(selected, expected)
