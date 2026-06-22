import jax
import jax.numpy as jnp
import numpy.testing as npt
from covvfit._variant_beta_binomial.mapping import (
    compute_log_mu,
    compute_mu,
    make_log_A_mask,
)


def test_make_log_A_mask_entries() -> None:
    A = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)
    log_A = make_log_A_mask(A, dtype=jnp.float32)

    assert log_A[0, 0] == 0.0
    assert log_A[1, 1] == 0.0
    assert log_A[0, 1] == jnp.finfo(jnp.float32).min


def test_compute_log_mu_shape() -> None:
    log_pi = jnp.array([[0.0, -1.0], [-2.0, 0.0]], dtype=float)
    A = jnp.array([[1, 0, 1], [0, 1, 1]], dtype=jnp.int32)

    log_mu = compute_log_mu(log_pi=log_pi, A=A)
    assert log_mu.shape == (2, 3)


def test_compute_mu_agrees_with_naive() -> None:
    log_pi = jnp.array([[0.2, -0.3, 0.1], [0.0, 0.1, -0.4]], dtype=float)
    A = jnp.array(
        [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 1],
        ],
        dtype=float,
    )

    mu = compute_mu(log_pi=log_pi, A=A)
    naive = jax.nn.softmax(log_pi, axis=-1) @ A
    npt.assert_allclose(mu, naive, atol=1e-6)


def test_compute_mu_extreme_probabilities_stable() -> None:
    log_pi = jnp.array([[0.0, -1e9, -1e9]], dtype=float)
    A = jnp.array([[1, 0], [0, 1], [1, 1]], dtype=jnp.int32)

    mu = compute_mu(log_pi=log_pi, A=A)
    assert jnp.isfinite(mu).all()
    npt.assert_allclose(mu[0], jnp.array([1.0, 0.0]), atol=1e-6)


def test_invalid_locus_gives_zero_probability() -> None:
    log_pi = jnp.array([[0.0, 1.0]], dtype=float)
    A = jnp.array([[1, 0], [0, 0]], dtype=jnp.int32)

    mu = compute_mu(log_pi=log_pi, A=A)
    assert jnp.isclose(mu[0, 1], 0.0)
