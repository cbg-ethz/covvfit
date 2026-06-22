import jax
import jax.numpy as jnp
from covvfit._variant_beta_binomial.growth import LinearGrowthParams
from covvfit._variant_beta_binomial.simulate import (
    sample_beta_binomial_counts,
    simulate_dataset,
    simulate_log_pi,
    simulate_mu,
)


def test_sample_beta_binomial_counts_shape() -> None:
    key = jax.random.PRNGKey(0)
    N = jnp.full((4, 3), 20, dtype=jnp.int32)
    alpha = jnp.full((4, 3), 2.0)
    beta = jnp.full((4, 3), 5.0)

    Y = sample_beta_binomial_counts(key=key, N=N, alpha=alpha, beta=beta)
    assert Y.shape == N.shape
    assert jnp.all(Y >= 0)
    assert jnp.all(Y <= N)


def test_simulate_log_pi_and_mu_shapes() -> None:
    growth = LinearGrowthParams(
        intercepts=jnp.array([[0.0, 1.0], [0.5, -0.2]]),
        slopes=jnp.array([0.1, -0.1]),
    )
    city = jnp.array([0, 1, 0], dtype=jnp.int32)
    time = jnp.array([0.0, 1.0, 2.0], dtype=float)
    A = jnp.array([[1, 0, 1], [0, 1, 1]], dtype=bool)

    log_pi = simulate_log_pi(growth_params=growth, city=city, time=time)
    mu = simulate_mu(log_pi=log_pi, A=A)

    assert log_pi.shape == (3, 2)
    assert mu.shape == (3, 3)
    assert jnp.all(mu >= 0.0)
    assert jnp.all(mu <= 1.0)


def test_simulate_dataset_smoke_with_mask() -> None:
    key = jax.random.PRNGKey(42)
    n_samples = 10
    n_loci = 4

    growth = LinearGrowthParams(
        intercepts=jnp.array([[0.0, 0.8, -0.3], [0.2, -0.5, 0.3]]),
        slopes=jnp.array([0.1, -0.05, 0.02]),
    )
    city = jnp.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    time = jnp.linspace(0.0, 5.0, n_samples)
    A = jnp.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 1]], dtype=bool)
    coverage = jnp.full((n_samples, n_loci), 50, dtype=jnp.int32)
    eta_rho = jnp.array([0.0, -0.2, 0.4, -0.1])

    result = simulate_dataset(
        key=key,
        A=A,
        growth_params=growth,
        eta_rho=eta_rho,
        city=city,
        time=time,
        coverage=coverage,
        mask_probability=0.15,
    )

    assert result.data.Y.shape == (n_samples, n_loci)
    assert result.data.obs_mask.shape == (n_samples, n_loci)
    assert jnp.any(~result.data.obs_mask)
