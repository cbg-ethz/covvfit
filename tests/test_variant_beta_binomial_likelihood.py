import jax.numpy as jnp
from covvfit._variant_beta_binomial.likelihood import (
    beta_binomial_logpmf,
    make_beta_binomial_shapes,
    masked_beta_binomial_loglik,
)


def test_make_beta_binomial_shapes_shape() -> None:
    mu = jnp.array([[0.2, 0.8], [0.3, 0.7]], dtype=float)
    rho = jnp.array([0.1, 0.2], dtype=float)

    alpha, beta = make_beta_binomial_shapes(mu=mu, rho=rho)
    assert alpha.shape == mu.shape
    assert beta.shape == mu.shape


def test_make_beta_binomial_shapes_positive() -> None:
    mu = jnp.array([[0.2, 0.8]], dtype=float)
    rho = jnp.array([0.1, 0.2], dtype=float)

    alpha, beta = make_beta_binomial_shapes(mu=mu, rho=rho)
    assert jnp.all(alpha > 0)
    assert jnp.all(beta > 0)


def test_make_beta_binomial_shapes_clipping() -> None:
    mu = jnp.array([[0.0, 1.0]], dtype=float)
    rho = jnp.array([0.2, 0.3], dtype=float)

    alpha, beta = make_beta_binomial_shapes(mu=mu, rho=rho)
    assert jnp.isfinite(alpha).all()
    assert jnp.isfinite(beta).all()
    assert jnp.all(alpha > 0)
    assert jnp.all(beta > 0)


def test_beta_binomial_logpmf_finite() -> None:
    Y = jnp.array([[1, 2], [0, 1]], dtype=jnp.int32)
    N = jnp.array([[3, 3], [2, 2]], dtype=jnp.int32)
    alpha = jnp.array([[1.0, 2.0], [1.5, 1.2]], dtype=float)
    beta = jnp.array([[2.0, 1.0], [1.3, 2.1]], dtype=float)

    out = beta_binomial_logpmf(Y=Y, N=N, alpha=alpha, beta=beta)
    assert jnp.isfinite(out).all()


def test_masked_entries_contribute_zero() -> None:
    Y = jnp.array([[1, 2]], dtype=jnp.int32)
    N = jnp.array([[3, 3]], dtype=jnp.int32)
    alpha = jnp.array([[1.0, 2.0]], dtype=float)
    beta = jnp.array([[2.0, 1.0]], dtype=float)
    mask_full = jnp.array([[True, True]])
    mask_partial = jnp.array([[True, False]])

    full = masked_beta_binomial_loglik(
        Y=Y, N=N, alpha=alpha, beta=beta, obs_mask=mask_full
    )
    part = masked_beta_binomial_loglik(
        Y=Y, N=N, alpha=alpha, beta=beta, obs_mask=mask_partial
    )

    cell = beta_binomial_logpmf(Y=Y, N=N, alpha=alpha, beta=beta)[0, 1]
    assert jnp.isclose(full - part, cell)
