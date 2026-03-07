"""Likelihood utilities for variant beta-binomial model."""

import jax.numpy as jnp
from jax.scipy.special import betaln, gammaln
from jaxtyping import Array, Bool, Float, Int

from covvfit._variant_beta_binomial.dispersion import (
    rho_to_log_kappa,
    select_rho_per_sample,
)

# Axis-name sentinels for Ruff/Pyflakes handling of jaxtyping string annotations.
samples = loci = shape = None


def make_beta_binomial_shapes(
    mu: Float[Array, "samples loci"],
    rho: Float[Array, "*shape"],
    city: Int[Array, "samples"] | None = None,
    eps_mu: float = 1e-6,
) -> tuple[Float[Array, "samples loci"], Float[Array, "samples loci"]]:
    """Create alpha/beta shape parameters for beta-binomial distribution."""
    mu = jnp.asarray(mu, dtype=float)
    if mu.ndim != 2:
        raise ValueError("mu must have shape (samples, loci).")

    mu_clip = jnp.clip(mu, eps_mu, 1.0 - eps_mu)

    rho_selected = select_rho_per_sample(rho=rho, city=city)
    log_kappa = rho_to_log_kappa(rho_selected)
    kappa = jnp.exp(log_kappa)

    if kappa.ndim == 1:
        kappa = jnp.broadcast_to(kappa, mu.shape)
    elif kappa.ndim == 2 and kappa.shape != mu.shape:
        raise ValueError("When city-specific, rho selected shape must match mu shape.")

    alpha = mu_clip * kappa
    beta = (1.0 - mu_clip) * kappa
    return alpha, beta


def beta_binomial_logpmf(
    Y: Int[Array, "samples loci"],
    N: Int[Array, "samples loci"],
    alpha: Float[Array, "samples loci"],
    beta: Float[Array, "samples loci"],
) -> Float[Array, "samples loci"]:
    """Compute beta-binomial log pmf elementwise."""
    Y = jnp.asarray(Y, dtype=float)
    N = jnp.asarray(N, dtype=float)
    alpha = jnp.asarray(alpha, dtype=float)
    beta = jnp.asarray(beta, dtype=float)

    if Y.shape != N.shape or Y.shape != alpha.shape or Y.shape != beta.shape:
        raise ValueError("Y, N, alpha and beta must have the same shape.")

    log_binom = gammaln(N + 1.0) - gammaln(Y + 1.0) - gammaln(N - Y + 1.0)
    return log_binom + betaln(Y + alpha, N - Y + beta) - betaln(alpha, beta)


def masked_beta_binomial_loglik(
    Y: Int[Array, "samples loci"],
    N: Int[Array, "samples loci"],
    alpha: Float[Array, "samples loci"],
    beta: Float[Array, "samples loci"],
    obs_mask: Bool[Array, "samples loci"],
) -> Float[Array, ""]:
    """Compute total masked beta-binomial log-likelihood."""
    loglik_matrix = beta_binomial_logpmf(Y=Y, N=N, alpha=alpha, beta=beta)
    obs_mask = jnp.asarray(obs_mask)
    if obs_mask.shape != loglik_matrix.shape:
        raise ValueError("obs_mask must have the same shape as Y and N.")
    if obs_mask.dtype != jnp.bool_:
        raise ValueError("obs_mask must have boolean dtype.")

    return jnp.where(obs_mask, loglik_matrix, 0.0).sum()
