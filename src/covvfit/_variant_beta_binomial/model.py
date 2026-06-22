"""Top-level composition for variant beta-binomial model."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from covvfit._variant_beta_binomial.data import DispersionParams, ModelData
from covvfit._variant_beta_binomial.dispersion import (
    build_eta_rho,
    transform_eta_to_rho,
)
from covvfit._variant_beta_binomial.growth import LinearGrowthParams, compute_log_pi
from covvfit._variant_beta_binomial.likelihood import (
    make_beta_binomial_shapes,
    masked_beta_binomial_loglik,
)
from covvfit._variant_beta_binomial.mapping import compute_mu

# Axis-name sentinels for Ruff/Pyflakes handling of jaxtyping string annotations.
shape = None


def model_loglik(
    growth_params: LinearGrowthParams,
    dispersion_params: DispersionParams | Float[Array, "*shape"],
    data: ModelData,
) -> Float[Array, ""]:
    """Compute total log-likelihood for observed counts."""
    log_pi = compute_log_pi(
        theta=growth_params,
        city=data.city,
        time=data.time,
        X=data.X,
    )
    mu = compute_mu(log_pi=log_pi, A=data.A)

    if isinstance(dispersion_params, DispersionParams):
        eta_rho = build_eta_rho(dispersion_params.eta_rho)
    else:
        eta_rho = build_eta_rho(jnp.asarray(dispersion_params, dtype=float))

    rho = transform_eta_to_rho(eta_rho)
    alpha, beta = make_beta_binomial_shapes(mu=mu, rho=rho, city=data.city)
    return masked_beta_binomial_loglik(
        Y=data.Y,
        N=data.N,
        alpha=alpha,
        beta=beta,
        obs_mask=data.obs_mask,
    )
