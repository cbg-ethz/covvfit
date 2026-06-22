"""Dispersion parameterization utilities for beta-binomial model."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

# Axis-name sentinels for Ruff/Pyflakes handling of jaxtyping string annotations.
samples = features = shape = None


def build_eta_rho(
    params_rho: Float[Array, "*shape"],
    city: Int[Array, "samples"] | None = None,
    X: Float[Array, "samples features"] | None = None,
) -> Float[Array, "*shape"]:
    """Construct unconstrained predictor for rho.

    Current implementation is identity, with placeholders for future predictors.
    """
    del city, X
    return jnp.asarray(params_rho, dtype=float)


def transform_eta_to_rho(
    eta_rho: Float[Array, "*shape"], eps_rho: float = 1e-8
) -> Float[Array, "*shape"]:
    """Map unconstrained predictor to rho in (eps_rho, 1-eps_rho)."""
    eta_rho = jnp.asarray(eta_rho, dtype=float)
    min_eps = float(jnp.finfo(eta_rho.dtype).eps)
    eps = jnp.asarray(max(eps_rho, min_eps), dtype=eta_rho.dtype)
    return eps + (1.0 - 2.0 * eps) * jax.nn.sigmoid(eta_rho)


def rho_to_log_kappa(rho: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """Compute log concentration kappa = (1-rho)/rho."""
    rho = jnp.asarray(rho, dtype=float)
    return jnp.log1p(-rho) - jnp.log(rho)


def select_rho_per_sample(
    rho: Float[Array, "*shape"], city: Int[Array, "samples"] | None
) -> Float[Array, "*shape"]:
    """Select rho by city when rho is city-specific."""
    rho = jnp.asarray(rho, dtype=float)

    if rho.ndim == 1:
        return rho
    if rho.ndim == 2:
        if city is None:
            raise ValueError("city is required when rho has shape (cities, loci).")
        city = jnp.asarray(city)
        if city.ndim != 1:
            raise ValueError("city must have shape (samples,).")
        return rho[city]

    raise ValueError("rho must have shape (loci,) or (cities, loci).")
