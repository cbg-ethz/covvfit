"""NumPyro models for variant beta-binomial inference."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from covvfit._variant_beta_binomial.data import DispersionParams, ModelData
from covvfit._variant_beta_binomial.growth import LinearGrowthParams
from covvfit._variant_beta_binomial.model import model_loglik


def _prepend_reference(x_rel: jnp.ndarray) -> jnp.ndarray:
    """Add a reference variant fixed at zero for identifiability."""
    if x_rel.ndim == 1:
        return jnp.concatenate([jnp.zeros((1,), dtype=x_rel.dtype), x_rel], axis=0)
    if x_rel.ndim == 2:
        return jnp.concatenate(
            [jnp.zeros((x_rel.shape[0], 1), dtype=x_rel.dtype), x_rel], axis=1
        )
    raise ValueError("Expected rank-1 or rank-2 array for reference prepend.")


def numpyro_model_hierarchical(data: ModelData, n_cities: int) -> None:
    """Hierarchical Bayesian model for growth + dispersion parameters.

    The first variant is constrained as reference (slope and intercept fixed at 0).
    """
    n_variants = data.A.shape[0]
    n_loci = data.A.shape[1]

    # Shared prior for relative slopes across variants (excluding reference).
    slope_loc = numpyro.sample("slope_loc", dist.Normal(0.0, 0.3))
    slope_scale = numpyro.sample("slope_scale", dist.HalfNormal(0.3))
    slope_raw = numpyro.sample(
        "slope_raw", dist.Normal(0.0, 1.0).expand([n_variants - 1]).to_event(1)
    )
    slopes_rel = slope_loc + slope_scale * slope_raw
    slopes = _prepend_reference(slopes_rel)

    # City-specific relative intercepts with hierarchical pooling.
    int_loc = numpyro.sample(
        "intercept_loc",
        dist.Normal(0.0, 1.0).expand([n_variants - 1]).to_event(1),
    )
    int_scale = numpyro.sample(
        "intercept_scale",
        dist.HalfNormal(1.0).expand([n_variants - 1]).to_event(1),
    )
    int_raw = numpyro.sample(
        "intercept_raw",
        dist.Normal(0.0, 1.0).expand([n_cities, n_variants - 1]).to_event(2),
    )
    intercepts_rel = int_loc[None, :] + int_scale[None, :] * int_raw
    intercepts = _prepend_reference(intercepts_rel)

    # Locus-specific hierarchical prior for dispersion predictor eta_rho.
    eta_loc = numpyro.sample("eta_loc", dist.Normal(-0.5, 1.0))
    eta_scale = numpyro.sample("eta_scale", dist.HalfNormal(1.0))
    eta_raw = numpyro.sample(
        "eta_raw", dist.Normal(0.0, 1.0).expand([n_loci]).to_event(1)
    )
    eta_rho = eta_loc + eta_scale * eta_raw

    growth_params = LinearGrowthParams(intercepts=intercepts, slopes=slopes)
    dispersion_params = DispersionParams(eta_rho=eta_rho)

    loglik = model_loglik(
        growth_params=growth_params,
        dispersion_params=dispersion_params,
        data=data,
    )
    numpyro.factor("obs_loglik", loglik)
