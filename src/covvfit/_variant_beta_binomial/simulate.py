"""Simulation utilities for variant beta-binomial model."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from covvfit._variant_beta_binomial.data import DispersionParams, ModelData
from covvfit._variant_beta_binomial.dispersion import transform_eta_to_rho
from covvfit._variant_beta_binomial.growth import LinearGrowthParams, compute_log_pi
from covvfit._variant_beta_binomial.likelihood import make_beta_binomial_shapes
from covvfit._variant_beta_binomial.mapping import compute_mu

# Axis-name sentinels for Ruff/Pyflakes handling of jaxtyping string annotations.
samples = variants = loci = features = shape = None


@dataclass(frozen=True)
class SimulationResult:
    """Container for simulated dataset and latent quantities."""

    data: ModelData
    growth_params: LinearGrowthParams
    dispersion_params: DispersionParams
    log_pi: Float[Array, "samples variants"]
    mu: Float[Array, "samples loci"]


def simulate_log_pi(
    growth_params: LinearGrowthParams,
    city: Int[Array, "samples"],
    time: Float[Array, "samples"],
    X: Float[Array, "samples features"] | None = None,
) -> Float[Array, "samples variants"]:
    """Simulate sample-wise log-prevalence scores."""
    return compute_log_pi(theta=growth_params, city=city, time=time, X=X)


def simulate_mu(
    log_pi: Float[Array, "samples variants"],
    A: Bool[Array, "variants loci"],
) -> Float[Array, "samples loci"]:
    """Simulate expected mutation frequencies."""
    return compute_mu(log_pi=log_pi, A=A)


def sample_beta_binomial_counts(
    key: jax.Array,
    N: Int[Array, "samples loci"],
    alpha: Float[Array, "samples loci"],
    beta: Float[Array, "samples loci"],
) -> Int[Array, "samples loci"]:
    """Sample counts from beta-binomial via beta then binomial draws."""
    N = jnp.asarray(N)
    alpha = jnp.asarray(alpha, dtype=float)
    beta = jnp.asarray(beta, dtype=float)

    key_beta, key_binom = jax.random.split(key)
    p = jax.random.beta(key_beta, a=alpha, b=beta, shape=alpha.shape)
    Y = jax.random.binomial(key_binom, n=N.astype(float), p=p, shape=N.shape)
    return Y.astype(jnp.int32)


def simulate_dataset(
    key: jax.Array,
    A: Bool[Array, "variants loci"],
    growth_params: LinearGrowthParams,
    eta_rho: Float[Array, "*shape"],
    city: Int[Array, "samples"],
    time: Float[Array, "samples"],
    coverage: Int[Array, "samples loci"],
    X: Float[Array, "samples features"] | None = None,
    mask_probability: float = 0.1,
) -> SimulationResult:
    """Simulate full beta-binomial dataset."""
    key_counts, key_mask = jax.random.split(key)

    log_pi = simulate_log_pi(growth_params=growth_params, city=city, time=time, X=X)
    mu = simulate_mu(log_pi=log_pi, A=A)

    rho = transform_eta_to_rho(eta_rho)
    alpha, beta = make_beta_binomial_shapes(mu=mu, rho=rho, city=city)

    coverage = jnp.asarray(coverage, dtype=jnp.int32)
    Y = sample_beta_binomial_counts(
        key=key_counts,
        N=coverage,
        alpha=alpha,
        beta=beta,
    )

    obs_mask = jax.random.uniform(key_mask, shape=coverage.shape) > mask_probability

    data = ModelData(
        Y=Y,
        N=coverage,
        city=jnp.asarray(city, dtype=jnp.int32),
        time=jnp.asarray(time, dtype=float),
        A=jnp.asarray(A, dtype=bool),
        obs_mask=jnp.asarray(obs_mask, dtype=bool),
        X=X,
    )

    return SimulationResult(
        data=data,
        growth_params=growth_params,
        dispersion_params=DispersionParams(eta_rho=jnp.asarray(eta_rho, dtype=float)),
        log_pi=log_pi,
        mu=mu,
    )
