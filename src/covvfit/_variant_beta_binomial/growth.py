"""Growth utilities for variant beta-binomial model."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

# Axis-name sentinels for Ruff/Pyflakes handling of jaxtyping string annotations.
variants = features = samples = None


@dataclass(frozen=True)
class LinearGrowthParams:
    """Simple demo growth parameters.

    intercepts has shape (cities, variants), slopes has shape (variants,).
    """

    intercepts: Float[Array, "cities variants"]
    slopes: Float[Array, "variants"]

    def __post_init__(self) -> None:
        intercepts = jnp.asarray(self.intercepts, dtype=float)
        slopes = jnp.asarray(self.slopes, dtype=float)

        object.__setattr__(self, "intercepts", intercepts)
        object.__setattr__(self, "slopes", slopes)

        if intercepts.ndim != 2:
            raise ValueError("intercepts must have shape (cities, variants).")
        if slopes.ndim != 1:
            raise ValueError("slopes must have shape (variants,).")
        if intercepts.shape[1] != slopes.shape[0]:
            raise ValueError("intercepts and slopes must share the variants dimension.")


@dataclass(frozen=True)
class LinearGrowthThetaEff:
    """Effective growth parameters for one sample."""

    intercepts: Float[Array, "variants"]
    slopes: Float[Array, "variants"]


def effective_theta(
    theta: LinearGrowthParams,
    city_index: Int[Array, ""] | int,
    x_row: Float[Array, "features"] | None = None,
) -> LinearGrowthThetaEff:
    """Return sample-specific effective growth parameters.

    `x_row` is currently unused and reserved for future covariate terms.
    """
    del x_row
    return LinearGrowthThetaEff(
        intercepts=theta.intercepts[city_index],
        slopes=theta.slopes,
    )


def log_f(
    t: Float[Array, ""] | float, theta_eff: LinearGrowthThetaEff
) -> Float[Array, " variants"]:
    """Compute unnormalized log-prevalence scores for one sample."""
    return theta_eff.intercepts + theta_eff.slopes * t


def compute_log_pi(
    theta: LinearGrowthParams,
    city: Int[Array, "samples"],
    time: Float[Array, "samples"],
    X: Float[Array, "samples features"] | None = None,
) -> Float[Array, "samples variants"]:
    """Compute sample-wise log-prevalence scores."""

    city = jnp.asarray(city)
    time = jnp.asarray(time, dtype=float)

    if city.ndim != 1 or time.ndim != 1 or city.shape[0] != time.shape[0]:
        raise ValueError("city and time must have shape (samples,).")

    if X is None:

        def _sample_log_pi(c, t):
            theta_eff = effective_theta(theta, c, None)
            return log_f(t, theta_eff)

        return jax.vmap(_sample_log_pi)(city, time)

    X = jnp.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != city.shape[0]:
        raise ValueError("X must have shape (samples, features).")

    def _sample_log_pi(c, t, x):
        theta_eff = effective_theta(theta, c, x)
        return log_f(t, theta_eff)

    return jax.vmap(_sample_log_pi)(city, time, X)
