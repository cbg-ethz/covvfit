"""Data containers for variant beta-binomial model."""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

# Axis-name sentinels for Ruff/Pyflakes handling of jaxtyping string annotations.
samples = loci = variants = features = shape = None


@dataclass(frozen=True)
class ModelData:
    """Observed data for the multi-city beta-binomial model."""

    Y: Int[Array, "samples loci"]
    N: Int[Array, "samples loci"]
    city: Int[Array, "samples"]
    time: Float[Array, "samples"]
    A: Bool[Array, "variants loci"]
    obs_mask: Bool[Array, "samples loci"]
    X: Float[Array, "samples features"] | None = None

    def __post_init__(self) -> None:
        Y = jnp.asarray(self.Y)
        N = jnp.asarray(self.N)
        city = jnp.asarray(self.city)
        time = jnp.asarray(self.time)
        A = jnp.asarray(self.A)
        obs_mask = jnp.asarray(self.obs_mask)

        object.__setattr__(self, "Y", Y)
        object.__setattr__(self, "N", N)
        object.__setattr__(self, "city", city)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "obs_mask", obs_mask)

        if self.X is not None:
            object.__setattr__(self, "X", jnp.asarray(self.X))

        if Y.ndim != 2 or N.ndim != 2:
            raise ValueError("Y and N must be 2D arrays with shape (samples, loci).")
        if Y.shape != N.shape:
            raise ValueError("Y and N must have identical shapes.")

        n_samples, n_loci = Y.shape

        if city.ndim != 1 or city.shape[0] != n_samples:
            raise ValueError("city must have shape (samples,).")
        if time.ndim != 1 or time.shape[0] != n_samples:
            raise ValueError("time must have shape (samples,).")
        if obs_mask.ndim != 2 or obs_mask.shape != (n_samples, n_loci):
            raise ValueError("obs_mask must have shape (samples, loci).")
        if A.ndim != 2 or A.shape[1] != n_loci:
            raise ValueError("A must have shape (variants, loci) with matching loci.")

        if not jnp.issubdtype(city.dtype, jnp.integer):
            raise ValueError("city must have integer dtype.")
        if obs_mask.dtype != jnp.bool_:
            raise ValueError("obs_mask must have boolean dtype.")

        if not jnp.all((A == 0) | (A == 1)):
            raise ValueError("A must be binary (values in {0, 1}).")
        if not jnp.all(jnp.sum(A, axis=0) > 0):
            raise ValueError("Every locus must be present in at least one variant.")

        if not jnp.issubdtype(Y.dtype, jnp.integer):
            raise ValueError("Y must have integer dtype.")
        if not jnp.issubdtype(N.dtype, jnp.integer):
            raise ValueError("N must have integer dtype.")
        if not jnp.all(Y >= 0):
            raise ValueError("Y must be non-negative.")
        if not jnp.all(N >= 0):
            raise ValueError("N must be non-negative.")
        if not jnp.all(Y <= N):
            raise ValueError("Y must satisfy Y <= N elementwise.")

        if self.X is not None:
            if self.X.ndim != 2 or self.X.shape[0] != n_samples:
                raise ValueError("X must have shape (samples, features).")


@dataclass(frozen=True)
class DispersionParams:
    """Unconstrained dispersion parameters for rho predictor."""

    eta_rho: Float[Array, "*shape"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "eta_rho", jnp.asarray(self.eta_rho, dtype=float))
        if self.eta_rho.ndim not in (1, 2):
            raise ValueError("eta_rho must have shape (loci,) or (cities, loci).")
