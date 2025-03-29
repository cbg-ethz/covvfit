from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

import covvfit._quasimultinomial as qm


class JointLogisticGrowthParams(NamedTuple):
    """This is a model of logistic growth (selection dynamics)
    in `K` cities for `V` competing variants.

    We assume that the relative growth advantages
    do not change between the cities, however
    we allow different introduction times, resulting
    in different offsets in the logistic growth model.

    This model has `V-1` relative growth rate parameters
    and `K*(V-1)` offsets.

    Attrs:

        relative_growths: relative growth rates, shape `(V-1,)`
        relative_offsets: relative offsets, shape `(K, V-1)`
    """

    relative_growths: Float[Array, " variants-1"]
    relative_offsets: Float[Array, "cities variants-1"]

    @property
    def n_cities(self) -> int:
        """Number of cities."""
        return self.relative_offsets.shape[0]

    @property
    def n_variants(self) -> int:
        """Number of variants."""
        return 1 + self.relative_offsets.shape[1]

    @property
    def n_params(self) -> int:
        """Number of all parameters in the model."""
        return (self.n_variants - 1) * (self.n_cities + 1)

    @staticmethod
    def _predict_log_abundance_single(
        relative_growths: Float[Array, " variants-1"],
        relative_offsets: Float[Array, " variants-1"],
        timepoints: Float[Array, " timepoints"],
    ) -> Float[Array, "timepoints variants"]:
        return qm.calculate_logps(
            ts=timepoints,
            midpoints=qm._add_first_variant(relative_offsets),
            growths=qm._add_first_variant(relative_growths),
        )

    def predict_log_abundance(
        self,
        timepoints: Float[Array, "cities timepoints"],
    ) -> Float[Array, "cities timepoints variants"]:
        """Predicts the abundances at the specified time points."""
        _new_shape = (self.n_cities, self.n_variants - 1)
        tiled_growths = jnp.broadcast_to(self.relative_growths[None, :], _new_shape)

        return jax.vmap(self._predict_log_abundance_single, in_axes=0)(
            tiled_growths, self.relative_offsets, timepoints
        )

    @classmethod
    def from_vector(cls, theta, n_variants: int) -> "JointLogisticGrowthParams":
        """Wraps a vector with parameters of shape `(dim,)` to the model.
        Note that `dim` should match the number of parameters.
        """
        return JointLogisticGrowthParams(
            relative_growths=qm.get_relative_growths(theta, n_variants=n_variants),
            relative_offsets=qm.get_relative_midpoints(theta, n_variants=n_variants),
        )

    def to_vector(self) -> Float[Array, " *batch"]:
        """Wraps all the parameter into a single vector.

        Note:
            This function is useful for optimization purposes, as many optimizers accept vectors,
            rather than tuples.
        """
        return qm.construct_theta(
            relative_growths=self.relative_growths,
            relative_midpoints=self.relative_offsets,
        )


class LogisticGrowthWithGaussianProcess(NamedTuple):
    """The logistic growth (selection dynamics) adjusted
    by the Gaussian process (represented using Hilbert space approximation)
    for a single city.

    The model is the following:

    $y(t) = softmax(f(t))$

    where $f_0(t) = 0$ for all $t$ (typical constraint to remove aliasing)
    and

    $f_v(t) = a_v t + b + g_v(t)$,

    where $g_v(t)$ is a random function represented in terms of basis.
    """

    relative_growths: Float[Array, " variants-1"]
    relative_offsets: Float[Array, " variants-1"]

    n_basis: int
    domain_lengthscale: float

    gp_lengthscales: Float[Array, " variants-1"]
    gp_amplitudes: Float[Array, " variants-1"]

    gp_coefficients: Float[Array, "n_basis variants-1"]

    @property
    def n_variants(self) -> int:
        """Number of variants."""
        return 1 + self.relative_offsets.shape[-1]

    @property
    def _sqrt_eigenvalues(self):
        base = jnp.pi / (2 * self.domain_lengthscale)
        return base * jnp.arange(1, self.n_basis + 1)

    def _basis(
        self, timepoints: Float[Array, " timepoints"]
    ) -> Float[Array, "timepoints n_basis"]:
        raise NotImplementedError

    def _get_fv(
        basis: Float[Array, "timepoints n_basis"],
        v: int,
    ) -> Float[Array, " timepoints"]:
        pass

    def predict_latent(
        self,
        timepoints: Float[Array, " timepoints"],
    ) -> Float[Array, "timepoints variants"]:
        f0 = jnp.zeros_like(timepoints)

        basis = self._basis(timepoints)

        def aux(v: int):
            return self._get_fv(basis, v)

        fs = jax.vmap(aux)(jnp.arange(1, self.n_variants))

        print(f0, fs, "Not implemented")
        raise NotImplementedError

    def predict_log_abundance(
        self,
        timepoints: Float[Array, " timepoints"],
    ) -> Float[Array, "timepoints variants"]:
        latent = self.predict_latent(timepoints)
        return jax.nn.log_softmax(latent, axis=-1)
