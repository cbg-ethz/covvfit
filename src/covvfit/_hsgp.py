"""Hilbert space approximations to Gaussian processes.

The main reference for the formulae here is the following: 

[GRM]: Gabriel Riutort-Mayol et al.
Practical Hilbert space approximate Bayesian Gaussian processes for
probabilistic programming
arXiv (2020)
URL: https://arxiv.org/abs/2004.11408

It is built on the methods proposed by:

[AS]: Arno Solin and Simo Särkkä.
Hilbert space methods for reduced-rank Gaussian process regression. 
Statistics and Computing (2019)
URL: https://link.springer.com/article/10.1007/s11222-019-09886-w

Another great resource is this blog post:

[JCO]: Juan Camilo Orduz.
A Conceptual and Practical Introduction to
Hilbert Space GPs Approximation Methods
URL: https://juanitorduz.github.io/hsgp_intro/
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class Interval(NamedTuple):
    lengthscale: float

    @property
    def bounds(self) -> tuple[float, float]:
        ell = self.lengthscale
        return (-ell, ell)

    @staticmethod
    def _range(n: int):
        return jnp.arange(1, n + 1, dtype=float)

    def sqrt_eigenvalues(self, n: int) -> Float[Array, " n"]:
        j = self._range(n)
        return 0.5 * j * jnp.pi / self.lengthscale

    def eigenvalues(self, n: int) -> Float[Array, " n"]:
        return jnp.square(self.sqrt_eigenvalues(n))

    def eigenfunctions(
        self, x: Float[Array, " batch"], n: int
    ) -> Float[Array, "n batch"]:
        L = self.lengthscale

        def psi_fn(sqrt_lambda: float):
            return jnp.sin(sqrt_lambda * (x + L)) / jnp.sqrt(L)

        return jax.vmap(psi_fn)(self.sqrt_eigenvalues(n))


class ISpectralKernel(eqx.Module):
    params: eqx.Module

    def evaluate_kernel(self, r: Float[Array, " *batch"]) -> Float[Array, " *batch"]:
        pass

    def gram_matrix(
        self, x: Float[Array, " nx"], y: Float[Array, " ny"]
    ) -> Float[Array, "nx ny"]:
        pass

    def spectral_density(
        self, omega: Float[Array, " *batch"]
    ) -> Float[Array, " *batch"]:
        pass


class BaseSpectralKernel(ISpectralKernel):
    def gram_matrix(
        self, x: Float[Array, " nx"], y: Float[Array, " ny"]
    ) -> Float[Array, "nx ny"]:
        r = jnp.abs(x[..., None] - y[None, ...])
        return self.evaluate_kernel(r)


class AmplitudeLengthParams(eqx.Module):
    amplitude: Float[Array, " "]
    lengthscale: Float[Array, " "]


class RBFKernel(BaseSpectralKernel):
    params: AmplitudeLengthParams

    def evaluate_kernel(self, r):
        # See the equation for k_infty in second column
        # of page 3 in [GRM]
        ampl = self.params.amplitude
        leng = self.params.lengthscale
        return jnp.square(ampl) * jnp.exp(-0.5 * jnp.square(r / leng))

    def spectral_density(self, omega):
        ampl = self.params.amplitude
        leng = self.params.lengthscale

        # Use Eq. (1) from [GRM]
        # noting that their alpha = our ampl**2.
        const = jnp.square(ampl) * jnp.sqrt(2 * jnp.pi) * leng
        exponent = jnp.exp(-0.5 * jnp.square(leng * omega))
        return const * exponent


class Matern12(BaseSpectralKernel):
    params: AmplitudeLengthParams

    def evaluate_kernel(self, r):
        ampl = self.params.amplitude
        leng = self.params.lengthscale
        x = r / leng
        return jnp.square(ampl) * jnp.exp(-x)

    def spectral_density(self, omega):
        # This one is not provided explicitly in [GRM]
        # but the formula just between Eq. (1) can be
        # used, setting nu = 1/2 and D = 1
        # Also, note that 4pi^2 factor from omega^2 is taken
        # away and that alpha = ampl^2
        ampl = self.params.amplitude
        leng = self.params.lengthscale

        # Use Eq. (2) from [GRM] with D = 1
        # noting that their alpha = our ampl**2 and
        # Gamma(4/2) = (2 - 1)! = 1
        alpha = jnp.square(ampl)
        return 2 * leng * alpha / (1 + jnp.square(omega * leng))


class Matern32(BaseSpectralKernel):
    params: AmplitudeLengthParams

    def evaluate_kernel(self, r):
        # See the equation for k_3/2 in second column
        # of page 3 in [GRM]
        ampl = self.params.amplitude
        leng = self.params.lengthscale
        sqrt3 = jnp.sqrt(3.0)
        x = r / leng
        return jnp.square(ampl) * (1 + sqrt3 * x) * jnp.exp(-sqrt3 * x)

    def spectral_density(self, omega):
        ampl = self.params.amplitude
        leng = self.params.lengthscale

        # Use Eq. (2) from [GRM] with D = 1
        # noting that their alpha = our ampl**2 and
        # Gamma(4/2) = (2 - 1)! = 1
        alpha = jnp.square(ampl)
        num = 12.0 * jnp.sqrt(3.0) * leng * alpha
        den = jnp.square(3.0 + jnp.square(leng * omega))
        return num / den


class Matern52(BaseSpectralKernel):
    params: AmplitudeLengthParams

    def evaluate_kernel(self, r):
        # See the equation for k_5/2 in second column
        # of page 3 in [GRM]
        ampl = self.params.amplitude
        leng = self.params.lengthscale
        sqrt5 = jnp.sqrt(5.0)

        x = r / leng

        bracket = 1.0 + (sqrt5 * x) + (5.0 / 3.0) * jnp.square(x)
        exponen = jnp.exp(-sqrt5 * x)
        return jnp.square(ampl) * bracket * exponen

    def spectral_density(self, omega):
        ampl = self.params.amplitude
        leng = self.params.lengthscale

        # Use Eq. (3) from [GRM] with D = 1
        # noting that their alpha = our ampl**2 and
        # Gamma(6/2) = (3 - 1)! = 2
        alpha = jnp.square(ampl)
        num = (400.0 / 3.0) * jnp.sqrt(5.0) * alpha * leng
        den = (5.0 + jnp.square(leng * omega)) ** 3

        return num / den


def approximate_gram_matrix(
    kernel: ISpectralKernel,
    x: Float[Array, " nx"],
    y: Float[Array, " ny"],
    n_basis: int,
    lengthscale: float,
) -> Float[Array, "nx ny"]:
    interval = Interval(lengthscale)

    sqrt_lambda = interval.sqrt_eigenvalues(n_basis)
    phi_x = interval.eigenfunctions(x, n=n_basis)  # Shape (n_basis, nx)
    phi_y = interval.eigenfunctions(y, n=n_basis)  # Shape (n_basis, ny)

    S = kernel.spectral_density(sqrt_lambda)  # shape (n_basis,)

    return jnp.einsum("n,nx,ny->xy", S, phi_x, phi_y)


def generate_approximated_prediction_function(
    x: Float[Array, " n_x"],
    n_basis: int,
    lengthscale: float,
):
    interval = Interval(lengthscale=lengthscale)
    eigenfunctions = interval.eigenfunctions(x, n=n_basis)
    sqrt_eigenvalues = interval.sqrt_eigenvalues(n_basis)

    def predict(
        kernel: ISpectralKernel,
        coefficients: Float[Array, " n_basis"],
    ):
        S = kernel.spectral_density(sqrt_eigenvalues)
        root_S = jnp.sqrt(S)  # (n_basis,)
        return jnp.einsum("n,n,nx->x", root_S, coefficients, eigenfunctions)

    return predict
