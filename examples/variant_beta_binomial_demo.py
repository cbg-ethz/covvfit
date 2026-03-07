"""Minimal end-to-end demo for variant beta-binomial model."""

import jax
import jax.numpy as jnp
from covvfit._variant_beta_binomial.growth import LinearGrowthParams
from covvfit._variant_beta_binomial.model import model_loglik
from covvfit._variant_beta_binomial.simulate import simulate_dataset


def main() -> None:
    key = jax.random.PRNGKey(0)

    C = 3
    G = 6
    N_samples = 60

    city = jnp.repeat(jnp.arange(C, dtype=jnp.int32), N_samples // C)
    time = jnp.tile(jnp.linspace(0.0, 10.0, N_samples // C), C)

    intercepts = jnp.array(
        [
            [0.0, 1.2, -0.6, 0.3],
            [0.3, 0.9, -0.2, -0.4],
            [-0.2, 0.7, 0.5, -0.1],
        ]
    )
    slopes = jnp.array([0.04, -0.02, 0.03, 0.01])
    growth = LinearGrowthParams(intercepts=intercepts, slopes=slopes)

    A = jnp.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1],
        ],
        dtype=bool,
    )

    coverage = jnp.full((N_samples, G), 100, dtype=jnp.int32)
    eta_rho = jnp.array([-0.2, 0.1, 0.0, -0.4, 0.2, -0.1])

    result = simulate_dataset(
        key=key,
        A=A,
        growth_params=growth,
        eta_rho=eta_rho,
        city=city,
        time=time,
        coverage=coverage,
        mask_probability=0.1,
    )

    ll = model_loglik(
        growth_params=result.growth_params,
        dispersion_params=result.dispersion_params,
        data=result.data,
    )

    print("log_pi.shape:", result.log_pi.shape)
    print("mu.shape:", result.mu.shape)
    print("Y.shape:", result.data.Y.shape)
    print("N.shape:", result.data.N.shape)
    print("total_loglik:", float(ll))


if __name__ == "__main__":
    main()
