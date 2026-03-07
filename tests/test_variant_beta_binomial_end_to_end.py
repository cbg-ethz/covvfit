import jax
import jax.numpy as jnp
from covvfit._variant_beta_binomial.growth import LinearGrowthParams
from covvfit._variant_beta_binomial.model import model_loglik
from covvfit._variant_beta_binomial.simulate import simulate_dataset


def _build_demo_inputs():
    key = jax.random.PRNGKey(7)
    C = 3
    G = 6
    N_samples = 60

    city = jnp.repeat(jnp.arange(C, dtype=jnp.int32), N_samples // C)
    time = jnp.tile(jnp.linspace(0.0, 10.0, N_samples // C), C)

    intercepts = jnp.array(
        [
            [0.0, 1.0, -0.4, 0.2],
            [0.2, 0.6, -0.1, -0.2],
            [-0.3, 0.8, 0.4, -0.1],
        ]
    )
    slopes = jnp.array([0.05, -0.03, 0.02, 0.01])
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
    coverage = jnp.full((N_samples, G), 80, dtype=jnp.int32)
    eta_rho = jnp.array([-0.2, 0.0, 0.3, -0.5, 0.1, 0.2])

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
    return result, eta_rho


def test_model_loglik_finite_scalar() -> None:
    result, eta_rho = _build_demo_inputs()

    ll = model_loglik(
        growth_params=result.growth_params,
        dispersion_params=eta_rho,
        data=result.data,
    )

    assert ll.shape == ()
    assert jnp.isfinite(ll)


def test_parameter_perturbation_changes_likelihood() -> None:
    result, eta_rho = _build_demo_inputs()

    ll1 = model_loglik(
        growth_params=result.growth_params,
        dispersion_params=eta_rho,
        data=result.data,
    )

    delta = jnp.zeros_like(result.growth_params.intercepts).at[:, 0].set(0.1)
    perturbed_growth = LinearGrowthParams(
        intercepts=result.growth_params.intercepts + delta,
        slopes=result.growth_params.slopes,
    )
    ll2 = model_loglik(
        growth_params=perturbed_growth,
        dispersion_params=eta_rho,
        data=result.data,
    )

    assert not jnp.isclose(ll1, ll2)
