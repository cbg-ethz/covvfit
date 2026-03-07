"""End-to-end hierarchical Bayesian inference demo for variant beta-binomial model."""

import argparse

import jax
import jax.numpy as jnp
from covvfit._variant_beta_binomial.growth import LinearGrowthParams
from covvfit._variant_beta_binomial.numpyro_model import numpyro_model_hierarchical
from covvfit._variant_beta_binomial.simulate import simulate_dataset
from numpyro.infer import MCMC, NUTS


def run_demo(
    seed: int = 0,
    n_samples: int = 180,
    coverage_n: int = 300,
    time_max: float = 12.0,
    num_warmup: int = 700,
    num_posterior_samples: int = 700,
) -> None:
    key = jax.random.PRNGKey(seed)
    key_sim, key_mcmc = jax.random.split(key)

    C = 3
    G = 6
    city = jnp.repeat(jnp.arange(C, dtype=jnp.int32), n_samples // C)
    time = jnp.tile(jnp.linspace(0.0, time_max, n_samples // C), C)

    true_intercepts = jnp.array(
        [
            [0.0, 0.7, -0.2, 0.1],
            [0.0, 0.5, 0.1, -0.3],
            [0.0, 0.8, 0.3, -0.1],
        ],
        dtype=float,
    )
    true_slopes = jnp.array([0.0, 0.05, -0.02, 0.03], dtype=float)

    growth_true = LinearGrowthParams(intercepts=true_intercepts, slopes=true_slopes)

    A = jnp.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    coverage = jnp.full((n_samples, G), coverage_n, dtype=jnp.int32)
    true_eta_rho = jnp.array([-0.3, -0.1, 0.2, -0.4, 0.1, 0.0], dtype=float)

    sim = simulate_dataset(
        key=key_sim,
        A=A,
        growth_params=growth_true,
        eta_rho=true_eta_rho,
        city=city,
        time=time,
        coverage=coverage,
        mask_probability=0.1,
    )

    nuts = NUTS(numpyro_model_hierarchical, target_accept_prob=0.95)
    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_posterior_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(key_mcmc, data=sim.data, n_cities=C)
    mcmc.print_summary(exclude_deterministic=False)

    posterior = mcmc.get_samples()
    slope_loc = posterior["slope_loc"]
    slope_scale = posterior["slope_scale"]
    slope_raw = posterior["slope_raw"]

    slopes_rel = slope_loc[:, None] + slope_scale[:, None] * slope_raw
    slopes = jnp.concatenate([jnp.zeros((slopes_rel.shape[0], 1)), slopes_rel], axis=1)

    slopes_post_mean = jnp.mean(slopes, axis=0)
    slopes_q05 = jnp.quantile(slopes, 0.05, axis=0)
    slopes_q95 = jnp.quantile(slopes, 0.95, axis=0)
    slopes_cover = (true_slopes >= slopes_q05) & (true_slopes <= slopes_q95)

    print("\nRecovery summary (slopes):")
    print("True slopes:", true_slopes)
    print("Posterior mean slopes:", slopes_post_mean)
    print("Posterior 90% lower:", slopes_q05)
    print("Posterior 90% upper:", slopes_q95)
    print("Truth in 90% interval:", slopes_cover)
    print("Abs. error slopes:", jnp.abs(slopes_post_mean - true_slopes))
    eta_loc_post = posterior["eta_loc"]
    print(
        "Posterior eta_loc mean (true mean eta_rho):",
        float(jnp.mean(eta_loc_post)),
        "(",
        float(jnp.mean(true_eta_rho)),
        ")",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hierarchical Bayesian inference demo for variant beta-binomial model."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=180,
        help="Total number of simulated samples (should be divisible by number of cities).",
    )
    parser.add_argument(
        "--coverage",
        type=int,
        default=300,
        help="Per-sample per-locus sequencing coverage.",
    )
    parser.add_argument(
        "--time-max",
        type=float,
        default=12.0,
        help="Maximum simulation time.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=700,
        help="Number of NUTS warmup steps.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=700,
        help="Number of posterior samples.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_demo(
        seed=args.seed,
        n_samples=args.n_samples,
        coverage_n=args.coverage,
        time_max=args.time_max,
        num_warmup=args.warmup,
        num_posterior_samples=args.samples,
    )
