"""End-to-end hierarchical Bayesian inference demo for variant beta-binomial model."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.optim as optim
from covvfit._variant_beta_binomial.dispersion import (
    rho_to_log_kappa,
    transform_eta_to_rho,
)
from covvfit._variant_beta_binomial.growth import LinearGrowthParams
from covvfit._variant_beta_binomial.numpyro_model import numpyro_model_hierarchical
from covvfit._variant_beta_binomial.simulate import simulate_dataset
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
    AutoIAFNormal,
    AutoLowRankMultivariateNormal,
    AutoNormal,
)

SviGuideName = Literal["lowrank", "autonormal", "iaf", "bnaf"]
ScenarioName = Literal["baseline", "dramatic_emergence"]


def _prepend_reference(x_rel: jax.Array) -> jax.Array:
    if x_rel.ndim == 2:
        return jnp.concatenate([jnp.zeros((x_rel.shape[0], 1)), x_rel], axis=1)
    if x_rel.ndim == 3:
        return jnp.concatenate(
            [jnp.zeros((x_rel.shape[0], x_rel.shape[1], 1)), x_rel], axis=2
        )
    raise ValueError("Expected rank-2 or rank-3 array.")


def _posterior_growth_samples(
    posterior: dict[str, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """Return posterior samples of slopes and city intercepts with reference variant prepended."""
    slope_rel = (
        posterior["slope_loc"][:, None]
        + posterior["slope_scale"][:, None] * posterior["slope_raw"]
    )
    slopes = _prepend_reference(slope_rel)

    int_rel = (
        posterior["intercept_loc"][:, None, :]
        + posterior["intercept_scale"][:, None, :] * posterior["intercept_raw"]
    )
    intercepts = _prepend_reference(int_rel)
    return slopes, intercepts


def _predict_pi_samples(
    slopes: jax.Array, intercepts_city: jax.Array, t_grid: jax.Array
) -> jax.Array:
    """Predict abundance probabilities for posterior growth samples in one city."""
    logits = intercepts_city[:, None, :] + slopes[:, None, :] * t_grid[None, :, None]
    return jax.nn.softmax(logits, axis=-1)


def _simulate_observed_frequency_samples(
    key: jax.Array,
    mu_post: jax.Array,
    eta_rho_post: jax.Array,
    predictive_coverage: int,
    predictive_draws: int | None = None,
) -> jax.Array:
    """Simulate posterior predictive observed mutation frequencies Y/N.

    Args:
        key: PRNG key.
        mu_post: latent mutation frequencies, shape (samples, cities, time, loci).
        eta_rho_post: posterior eta_rho samples, shape (samples, loci).
        predictive_coverage: total read count N used for predictive simulation.
        predictive_draws: number of posterior predictive draws. If larger than
            available posterior samples, posterior states are sampled with replacement.
    """
    n_post = mu_post.shape[0]
    if predictive_draws is None:
        predictive_draws = n_post

    if predictive_draws != n_post:
        key, key_subsample = jax.random.split(key)
        idx = jax.random.randint(
            key_subsample, shape=(predictive_draws,), minval=0, maxval=n_post
        )
        mu_post = mu_post[idx]
        eta_rho_post = eta_rho_post[idx]

    eps_mu = 1e-6
    mu_clip = jnp.clip(mu_post, eps_mu, 1.0 - eps_mu)

    rho = transform_eta_to_rho(eta_rho_post)  # (samples, loci)
    kappa = jnp.exp(rho_to_log_kappa(rho))  # (samples, loci)
    kappa = kappa[:, None, None, :]  # broadcast to (samples, 1, 1, loci)

    alpha = mu_clip * kappa
    beta = (1.0 - mu_clip) * kappa

    key_beta, key_binom = jax.random.split(key)
    p = jax.random.beta(key_beta, a=alpha, b=beta, shape=alpha.shape)
    y = jax.random.binomial(
        key_binom,
        n=jnp.asarray(predictive_coverage, dtype=float),
        p=p,
        shape=p.shape,
    )
    return y / float(predictive_coverage)


def _plot_abundance_curves(
    output_path: Path,
    t_grid: jax.Array,
    pi_truth: jax.Array,
    pi_post: jax.Array,
) -> plt.Figure:
    """Plot posterior abundance curves and uncertainty bands against truth for each city."""
    n_cities = pi_truth.shape[0]
    n_variants = pi_truth.shape[2]
    colors = [f"C{i}" for i in range(n_variants)]

    fig, axes = plt.subplots(
        nrows=n_cities, ncols=1, figsize=(10, 3.2 * n_cities), sharex=True
    )
    if n_cities == 1:
        axes = [axes]

    for c in range(n_cities):
        ax = axes[c]
        city_post = pi_post[:, c, :, :]  # (samples, time, variants)
        city_truth = pi_truth[c, :, :]
        q05 = jnp.quantile(city_post, 0.05, axis=0)
        q50 = jnp.quantile(city_post, 0.50, axis=0)
        q95 = jnp.quantile(city_post, 0.95, axis=0)

        for v in range(n_variants):
            ax.fill_between(t_grid, q05[:, v], q95[:, v], color=colors[v], alpha=0.2)
            ax.plot(
                t_grid,
                q50[:, v],
                color=colors[v],
                linewidth=2,
                label=f"Variant {v} posterior",
            )
            ax.plot(
                t_grid,
                city_truth[:, v],
                color=colors[v],
                linestyle="--",
                linewidth=1.5,
                label=f"Variant {v} truth",
            )

        ax.set_title(f"City {c}: variant abundance trajectories")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Abundance")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[: 2 * n_variants],
        labels[: 2 * n_variants],
        loc="upper center",
        ncols=4,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    return fig


def _plot_mutation_predictions(
    output_path: Path,
    t_grid: jax.Array,
    mu_truth: jax.Array,
    mu_post: jax.Array,
    obs_freq_post: jax.Array,
    sim_data,
) -> plt.Figure:
    """Plot observed mutation frequencies with latent and observed predictive bands by city/locus."""
    n_cities = mu_truth.shape[0]
    n_loci = mu_truth.shape[2]

    fig, axes = plt.subplots(
        nrows=n_cities,
        ncols=n_loci,
        figsize=(3.2 * n_loci, 2.5 * n_cities),
        sharex=True,
        sharey=True,
    )

    if n_cities == 1:
        axes = axes[None, :]
    if n_loci == 1:
        axes = axes[:, None]

    for c in range(n_cities):
        city_indices = jnp.where(sim_data.city == c)[0]
        city_times = sim_data.time[city_indices]
        order = jnp.argsort(city_times)
        city_indices = city_indices[order]
        city_times = city_times[order]

        y_city = sim_data.Y[city_indices]
        n_city = sim_data.N[city_indices]
        mask_city = sim_data.obs_mask[city_indices]
        obs_freq_city = y_city / n_city

        city_mu_post = mu_post[:, c, :, :]  # (samples, time, loci)
        city_obs_post = obs_freq_post[:, c, :, :]  # (samples, time, loci)
        city_mu_truth = mu_truth[c, :, :]
        q05 = jnp.quantile(city_mu_post, 0.05, axis=0)
        q50 = jnp.quantile(city_mu_post, 0.50, axis=0)
        q95 = jnp.quantile(city_mu_post, 0.95, axis=0)
        oq05 = jnp.quantile(city_obs_post, 0.05, axis=0)
        oq50 = jnp.quantile(city_obs_post, 0.50, axis=0)
        oq95 = jnp.quantile(city_obs_post, 0.95, axis=0)

        for g in range(n_loci):
            ax = axes[c, g]
            # Latent mutation-frequency posterior (mu)
            ax.fill_between(t_grid, q05[:, g], q95[:, g], color="C0", alpha=0.20)
            ax.plot(t_grid, q50[:, g], color="C0", linewidth=1.8)
            # Posterior predictive observed-frequency band (Y/N)
            ax.fill_between(t_grid, oq05[:, g], oq95[:, g], color="C1", alpha=0.18)
            ax.plot(t_grid, oq50[:, g], color="C1", linewidth=1.5)
            ax.plot(
                t_grid,
                city_mu_truth[:, g],
                color="black",
                linestyle="--",
                linewidth=1.5,
            )

            obs_mask = mask_city[:, g]
            ax.scatter(
                city_times[obs_mask],
                obs_freq_city[:, g][obs_mask],
                s=16,
                color="C3",
                alpha=0.8,
            )

            if c == 0:
                ax.set_title(f"Locus {g}")
            if g == 0:
                ax.set_ylabel(f"City {c}\nFrequency")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.2)

    for g in range(n_loci):
        axes[-1, g].set_xlabel("Time")

    fig.suptitle(
        "Mutation frequencies: latent vs observed predictive uncertainty", y=1.01
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    return fig


@dataclass(frozen=True)
class InferenceResult:
    method: str
    posterior: dict[str, jax.Array]


@dataclass(frozen=True)
class ScenarioConfig:
    name: ScenarioName
    description: str
    n_cities: int
    true_intercepts: jax.Array
    true_slopes: jax.Array
    A: jax.Array
    true_eta_rho: jax.Array
    mask_probability: float


def _make_scenario_config(scenario: ScenarioName) -> ScenarioConfig:
    if scenario == "baseline":
        return ScenarioConfig(
            name="baseline",
            description=(
                "Moderate growth with all variants present over time; useful for"
                " standard recovery checks."
            ),
            n_cities=3,
            true_intercepts=jnp.array(
                [
                    [0.0, 0.7, -0.2, 0.1],
                    [0.0, 0.5, 0.1, -0.3],
                    [0.0, 0.8, 0.3, -0.1],
                ],
                dtype=float,
            ),
            true_slopes=jnp.array([0.0, 0.05, -0.02, 0.03], dtype=float),
            A=jnp.array(
                [
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 1, 0, 0, 1],
                    [1, 1, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1],
                ],
                dtype=bool,
            ),
            true_eta_rho=jnp.array([-0.3, -0.1, 0.2, -0.4, 0.1, 0.0], dtype=float),
            mask_probability=0.1,
        )

    if scenario == "dramatic_emergence":
        return ScenarioConfig(
            name="dramatic_emergence",
            description=(
                "Three-variant emergence scenario: variant 0 dominates early, then"
                " variants 1 and 2 emerge; variant 2 becomes the final dominant"
                " lineage."
            ),
            n_cities=3,
            true_intercepts=jnp.array(
                [
                    [0.0, -7.0, -9.0],
                    [0.0, -7.5, -9.5],
                    [0.0, -6.8, -8.8],
                ],
                dtype=float,
            ),
            true_slopes=jnp.array([0.0, 0.90, 1.20], dtype=float),
            A=jnp.array(
                [
                    [1, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1],
                ],
                dtype=bool,
            ),
            true_eta_rho=jnp.array([-0.35, -0.15, 0.0, 0.05, 0.2], dtype=float),
            mask_probability=0.08,
        )

    raise ValueError(f"Unsupported scenario: {scenario}")


def _run_hmc_inference(
    key: jax.Array,
    data,
    n_cities: int,
    num_warmup: int,
    num_posterior_samples: int,
    target_accept_prob: float,
) -> InferenceResult:
    nuts = NUTS(numpyro_model_hierarchical, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_posterior_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(key, data=data, n_cities=n_cities)
    mcmc.print_summary(exclude_deterministic=False)
    extra = mcmc.get_extra_fields()
    if "diverging" in extra:
        print(f"Number of divergences: {int(jnp.sum(extra['diverging']))}")
    return InferenceResult(method="hmc", posterior=mcmc.get_samples())


def _run_svi_inference(
    key: jax.Array,
    data,
    n_cities: int,
    num_posterior_samples: int,
    svi_steps: int,
    svi_lr: float,
    svi_guide: SviGuideName,
    svi_rank: int,
    svi_num_flows: int,
    svi_iaf_hidden_dims: tuple[int, ...],
    svi_bnaf_hidden_factors: tuple[int, ...],
    svi_restarts: int,
) -> InferenceResult:
    if svi_restarts < 1:
        raise ValueError("svi_restarts must be >= 1.")

    def _make_guide():
        if svi_guide == "lowrank":
            return AutoLowRankMultivariateNormal(
                numpyro_model_hierarchical, rank=svi_rank
            )
        if svi_guide == "autonormal":
            return AutoNormal(numpyro_model_hierarchical)
        if svi_guide == "iaf":
            return AutoIAFNormal(
                numpyro_model_hierarchical,
                num_flows=svi_num_flows,
                hidden_dims=svi_iaf_hidden_dims,
            )
        if svi_guide == "bnaf":
            return AutoBNAFNormal(
                numpyro_model_hierarchical,
                num_flows=svi_num_flows,
                hidden_factors=list(svi_bnaf_hidden_factors),
            )
        raise ValueError(f"Unsupported SVI guide: {svi_guide}")

    best_result = None
    best_guide = None
    best_loss = None
    best_restart = None

    for restart in range(svi_restarts):
        key, key_restart = jax.random.split(key)
        guide = _make_guide()
        svi = SVI(
            numpyro_model_hierarchical,
            guide,
            optim.Adam(svi_lr),
            Trace_ELBO(),
        )
        try:
            svi_result = svi.run(
                key_restart,
                svi_steps,
                data=data,
                n_cities=n_cities,
                progress_bar=False,
            )
        except ValueError as exc:
            if (
                svi_guide == "iaf"
                and "Hidden dimension must not be less than input dimension" in str(exc)
            ):
                raise ValueError(
                    "AutoIAFNormal requires every --svi-iaf-hidden-dims entry to be >= latent dimension. "
                    "Increase hidden dims (for example: --svi-iaf-hidden-dims 128,128)."
                ) from exc
            raise
        final_loss = float(jnp.asarray(svi_result.losses)[-1])
        if not jnp.isfinite(final_loss):
            print(
                f"  SVI restart {restart + 1}/{svi_restarts}: final_loss is non-finite ({final_loss}); skipping."
            )
            continue
        print(
            f"  SVI restart {restart + 1}/{svi_restarts}: final_loss={final_loss:.4f}"
        )

        if best_loss is None or final_loss < best_loss:
            best_loss = final_loss
            best_result = svi_result
            best_guide = guide
            best_restart = restart + 1

    if best_result is None or best_guide is None or best_loss is None:
        raise ValueError(
            "All SVI restarts produced non-finite final losses. "
            "Try lowering --svi-lr, increasing --svi-steps/restarts, "
            "or using a more stable guide (for example --svi-guide bnaf or lowrank)."
        )

    post_key = jax.random.fold_in(key, 1)
    posterior = best_guide.sample_posterior(
        post_key,
        best_result.params,
        sample_shape=(num_posterior_samples,),
        data=data,
        n_cities=n_cities,
    )
    losses = jnp.asarray(best_result.losses)
    print("\nSVI summary:")
    print("  guide:", svi_guide)
    if svi_guide == "lowrank":
        print("  guide_rank:", svi_rank)
    if svi_guide in ("iaf", "bnaf"):
        print("  num_flows:", svi_num_flows)
    if svi_guide == "iaf":
        print("  iaf_hidden_dims:", svi_iaf_hidden_dims)
    if svi_guide == "bnaf":
        print("  bnaf_hidden_factors:", svi_bnaf_hidden_factors)
    print("  restarts:", svi_restarts)
    print("  selected_restart:", best_restart)
    print("  steps:", svi_steps)
    print("  learning_rate:", svi_lr)
    print("  initial_loss:", float(losses[0]))
    print("  final_loss:", float(losses[-1]))
    return InferenceResult(method="svi", posterior=posterior)


def _run_inference(
    method: Literal["hmc", "svi"],
    key: jax.Array,
    data,
    n_cities: int,
    num_warmup: int,
    num_posterior_samples: int,
    target_accept_prob: float,
    svi_steps: int,
    svi_lr: float,
    svi_guide: SviGuideName,
    svi_rank: int,
    svi_num_flows: int,
    svi_iaf_hidden_dims: tuple[int, ...],
    svi_bnaf_hidden_factors: tuple[int, ...],
    svi_restarts: int,
) -> InferenceResult:
    if method == "hmc":
        return _run_hmc_inference(
            key=key,
            data=data,
            n_cities=n_cities,
            num_warmup=num_warmup,
            num_posterior_samples=num_posterior_samples,
            target_accept_prob=target_accept_prob,
        )
    if method == "svi":
        return _run_svi_inference(
            key=key,
            data=data,
            n_cities=n_cities,
            num_posterior_samples=num_posterior_samples,
            svi_steps=svi_steps,
            svi_lr=svi_lr,
            svi_guide=svi_guide,
            svi_rank=svi_rank,
            svi_num_flows=svi_num_flows,
            svi_iaf_hidden_dims=svi_iaf_hidden_dims,
            svi_bnaf_hidden_factors=svi_bnaf_hidden_factors,
            svi_restarts=svi_restarts,
        )
    raise ValueError(f"Unsupported inference method: {method}")


def run_demo(
    seed: int = 0,
    scenario: ScenarioName = "baseline",
    n_samples: int = 180,
    coverage_n: int = 300,
    time_max: float = 12.0,
    inference_method: Literal["hmc", "svi"] = "hmc",
    num_warmup: int = 700,
    num_posterior_samples: int = 700,
    target_accept_prob: float = 0.95,
    svi_steps: int = 5000,
    svi_lr: float = 1e-2,
    svi_guide: SviGuideName = "lowrank",
    svi_rank: int = 8,
    svi_num_flows: int = 2,
    svi_iaf_hidden_dims: tuple[int, ...] = (128, 128),
    svi_bnaf_hidden_factors: tuple[int, ...] = (8, 8),
    svi_restarts: int = 3,
    output_dir: str = "generated/variant_beta_binomial_demo",
    n_grid: int = 200,
    predictive_coverage: int | None = None,
    predictive_draws: int = 4000,
    show_plots: bool = False,
) -> None:
    key = jax.random.PRNGKey(seed)
    key_sim, key_infer = jax.random.split(key)

    config = _make_scenario_config(scenario)
    C = config.n_cities
    if n_samples % C != 0:
        raise ValueError(
            f"n_samples ({n_samples}) must be divisible by number of cities ({C}) for scenario '{scenario}'."
        )
    samples_per_city = n_samples // C
    city = jnp.repeat(jnp.arange(C, dtype=jnp.int32), samples_per_city)
    time = jnp.tile(jnp.linspace(0.0, time_max, samples_per_city), C)

    true_intercepts = config.true_intercepts
    true_slopes = config.true_slopes
    A = config.A
    G = A.shape[1]
    growth_true = LinearGrowthParams(intercepts=true_intercepts, slopes=true_slopes)
    coverage = jnp.full((n_samples, G), coverage_n, dtype=jnp.int32)
    true_eta_rho = config.true_eta_rho

    pi_start = jax.nn.softmax(true_intercepts[0] + true_slopes * 0.0, axis=-1)
    pi_end = jax.nn.softmax(true_intercepts[0] + true_slopes * time_max, axis=-1)
    print(f"Scenario: {config.name}")
    print(config.description)
    print("City 0 variant prevalence at t=0:", pi_start)
    print(f"City 0 variant prevalence at t={time_max}:", pi_end)

    sim = simulate_dataset(
        key=key_sim,
        A=A,
        growth_params=growth_true,
        eta_rho=true_eta_rho,
        city=city,
        time=time,
        coverage=coverage,
        mask_probability=config.mask_probability,
    )

    inference_result = _run_inference(
        method=inference_method,
        key=key_infer,
        data=sim.data,
        n_cities=C,
        num_warmup=num_warmup,
        num_posterior_samples=num_posterior_samples,
        target_accept_prob=target_accept_prob,
        svi_steps=svi_steps,
        svi_lr=svi_lr,
        svi_guide=svi_guide,
        svi_rank=svi_rank,
        svi_num_flows=svi_num_flows,
        svi_iaf_hidden_dims=svi_iaf_hidden_dims,
        svi_bnaf_hidden_factors=svi_bnaf_hidden_factors,
        svi_restarts=svi_restarts,
    )
    posterior = inference_result.posterior
    print(f"\nInference method: {inference_result.method.upper()}")
    slopes, intercepts = _posterior_growth_samples(posterior)
    eta_rho_post = (
        posterior["eta_loc"][:, None]
        + posterior["eta_scale"][:, None] * posterior["eta_raw"]
    )

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

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_grid = jnp.linspace(0.0, time_max, n_grid)
    a_float = A.astype(float)

    pi_truth = []
    mu_truth = []
    pi_post = []
    mu_post = []
    for c in range(C):
        logits_truth = (
            true_intercepts[c][None, :] + true_slopes[None, :] * t_grid[:, None]
        )
        pi_city_truth = jax.nn.softmax(logits_truth, axis=-1)
        mu_city_truth = pi_city_truth @ a_float

        pi_city_post = _predict_pi_samples(
            slopes=slopes, intercepts_city=intercepts[:, c, :], t_grid=t_grid
        )
        mu_city_post = jnp.einsum("stv,vg->stg", pi_city_post, a_float)

        pi_truth.append(pi_city_truth)
        mu_truth.append(mu_city_truth)
        pi_post.append(pi_city_post)
        mu_post.append(mu_city_post)

    pi_truth = jnp.stack(pi_truth, axis=0)
    mu_truth = jnp.stack(mu_truth, axis=0)
    pi_post = jnp.stack(pi_post, axis=1)
    mu_post = jnp.stack(mu_post, axis=1)

    if predictive_coverage is None:
        predictive_coverage = coverage_n
    key_pred = jax.random.fold_in(key, 97)
    obs_freq_post = _simulate_observed_frequency_samples(
        key=key_pred,
        mu_post=mu_post,
        eta_rho_post=eta_rho_post,
        predictive_coverage=predictive_coverage,
        predictive_draws=predictive_draws,
    )

    abundance_path = out_dir / "abundance_curves_by_city.png"
    mutation_path = out_dir / "mutation_predictions_by_city_locus.png"

    fig1 = _plot_abundance_curves(
        output_path=abundance_path,
        t_grid=t_grid,
        pi_truth=pi_truth,
        pi_post=pi_post,
    )
    fig2 = _plot_mutation_predictions(
        output_path=mutation_path,
        t_grid=t_grid,
        mu_truth=mu_truth,
        mu_post=mu_post,
        obs_freq_post=obs_freq_post,
        sim_data=sim.data,
    )
    print("\nSaved plots:")
    print("-", abundance_path)
    print("-", mutation_path)
    if show_plots:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)


def _parse_args() -> argparse.Namespace:
    def _parse_int_list(value: str) -> tuple[int, ...]:
        items = [item.strip() for item in value.split(",") if item.strip()]
        if not items:
            raise argparse.ArgumentTypeError("Expected a comma-separated list of ints.")
        try:
            values = tuple(int(item) for item in items)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Expected a comma-separated list of ints."
            ) from exc
        if any(v < 1 for v in values):
            raise argparse.ArgumentTypeError("All values must be >= 1.")
        return values

    parser = argparse.ArgumentParser(
        description="Run hierarchical Bayesian inference demo for variant beta-binomial model."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["baseline", "dramatic_emergence"],
        default="baseline",
        help="Simulation preset. Use dramatic_emergence for sharp takeover dynamics.",
    )
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
        "--inference-method",
        type=str,
        choices=["hmc", "svi"],
        default="hmc",
        help="Inference backend to use.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=700,
        help="Number of NUTS warmup steps (HMC only).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=700,
        help="Number of posterior samples to draw (HMC or guide posterior samples for SVI).",
    )
    parser.add_argument(
        "--target-accept-prob",
        type=float,
        default=0.95,
        help="NUTS target acceptance probability (HMC only).",
    )
    parser.add_argument(
        "--svi-steps",
        type=int,
        default=5000,
        help="Number of optimization steps for SVI.",
    )
    parser.add_argument(
        "--svi-lr",
        type=float,
        default=1e-2,
        help="Learning rate for SVI optimizer.",
    )
    parser.add_argument(
        "--svi-guide",
        type=str,
        choices=["lowrank", "autonormal", "iaf", "bnaf"],
        default="lowrank",
        help="SVI guide family (NumPyro autoguides only; iaf is more fragile).",
    )
    parser.add_argument(
        "--svi-rank",
        type=int,
        default=8,
        help="Rank for low-rank SVI guide (used when --svi-guide=lowrank).",
    )
    parser.add_argument(
        "--svi-num-flows",
        type=int,
        default=2,
        help="Number of flow transforms (used with --svi-guide=iaf/bnaf).",
    )
    parser.add_argument(
        "--svi-iaf-hidden-dims",
        type=_parse_int_list,
        default=(128, 128),
        help="Comma-separated hidden dimensions for AutoIAFNormal (e.g. 128,128).",
    )
    parser.add_argument(
        "--svi-bnaf-hidden-factors",
        type=_parse_int_list,
        default=(8, 8),
        help="Comma-separated hidden factors for AutoBNAFNormal (e.g. 8,8).",
    )
    parser.add_argument(
        "--svi-restarts",
        type=int,
        default=3,
        help="Number of SVI restarts; best final ELBO is selected.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated/variant_beta_binomial_demo",
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=200,
        help="Number of time grid points for predictive plotting.",
    )
    parser.add_argument(
        "--predictive-coverage",
        type=int,
        default=None,
        help="Coverage N used to simulate observed-frequency predictive bands (default: --coverage).",
    )
    parser.add_argument(
        "--predictive-draws",
        type=int,
        default=4000,
        help="Number of posterior predictive draws used for observed-frequency bands.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively after saving.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_demo(
        seed=args.seed,
        scenario=args.scenario,
        n_samples=args.n_samples,
        coverage_n=args.coverage,
        time_max=args.time_max,
        inference_method=args.inference_method,
        num_warmup=args.warmup,
        num_posterior_samples=args.samples,
        target_accept_prob=args.target_accept_prob,
        svi_steps=args.svi_steps,
        svi_lr=args.svi_lr,
        svi_guide=args.svi_guide,
        svi_rank=args.svi_rank,
        svi_num_flows=args.svi_num_flows,
        svi_iaf_hidden_dims=args.svi_iaf_hidden_dims,
        svi_bnaf_hidden_factors=args.svi_bnaf_hidden_factors,
        svi_restarts=args.svi_restarts,
        output_dir=args.output_dir,
        n_grid=args.n_grid,
        predictive_coverage=args.predictive_coverage,
        predictive_draws=args.predictive_draws,
        show_plots=args.show_plots,
    )
