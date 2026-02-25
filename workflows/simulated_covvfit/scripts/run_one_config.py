from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from covvfit import preprocess
from covvfit import quasimultinomial as qm
from covvfit._padding import create_padded_array
from covvfit.simulation import logistic

snakemake: Any = globals().get("snakemake")
if snakemake is None:
    raise RuntimeError("This script is intended to be executed by Snakemake.")


def _fit_covvfit(
    ts_observed: list[np.ndarray],
    ys_observed: list[np.ndarray],
    ts_predict: list[np.ndarray],
    *,
    n_variants: int,
    ns,
    n_starts: int,
    maxiter: int,
    random_seed: int,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    time_scaler = preprocess.TimeScaler()
    ts_scaled = time_scaler.fit_transform(ts_observed)
    ts_predict_scaled = time_scaler.transform(ts_predict)

    loss_fit = qm.construct_total_loss(
        ys=[jnp.asarray(y) for y in ys_observed],
        ts=[jnp.asarray(t) for t in ts_scaled],
        ns=ns,
        average_loss=True,
    )
    loss_cov = qm.construct_total_loss(
        ys=[jnp.asarray(y) for y in ys_observed],
        ts=[jnp.asarray(t) for t in ts_scaled],
        ns=ns,
        average_loss=False,
    )
    theta0 = qm.construct_theta0(n_cities=len(ts_observed), n_variants=n_variants)
    solution = qm.jax_multistart_minimize(
        loss_fit,
        theta0=theta0,
        n_starts=n_starts,
        maxiter=maxiter,
        random_seed=random_seed,
    )
    preds = [
        np.asarray(arr)
        for arr in qm.fitted_values(
            [jnp.asarray(t) for t in ts_predict_scaled],
            theta=solution.x,
            cities=list(range(len(ts_observed))),
            n_variants=n_variants,
        )
    ]
    rel_growths = np.asarray(qm.get_relative_growths(solution.x, n_variants=n_variants))
    try:
        covariance = np.asarray(qm.get_covariance(loss_cov, solution.x))
        rel_se = np.sqrt(
            np.clip(np.diag(covariance)[: n_variants - 1], a_min=0.0, a_max=None)
        )
        z = 1.96
        ci_low = rel_growths - z * rel_se
        ci_high = rel_growths + z * rel_se
    except Exception:
        ci_low = np.full_like(rel_growths, np.nan, dtype=float)
        ci_high = np.full_like(rel_growths, np.nan, dtype=float)
    return {
        "predictions": preds,
        "relative_growths": rel_growths,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 2:
        return np.nan
    return float(np.corrcoef(x_arr[valid], y_arr[valid])[0, 1])


cfg = snakemake.config
inf_cfg = cfg["inference"]
wildcards = snakemake.wildcards

mvalue = float(wildcards.mvalue)
sample_size = int(wildcards.depth)
scenario = str(getattr(wildcards, "scenario", "default"))
all_scenarios = cfg.get("simulation_scenarios")
if all_scenarios is None:
    all_scenarios = {"default": cfg["simulation"]}
if scenario not in all_scenarios:
    raise ValueError(
        f"Unknown scenario '{scenario}'. Available: {sorted(all_scenarios)}"
    )
sim_cfg = all_scenarios[scenario]

n_cities = int(sim_cfg["n_cities"])
city_names = sim_cfg["city_names"]
variant_names = sim_cfg["variant_names"]
growth_rates = jnp.asarray(sim_cfg["growth_rates"], dtype=float)
midpoints = jnp.asarray(sim_cfg["midpoints"], dtype=float)
n_observations = np.asarray(sim_cfg["n_observations"], dtype=int)
time0 = float(sim_cfg["time0"])
time1 = float(sim_cfg["time1"])
min_observed = int(sim_cfg["min_observed_per_city"])

if len(city_names) != n_cities:
    raise ValueError("`city_names` must have length `n_cities`.")
if n_observations.shape != (n_cities,):
    raise ValueError("`n_observations` must provide one value per city.")
if midpoints.shape != (n_cities, len(variant_names)):
    raise ValueError("`midpoints` must have shape (n_cities, n_variants).")

settings = logistic.SimulationSettings(
    n_cities=n_cities,
    n_variants=len(variant_names),
    growth_rates=growth_rates,
    midpoints=midpoints,
    n_multinomial=jnp.asarray([sample_size] * n_cities),
    n_observations=jnp.asarray(n_observations),
    time0=time0,
    time1=time1,
)

seed_base = int(sim_cfg["seed"])
scenario_offset = sum(ord(c) for c in scenario) * 1000
seed = seed_base + scenario_offset + int(round(mvalue * 10_000)) + sample_size * 100_000
rng = np.random.default_rng(seed)

ts_full: list[np.ndarray] = []
ys_true_full: list[np.ndarray] = []
ys_obs_full: list[np.ndarray] = []
obs_mask_full: list[np.ndarray] = []
ts_observed: list[np.ndarray] = []
ys_observed: list[np.ndarray] = []
ns_observed: list[np.ndarray] = []

for city_idx in range(n_cities):
    ts_city, ys_true_city = settings.calculate_abundances_one_city(city_index=city_idx)
    ts_city = np.asarray(ts_city, dtype=float)
    ys_true_city = np.asarray(ys_true_city, dtype=float)
    draws = np.vstack([rng.multinomial(sample_size, probs) for probs in ys_true_city])
    ys_obs_city = draws / sample_size

    mask = rng.random(len(ts_city)) >= mvalue
    if mask.sum() < min_observed:
        keep_idx = np.linspace(
            0, len(ts_city) - 1, min(min_observed, len(ts_city)), dtype=int
        )
        mask[keep_idx] = True

    ts_full.append(ts_city)
    ys_true_full.append(ys_true_city)
    ys_obs_full.append(ys_obs_city)
    obs_mask_full.append(mask)
    ts_observed.append(ts_city[mask])
    ys_observed.append(ys_obs_city[mask])
    ns_observed.append(np.full(mask.sum(), sample_size, dtype=float))

fit_weighted = _fit_covvfit(
    ts_observed=ts_observed,
    ys_observed=ys_observed,
    ts_predict=ts_full,
    n_variants=len(variant_names),
    ns=[jnp.asarray(x) for x in ns_observed],
    n_starts=int(inf_cfg["n_starts"]),
    maxiter=int(inf_cfg["maxiter"]),
    random_seed=seed + 11,
)
pred_weighted = fit_weighted["predictions"]

true_relative_growths = np.asarray(growth_rates, dtype=float)[1:] - float(
    growth_rates[0]
)
advantage_data = {}
for i, variant in enumerate(variant_names[1:]):
    advantage_data[variant] = {
        "true": float(true_relative_growths[i]),
        "estimate": float(np.asarray(fit_weighted["relative_growths"])[i]),
        "low": float(np.asarray(fit_weighted["ci_low"])[i]),
        "high": float(np.asarray(fit_weighted["ci_high"])[i]),
    }

# Cities can have different numbers of sampled timepoints.
# Use the shared covvfit padding utility for consistency.
lengths = [len(mask) for mask in obs_mask_full]
obs_matrix = create_padded_array(
    values=[mask.astype(float) for mask in obs_mask_full],
    lengths=lengths,
    padding_length=max(lengths),
    padding_value=np.nan,
    _out_dtype=float,
)
mval_day = 1.0 - np.nanmean(np.asarray(obs_matrix), axis=0)

records: list[dict] = []
date_origin = pd.to_datetime(sim_cfg["date_origin"])

for city_idx, city_name in enumerate(city_names):
    ts_city = ts_full[city_idx]
    dt_index = date_origin + pd.to_timedelta(ts_city * 100, unit="D")
    ys_true_city = ys_true_full[city_idx]
    ys_obs_city = ys_obs_full[city_idx]
    mask_city = obs_mask_full[city_idx]

    for t_idx, day in enumerate(ts_city):
        for variant_idx, variant in enumerate(variant_names):
            records.append(
                {
                    "index": dt_index[t_idx].date().isoformat(),
                    "day": float(day),
                    "city": city_name,
                    "variant": variant,
                    "ground": float(ys_true_city[t_idx, variant_idx]),
                    "prediction": float(pred_weighted[city_idx][t_idx, variant_idx]),
                    "observed": (
                        float(ys_obs_city[t_idx, variant_idx])
                        if bool(mask_city[t_idx])
                        else np.nan
                    ),
                    "is_observed": int(mask_city[t_idx]),
                    "mval_day": float(mval_day[t_idx]),
                    "mval": float(mvalue),
                    "bw": float(sample_size),
                    "scenario": scenario,
                    "reference_variant": variant_names[0],
                    "adv_variant": variant if variant_idx > 0 else np.nan,
                    "true_relative_growth": (
                        advantage_data[variant]["true"] if variant_idx > 0 else np.nan
                    ),
                    "estimated_relative_growth": (
                        advantage_data[variant]["estimate"]
                        if variant_idx > 0
                        else np.nan
                    ),
                    "ci_low": (
                        advantage_data[variant]["low"] if variant_idx > 0 else np.nan
                    ),
                    "ci_high": (
                        advantage_data[variant]["high"] if variant_idx > 0 else np.nan
                    ),
                }
            )

out_df = pd.DataFrame.from_records(records)

# Store per-config global fit quality to simplify downstream plotting.
cor = _safe_corr(out_df["prediction"], out_df["ground"])
out_df["cor"] = cor

out_path = Path(str(snakemake.output[0]))
out_path.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(out_path, index=False)
