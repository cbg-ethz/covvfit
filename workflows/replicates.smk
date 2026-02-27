# Specify the configuration file when running:
# snakemake --snakefile replicates.smk --configfile config_replicates.yaml --cores 4 -p

import numpy as np
import pandas as pd
import os
import sys

WORKFLOW_DIR = os.path.dirname(workflow.snakefile)
PROJECT_ROOT = os.path.abspath(os.path.join(WORKFLOW_DIR, ".."))

sys.path.insert(0, PROJECT_ROOT)
from WWdec.main import *

sys.path.insert(0, WORKFLOW_DIR)
from simulation_utils import generate_final_df

VOC_DIR = config["simulation"]["directory"]
OUTPUT_DIR = config["simulation"]["output_dir"]
VARIANT_NAMES = config["simulation"]["variant_names"]
SEED_LIST = config["simulation"]["seed_list"]
N_TIMEPOINTS_LIST = config["simulation"]["n_timepoints_list"]
COMPUTE_FITNESS = bool(config["simulation"].get("compute_fitness", False))

MVALUE = float(config["simulation"]["mvalue"])
BW = float(config["simulation"]["bw"])
F_SCALE = float(config["simulation"]["f_scale"])


def seed_tag(seed):
    return f"{int(seed):03d}"


SEED_TAGS = [seed_tag(s) for s in SEED_LIST]
SEED_MAP = {seed_tag(s): int(s) for s in SEED_LIST}
NTP_TAGS = [str(int(n)) for n in N_TIMEPOINTS_LIST]


def _select_subsample_indices(n_total, n_target):
    if n_target > n_total:
        raise ValueError(f"Requested {n_target} timepoints, but only {n_total} are available.")
    if n_target == n_total:
        return np.arange(n_total, dtype=int)

    step = max(1, n_total // n_target)
    idx = np.arange(0, step * n_target, step, dtype=int)
    idx = idx[:n_target]
    idx[-1] = n_total - 1
    idx = np.unique(idx)

    if len(idx) != n_target:
        idx = np.round(np.linspace(0, n_total - 1, n_target)).astype(int)
        idx = np.unique(idx)
        if len(idx) != n_target:
            raise ValueError(
                f"Could not build {n_target} unique subsample indices from {n_total} points."
            )
    return idx


rule all:
    input:
        [
            f"{OUTPUT_DIR}/final_results.csv",
            *([f"{OUTPUT_DIR}/final_fitness_covvfit.csv"] if COMPUTE_FITNESS else []),
        ]


rule simulate_full_data:
    output:
        csv=f"{OUTPUT_DIR}/simulated_seed{{seed}}_full.csv",
        ground_truth=temp(f"{OUTPUT_DIR}/ground_seed{{seed}}_full.csv")
    params:
        a1=config["simulation"]["a1"],
        m1=config["simulation"]["m1"],
        points=config["simulation"]["points"],
        mu=config["simulation"]["mu"],
        n=config["simulation"]["n"],
        mut_rate=config["simulation"]["mut_rate"],
        overdisp=config["simulation"]["overdisp"]
    run:
        seed = SEED_MAP[wildcards.seed]

        final_df, xx1, bb1, yy1, observed_counts, coverage_matrix = generate_final_df(
            rng_seed=seed,
            directory=VOC_DIR,
            variant_names=VARIANT_NAMES,
            points=params.points,
            a1=np.array(params.a1),
            m1=np.array(params.m1),
            mu=params.mu,
            n=params.n,
            dropout_prob=MVALUE,
            mut_rate=params.mut_rate,
            overdisp=params.overdisp,
        )

        final_df.to_csv(output.csv, index=False)

        ground_df = pd.DataFrame(
            bb1.T,
            index=pd.to_datetime(xx1, unit="D", origin="1970-01-01"),
            columns=VARIANT_NAMES,
        )
        ground_df_melted = ground_df.reset_index().melt(
            id_vars="index", value_name="ground", var_name="variant"
        )
        ground_df_melted.to_csv(output.ground_truth, index=False)


rule subsample_simulated_data:
    input:
        csv=f"{OUTPUT_DIR}/simulated_seed{{seed}}_full.csv",
        ground_truth=f"{OUTPUT_DIR}/ground_seed{{seed}}_full.csv"
    output:
        csv=f"{OUTPUT_DIR}/simulated_seed{{seed}}_ntp{{ntp}}.csv",
        ground_truth=temp(f"{OUTPUT_DIR}/ground_seed{{seed}}_ntp{{ntp}}.csv")
    run:
        n_target = int(wildcards.ntp)

        sim_df = pd.read_csv(input.csv)
        sim_df["date"] = pd.to_datetime(sim_df["date"], errors="coerce")

        ground_df = pd.read_csv(input.ground_truth)
        ground_df["index"] = pd.to_datetime(ground_df["index"], errors="coerce")

        all_dates = np.array(sorted(ground_df["index"].dropna().unique()))
        idx = _select_subsample_indices(len(all_dates), n_target)
        selected_dates = set(all_dates[idx])

        sim_sub = sim_df[sim_df["date"].isin(selected_dates)].copy()
        ground_sub = ground_df[ground_df["index"].isin(selected_dates)].copy()

        sim_sub.to_csv(output.csv, index=False)
        ground_sub.to_csv(output.ground_truth, index=False)


rule deconvolve_robust:
    input:
        csv=f"{OUTPUT_DIR}/simulated_seed{{seed}}_ntp{{ntp}}.csv"
    output:
        csv=temp(f"{OUTPUT_DIR}/robust_seed{{seed}}_ntp{{ntp}}.csv")
    params:
        variant_names=VARIANT_NAMES
    run:
        final_df = pd.read_csv(input.csv)

        final_df["pos"] = pd.to_numeric(final_df["pos"], errors="coerce")
        final_df["value"] = pd.to_numeric(final_df["value"], errors="coerce")
        final_df["date"] = pd.to_datetime(final_df["date"], errors="coerce")

        t_kdec = KernelDeconv(
            final_df[params.variant_names],
            final_df["value"],
            final_df["date"],
            kernel=GaussianKernel(BW),
            reg=RobustReg(f_scale=F_SCALE),
            confint=NullConfint(),
        )
        t_kdec = t_kdec.deconv_all()
        res = t_kdec.fitted.sort_index().reset_index()
        res.to_csv(output.csv, index=False)


rule merge_results:
    input:
        robust=f"{OUTPUT_DIR}/robust_seed{{seed}}_ntp{{ntp}}.csv",
        ground_truth=f"{OUTPUT_DIR}/ground_seed{{seed}}_ntp{{ntp}}.csv"
    output:
        csv=f"{OUTPUT_DIR}/merged_seed{{seed}}_ntp{{ntp}}.csv"
    run:
        robust_df = pd.read_csv(input.robust)
        ground_df = pd.read_csv(input.ground_truth)

        robust_melted = robust_df.melt(
            id_vars=["index"],
            var_name="variant",
            value_name="estimate",
        )

        merged = pd.merge(
            robust_melted,
            ground_df,
            on=["index", "variant"],
            suffixes=("_robust", "_ground"),
        )

        merged["seed"] = SEED_MAP[wildcards.seed]
        merged["n_timepoints"] = int(wildcards.ntp)
        merged["mval"] = MVALUE
        merged["bw"] = BW
        merged["f_scale"] = F_SCALE

        merged.to_csv(output.csv, index=False)


rule concatenate_results:
    input:
        expand(
            f"{OUTPUT_DIR}/merged_seed{{seed}}_ntp{{ntp}}.csv",
            seed=SEED_TAGS,
            ntp=NTP_TAGS,
        )
    output:
        csv=f"{OUTPUT_DIR}/final_results.csv"
    run:
        dfs = [pd.read_csv(file) for file in input]
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(output.csv, index=False)


rule fit_covvfit_fitness:
    input:
        merged=f"{OUTPUT_DIR}/merged_seed{{seed}}_ntp{{ntp}}.csv"
    output:
        csv=f"{OUTPUT_DIR}/fitness_seed{{seed}}_ntp{{ntp}}.csv"
    params:
        n_starts=10,
        variant_names=VARIANT_NAMES
    run:
        import jax.numpy as jnp
        from covvfit import preprocess
        from covvfit import quasimultinomial as qm

        fit_df = pd.read_csv(input.merged)
        fit_df["date"] = pd.to_datetime(fit_df["index"], errors="coerce")

        long_df = fit_df[["date", "variant", "estimate"]].rename(
            columns={"estimate": "proportion"}
        )

        wide_df = (
            long_df.pivot_table(
                index="date", columns="variant", values="proportion", aggfunc="mean"
            )
            .reset_index()
            .sort_values("date")
        )

        for variant in params.variant_names:
            if variant not in wide_df.columns:
                wide_df[variant] = 0.0

        wide_df[params.variant_names] = wide_df[params.variant_names].fillna(0.0)
        row_sums = wide_df[params.variant_names].sum(axis=1)
        valid = row_sums > 0
        wide_df = wide_df.loc[valid].copy()
        wide_df[params.variant_names] = wide_df[params.variant_names].div(
            row_sums.loc[valid], axis=0
        )

        ts = (
            (wide_df["date"] - wide_df["date"].min()) / pd.to_timedelta(1, "D")
        ).to_numpy(dtype=float)
        ys = wide_df[params.variant_names].to_numpy(dtype=float)

        ts_lst = [jnp.asarray(ts)]
        ys_lst = [jnp.asarray(ys)]

        time_scaler = preprocess.TimeScaler()
        ts_scaled = time_scaler.fit_transform(ts_lst)

        n_variants = len(params.variant_names)
        loss = qm.construct_total_loss(ys=ys_lst, ts=ts_scaled, average_loss=False)
        theta0 = qm.construct_theta0(n_cities=1, n_variants=n_variants)

        solution = qm.jax_multistart_minimize(loss, theta0, n_starts=params.n_starts)
        theta_star = solution.x

        ys_fitted = qm.fitted_values(
            ts_scaled, theta=theta_star, cities=["sim"], n_variants=n_variants
        )
        covariance = qm.get_covariance(loss, theta_star)
        overdisp = qm.compute_overdispersion(observed=ys_lst, predicted=ys_fitted).overall
        covariance_scaled = overdisp * covariance

        standard_errors = qm.get_standard_errors(covariance_scaled)
        confints = qm.get_confidence_intervals(
            theta_star, standard_errors, confidence_level=0.95
        )

        rel = np.asarray(qm.get_relative_growths(theta_star, n_variants=n_variants))
        rel_low = np.asarray(qm.get_relative_growths(confints[0], n_variants=n_variants))
        rel_up = np.asarray(qm.get_relative_growths(confints[1], n_variants=n_variants))

        rel_per_day = rel / float(time_scaler.time_unit)
        rel_low_per_day = rel_low / float(time_scaler.time_unit)
        rel_up_per_day = rel_up / float(time_scaler.time_unit)

        fitness_df = pd.DataFrame(
            {
                "reference_variant": params.variant_names[0],
                "variant": params.variant_names[1:],
                "fitness_per_day": rel_per_day,
                "fitness_ci_lower": rel_low_per_day,
                "fitness_ci_upper": rel_up_per_day,
                "fitness_per_week": 7.0 * rel_per_day,
                "fitness_per_week_ci_lower": 7.0 * rel_low_per_day,
                "fitness_per_week_ci_upper": 7.0 * rel_up_per_day,
                "overdispersion": float(overdisp),
                "seed": SEED_MAP[wildcards.seed],
                "n_timepoints": int(wildcards.ntp),
                "mval": MVALUE,
                "bw": BW,
                "f_scale": F_SCALE,
            }
        )

        fitness_df.to_csv(output.csv, index=False)


rule concatenate_covvfit_fitness:
    input:
        expand(
            f"{OUTPUT_DIR}/fitness_seed{{seed}}_ntp{{ntp}}.csv",
            seed=SEED_TAGS,
            ntp=NTP_TAGS,
        )
    output:
        csv=f"{OUTPUT_DIR}/final_fitness_covvfit.csv"
    run:
        dfs = [pd.read_csv(file) for file in input]
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(output.csv, index=False)
