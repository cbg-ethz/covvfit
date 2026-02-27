# Specify the configuration file
# configfile: "config_hyperparam.yaml"

import yaml
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, '../')
from WWdec.main import *

# Add the path to the simulation_utils.py file
sys.path.insert(0, os.path.dirname(__file__))

# Import functions from utils.py
from simulation_utils import (
    generate_final_df,
)

# Define paths dynamically from the configuration file
VOC_DIR = config["simulation"]["directory"]
OUTPUT_DIR = config["simulation"]["output_dir"]
VARIANT_NAMES = config["simulation"]["variant_names"]
MVALUE_LIST = config["simulation"]["mvalue_list"]
BW_LIST = config["simulation"]["bw_list"]
FSCALE_LIST = config["simulation"]["f_scale_list"]  # New hyperparameter array

# Rule to define the final outputs of the workflow
rule all:
    input:
        f"{OUTPUT_DIR}/final_results.csv",
        f"{OUTPUT_DIR}/final_fitness_covvfit.csv"

# Rule to simulate data
rule simulate_data:
    output:
        csv=f"{OUTPUT_DIR}/simulated_mval{{mvalue}}_bw{{bw}}.csv",
        ground_truth=temp(f"{OUTPUT_DIR}/ground_mval{{mvalue}}_bw{{bw}}.csv")
    params:
        a1=config["simulation"]["a1"],
        m1=config["simulation"]["m1"],
        rng_seed=config["simulation"]["rng_seed"],
        points=config["simulation"]["points"],
        mu=config["simulation"]["mu"],
        n=config["simulation"]["n"],
        dropout_prob=lambda wildcards: float(wildcards.mvalue),  # Use mvalue as dropout_prob
        mut_rate=config["simulation"]["mut_rate"],
        overdisp=config["simulation"]["overdisp"]
    run:
        # Simulate data
        final_df, xx1, bb1, yy1, observed_counts, coverage_matrix = generate_final_df(
            rng_seed=params.rng_seed,
            directory=VOC_DIR,
            variant_names=VARIANT_NAMES,
            points=params.points,
            a1=np.array(params.a1),
            m1=np.array(params.m1),
            mu=params.mu,
            n=params.n,
            dropout_prob=params.dropout_prob,  # Dynamically set dropout_prob
            mut_rate=params.mut_rate,
            overdisp=params.overdisp
        )

        # Save simulated data
        final_df.to_csv(output.csv, index=False)

        # Generate and save ground truth
        ground_df = pd.DataFrame(
            bb1.T,
            index=pd.to_datetime(xx1, unit="D", origin="1970-01-01"),
            columns=VARIANT_NAMES
        )
        ground_df_melted = ground_df.reset_index().melt(
            id_vars="index", value_name="ground", var_name="variant"
        )
        ground_df_melted.to_csv(output.ground_truth, index=False)

# Rule to perform robust deconvolution with hyperparameter f_scale
rule deconvolve_robust:
    input:
        csv=f"{OUTPUT_DIR}/simulated_mval{{mvalue}}_bw{{bw}}.csv"
    output:
        csv=temp(f"{OUTPUT_DIR}/robust_mval{{mvalue}}_bw{{bw}}_fscale{{fscale}}.csv")
    params:
        bw=lambda wildcards: float(wildcards.bw),
        f_scale=lambda wildcards: float(wildcards.fscale),  # Dynamically set f_scale
        variant_names=VARIANT_NAMES
    run:
        # Load simulated data
        final_df = pd.read_csv(input.csv)

        # Ensure correct data types
        final_df["pos"] = pd.to_numeric(final_df["pos"], errors="coerce")
        final_df["value"] = pd.to_numeric(final_df["value"], errors="coerce")
        final_df["date"] = pd.to_datetime(final_df["date"], errors="coerce")

        # Perform robust deconvolution
        t_kdec = KernelDeconv(
            final_df[params.variant_names],
            final_df["value"],
            final_df["date"],
            kernel=GaussianKernel(params.bw),
            reg=RobustReg(f_scale=params.f_scale),  # Use f_scale from params
            confint=NullConfint(),
        )
        t_kdec = t_kdec.deconv_all()
        # Save results
        res = t_kdec.fitted.sort_index().reset_index()
        res.to_csv(output.csv, index=False)

# Rule to merge results
rule merge_results:
    input:
        robust=f"{OUTPUT_DIR}/robust_mval{{mvalue}}_bw{{bw}}_fscale{{fscale}}.csv",
        ground_truth=f"{OUTPUT_DIR}/ground_mval{{mvalue}}_bw{{bw}}.csv"
    output:
        csv=f"{OUTPUT_DIR}/merged_mval{{mvalue}}_bw{{bw}}_fscale{{fscale}}.csv"
    params:
        mvalue=lambda wildcards: float(wildcards.mvalue),
        bw=lambda wildcards: float(wildcards.bw),
        f_scale=lambda wildcards: float(wildcards.fscale)
    run:
        import pandas as pd

        # Load robust results and ground truth
        robust_df = pd.read_csv(input.robust)
        ground_df = pd.read_csv(input.ground_truth)

        # Melt the robust results
        robust_melted = robust_df.melt(
            id_vars=["index"],  # Keep the "index" column
            var_name="variant",  # Column name for melted variants
            value_name="estimate"  # Column name for melted values
        )

        # Merge robust results with ground truth
        merged = pd.merge(
            robust_melted,
            ground_df,
            on=["index", "variant"],
            suffixes=("_robust", "_ground")
        )

        # Add metadata
        merged["mval"] = params.mvalue
        merged["bw"] = params.bw
        merged["f_scale"] = params.f_scale

        # Save merged results
        merged.to_csv(output.csv, index=False)

# Rule to concatenate results
rule concatenate_results:
    input:
        expand(
            f"{OUTPUT_DIR}/merged_mval{{mvalue:.2f}}_bw{{bw:.2f}}_fscale{{fscale:.3f}}.csv",
            mvalue=MVALUE_LIST,
            bw=BW_LIST,
            fscale=FSCALE_LIST
        )
    output:
        csv=f"{OUTPUT_DIR}/final_results.csv"
    run:
        import pandas as pd

        # Concatenate all merged files
        dfs = [pd.read_csv(file) for file in input]
        final_df = pd.concat(dfs, ignore_index=True)

        # Save the concatenated DataFrame
        final_df.to_csv(output.csv, index=False)


# Rule to infer variant fitness from deconvolved trajectories using covvfit
rule fit_covvfit_fitness:
    input:
        merged=f"{OUTPUT_DIR}/merged_mval{{mvalue}}_bw{{bw}}_fscale{{fscale}}.csv"
    output:
        csv=f"{OUTPUT_DIR}/fitness_mval{{mvalue}}_bw{{bw}}_fscale{{fscale}}.csv"
    params:
        n_starts=10,
        variant_names=VARIANT_NAMES,
        mvalue=lambda wildcards: float(wildcards.mvalue),
        bw=lambda wildcards: float(wildcards.bw),
        f_scale=lambda wildcards: float(wildcards.fscale)
    run:
        import jax.numpy as jnp
        import pandas as pd
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
                "mval": params.mvalue,
                "bw": params.bw,
                "f_scale": params.f_scale,
            }
        )

        fitness_df.to_csv(output.csv, index=False)


# Rule to concatenate covvfit fitness estimates from all hyperparameter settings
rule concatenate_covvfit_fitness:
    input:
        expand(
            f"{OUTPUT_DIR}/fitness_mval{{mvalue:.2f}}_bw{{bw:.2f}}_fscale{{fscale:.3f}}.csv",
            mvalue=MVALUE_LIST,
            bw=BW_LIST,
            fscale=FSCALE_LIST,
        )
    output:
        csv=f"{OUTPUT_DIR}/final_fitness_covvfit.csv"
    run:
        import pandas as pd

        dfs = [pd.read_csv(file) for file in input]
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(output.csv, index=False)

