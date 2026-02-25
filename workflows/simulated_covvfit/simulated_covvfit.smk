configfile: "workflows/simulated_covvfit/config.yaml"

OUTPUT_DIR = config["output_dir"]
MVALUES = [f"{float(x):.2f}" for x in config["grid"]["missing_rate"]]
DEPTHS = [str(int(x)) for x in config["grid"]["sample_size"]]


rule all:
    input:
        f"{OUTPUT_DIR}/final_results.csv",
        f"{OUTPUT_DIR}/plots/sim_full.pdf",
        f"{OUTPUT_DIR}/plots/sim_full.jpeg",
        f"{OUTPUT_DIR}/plots/sim_subset.pdf",
        f"{OUTPUT_DIR}/plots/missingness_panel.pdf",
        f"{OUTPUT_DIR}/plots/r2_heatmaps.pdf",
        f"{OUTPUT_DIR}/plots/fitness_advantages_panel.pdf",


rule run_one_config:
    output:
        f"{OUTPUT_DIR}/per_config/result_mval{{mvalue}}_depth{{depth}}.csv"
    script:
        "scripts/run_one_config.py"


rule concatenate_results:
    input:
        expand(
            f"{OUTPUT_DIR}/per_config/result_mval{{mvalue}}_depth{{depth}}.csv",
            mvalue=MVALUES,
            depth=DEPTHS,
        )
    output:
        f"{OUTPUT_DIR}/final_results.csv"
    run:
        import pandas as pd

        dfs = [pd.read_csv(path) for path in input]
        pd.concat(dfs, ignore_index=True).to_csv(output[0], index=False)


rule make_plots:
    input:
        f"{OUTPUT_DIR}/final_results.csv"
    output:
        f"{OUTPUT_DIR}/plots/sim_full.pdf",
        f"{OUTPUT_DIR}/plots/sim_full.jpeg",
        f"{OUTPUT_DIR}/plots/sim_subset.pdf",
        f"{OUTPUT_DIR}/plots/missingness_panel.pdf",
        f"{OUTPUT_DIR}/plots/r2_heatmaps.pdf",
        f"{OUTPUT_DIR}/plots/fitness_advantages_panel.pdf",
    script:
        "scripts/plot_results.py"
