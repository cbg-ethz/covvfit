# Simulated Covvfit Workflow

This Snakemake workflow benchmarks Covvfit on synthetic logistic-competition data
over a grid of missing-value rates and multinomial sample sizes.

Run from the repository root:

```bash
snakemake -s workflows/simulated_covvfit/simulated_covvfit.smk --cores 4
```

Optional:

```bash
snakemake -s workflows/simulated_covvfit/simulated_covvfit.smk \
  --configfile workflows/simulated_covvfit/config.yaml \
  --cores 4
```

Outputs are written to `generated/simulated_covvfit/`:

- `final_results.csv`
- `plots/sim_full.pdf`
- `plots/sim_full.jpeg`
- `plots/sim_subset.pdf`
- `plots/missingness_panel.pdf`
- `plots/r2_heatmaps.pdf`
- `plots/fitness_advantages_panel.pdf`
