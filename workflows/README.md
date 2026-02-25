# Workflows

This directory contains Snakemake workflows.

## Benchmark on simulated data

Workflow to simulate variant competition data and benchmark Covvfit
across missingness/sample-size settings:

```bash
$ snakemake -s workflows/simulated_covvfit/simulated_covvfit.smk --cores 4
```

See [workflows/simulated_covvfit/README.md](workflows/simulated_covvfit/README.md)
for outputs and configuration details.


## Assessing bootstrap confidence intervals

This workflow constructs bootstrap confidence and calculates their
coverage in simulated data setting.

Run Snakemake workflow by using: 

```bash
$ snakemake -s workflows/boostrap_simulation.smk -c6
```
