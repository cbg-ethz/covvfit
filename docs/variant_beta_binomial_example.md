# Hierarchical Bayesian Example

This page explains how to run the end-to-end hierarchical Bayesian demo for the new variant beta-binomial model.

For detailed practical VI recommendations (guide choice, tuning, and failure-mode debugging), see the [VI guidelines](./variational_inference_guidelines.md).

The script is:

- `examples/variant_beta_binomial_hierarchical_bayes.py`

It does four things in one run:

1. Simulates multi-city beta-binomial count data from known ground truth.
2. Fits a hierarchical Bayesian model using either:
   - HMC (`NUTS`), or
   - variational inference (`SVI` with NumPyro autoguides).
3. Prints MCMC diagnostics (`n_eff`, `r_hat`, divergences).
4. Prints recovery summary comparing posterior slopes with the true slopes.
5. Saves posterior predictive plots with uncertainty bands.

## Prerequisites

From the repository root:

```bash
micromamba activate covvfit
```

If this is your first run in the environment, ensure dependencies are installed (`numpyro`, `jax`, etc.).

## Quick Run

Use recovery-friendly defaults:

```bash
python examples/variant_beta_binomial_hierarchical_bayes.py
```

This uses `--inference-method hmc` by default.

## Dramatic Emergence Scenario (3 Variants)

To simulate sharp logistic takeover dynamics (one early dominant variant, then two emergent variants, with one becoming dominant), run:

```bash
python examples/variant_beta_binomial_hierarchical_bayes.py \
  --scenario dramatic_emergence \
  --inference-method hmc \
  --n-samples 240 \
  --coverage 400 \
  --time-max 12 \
  --warmup 700 \
  --samples 700 \
  --output-dir generated/variant_beta_binomial_dramatic
```

## Run with VI

```bash
python examples/variant_beta_binomial_hierarchical_bayes.py \
  --inference-method svi \
  --svi-steps 8000 \
  --svi-lr 0.01 \
  --svi-guide lowrank \
  --svi-rank 8 \
  --svi-restarts 3 \
  --samples 1000
```

Flow-based NumPyro autoguide example (no extra dependencies):

```bash
python examples/variant_beta_binomial_hierarchical_bayes.py \
  --inference-method svi \
  --svi-guide bnaf \
  --svi-num-flows 2 \
  --svi-bnaf-hidden-factors 8,8 \
  --svi-lr 0.001 \
  --svi-steps 10000 \
  --svi-restarts 3 \
  --samples 1000
```

By default, plots are saved to:

- `generated/variant_beta_binomial_demo/abundance_curves_by_city.png`
- `generated/variant_beta_binomial_demo/mutation_predictions_by_city_locus.png`

## Custom Run

You can override the simulation and inference settings from CLI:

```bash
python examples/variant_beta_binomial_hierarchical_bayes.py \
  --seed 1 \
  --inference-method hmc \
  --n-samples 240 \
  --coverage 400 \
  --time-max 14 \
  --output-dir generated/variant_beta_binomial_custom \
  --n-grid 250 \
  --predictive-draws 6000 \
  --warmup 900 \
  --samples 900
```

Available options:

- `--seed`: random seed
- `--scenario`: `baseline` or `dramatic_emergence`
- `--inference-method`: `hmc` or `svi`
- `--n-samples`: total simulated samples
- `--coverage`: per-locus read depth
- `--time-max`: maximum time used in simulation
- `--warmup`: NUTS warmup steps (HMC only)
- `--target-accept-prob`: NUTS target acceptance probability (HMC only)
- `--svi-steps`: optimization steps for SVI
- `--svi-lr`: learning rate for SVI
- `--svi-guide`: `lowrank` (default), `autonormal`, `iaf`, or `bnaf` (all NumPyro autoguides)
- `--svi-rank`: rank for low-rank guide
- `--svi-num-flows`: number of flow transforms for `iaf`/`bnaf`
- `--svi-iaf-hidden-dims`: comma-separated hidden dimensions for `iaf` (e.g. `128,128`)
- `--svi-bnaf-hidden-factors`: comma-separated hidden factors for `bnaf` (e.g. `8,8`)
- `--svi-restarts`: number of SVI restarts; best ELBO run is used
- `--samples`: posterior draws (HMC samples or SVI guide posterior samples)
- `--output-dir`: output directory for generated plots
- `--n-grid`: number of time points for smooth predictive curves
- `--predictive-coverage`: read depth used for simulated observed-frequency bands
- `--predictive-draws`: number of posterior predictive draws for smoother observed-frequency bands
- `--show-plots`: display plots interactively after saving

## What to Look At in the Output

Focus on:

- `Number of divergences`: target is `0`
- `r_hat`: should be close to `1.00`
- `Recovery summary (slopes)`:
  - `Posterior mean slopes` should be close to `True slopes`
  - `Truth in 90% interval` should ideally be all `True`
- Inference-specific summary:
  - HMC: check divergences and effective sample diagnostics
  - SVI: check optimization loss (`initial_loss` to `final_loss`) and selected restart
- Generated figures:
  - abundance plots per city: posterior median + 90% band + ground truth
  - mutation plots per city/locus:
    - observed points (`Y/N`)
    - latent mutation band (`mu`)
    - observed-frequency posterior predictive band (simulated `Y/N`)
    - ground truth

## Notes

- Runs are stochastic. Different seeds can give slightly different posterior summaries.
- If observed-frequency bands look jagged, increase `--predictive-draws`.
- If recovery is unstable, increase `--n-samples`, `--coverage`, and/or MCMC budget (`--warmup`, `--samples`).
- This demo intentionally uses only NumPyro autoguides for VI; no FlowJAX dependency is required.
- For flow guides, `bnaf` is currently more numerically stable in this demo than `iaf`.
- For `iaf`, hidden dimensions must be large enough for the latent dimension (if you see an input-dimension error, increase `--svi-iaf-hidden-dims`).
