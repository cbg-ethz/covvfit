# Hierarchical Bayesian Example

This page explains how to run the end-to-end hierarchical Bayesian demo for the new variant beta-binomial model.

The script is:

- `examples/variant_beta_binomial_hierarchical_bayes.py`

It does four things in one run:

1. Simulates multi-city beta-binomial count data from known ground truth.
2. Fits a hierarchical Bayesian model with NumPyro (NUTS).
3. Prints MCMC diagnostics (`n_eff`, `r_hat`, divergences).
4. Prints recovery summary comparing posterior slopes with the true slopes.

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

## Custom Run

You can override the simulation and inference settings from CLI:

```bash
python examples/variant_beta_binomial_hierarchical_bayes.py \
  --seed 1 \
  --n-samples 240 \
  --coverage 400 \
  --time-max 14 \
  --warmup 900 \
  --samples 900
```

Available options:

- `--seed`: random seed
- `--n-samples`: total simulated samples
- `--coverage`: per-locus read depth
- `--time-max`: maximum time used in simulation
- `--warmup`: NUTS warmup steps
- `--samples`: posterior draws

## What to Look At in the Output

Focus on:

- `Number of divergences`: target is `0`
- `r_hat`: should be close to `1.00`
- `Recovery summary (slopes)`:
  - `Posterior mean slopes` should be close to `True slopes`
  - `Truth in 90% interval` should ideally be all `True`

## Notes

- Runs are stochastic. Different seeds can give slightly different posterior summaries.
- If recovery is unstable, increase `--n-samples`, `--coverage`, and/or MCMC budget (`--warmup`, `--samples`).
