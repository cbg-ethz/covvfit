# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import matplotlib.pyplot as plt

from numpyro.infer import NUTS, MCMC
from sklearn.preprocessing import SplineTransformer
from jax import random


# Build a spline basis
spl = SplineTransformer(
    degree=3,
    n_knots=10,
    extrapolation="constant",  # extrapolation behaviour
    include_bias=False,  # we add our own intercept in the design matrix
)

# Simulate observations on [0,1]
n_points = 50
rng_key = random.PRNGKey(0)
x_train = np.linspace(0, 1, n_points).reshape(-1, 1)
f_true = lambda x: 1.5 - 0.7 * x - 0.3 * x**2
y_train = f_true(x_train) + 0.15 * random.normal(rng_key, x_train.shape)

# Design matrix: [1, x] for the linear trennd +  spline columns
X_lin = np.hstack([np.ones_like(x_train), x_train])  # shape (N, 2)
X_spline = spl.fit_transform(x_train)  # shape (N, K)
X_train = jnp.asarray(np.hstack([X_lin, X_spline]))  # shape (N, 2 + K)


def model(X, y=None):
    n_lin = 2
    n_spl = X.shape[1] - n_lin  # Number of spline columns

    beta = numpyro.sample(
        "beta", dist.Normal(0, 10).expand([n_lin])
    )  # slope and intercept
    gamma = numpyro.sample(
        "gamma", dist.Normal(0, 1).expand([n_spl])
    )  # splines coefficient (hopefully minor)
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    mu = X[..., :2] @ beta + X[..., 2:] @ gamma
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=1000)

mcmc.run(rng_key, X_train, y_train.squeeze())
samples = mcmc.get_samples()

# Posterior predictive on [0, 1.5]
x_pred = np.linspace(0, 1.5, 300).reshape(-1, 1)
X_lin_p = np.hstack([np.ones_like(x_pred), x_pred])
X_spl_p = spl.transform(x_pred)  # constant beyond 1
X_pred = jnp.asarray(np.hstack([X_lin_p, X_spl_p]))

# Get predictions
y_pred = (
    X_pred @ np.concatenate([samples["beta"], samples["gamma"]], axis=1)[:, :, None]
).squeeze()

# Plot the data and the fit
fig, ax = plt.subplots()
ax.spines[["top", "right"]].set_visible(False)

ax.scatter(x_train, y_train, s=25, alpha=0.6, label="data", c="maroon")

ax.plot(
    x_pred.squeeze(),
    np.median(y_pred, axis=0),
    color="darkblue",
)


for coverage in [0.75, 0.9, 0.95]:
    alpha = 1 - coverage
    lower = alpha / 2
    upper = 1 - alpha / 2

    ax.fill_between(
        x_pred.squeeze(),
        np.quantile(y_pred, q=lower, axis=0),
        np.quantile(y_pred, q=upper, axis=0),
        color="darkblue",
        alpha=0.3,
    )

ax.axvspan(1, 1.5, color="grey", alpha=0.3)
ax.set_xlim(0, 1.5)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Extrapolation with Bayesian spline regression")

# %%
