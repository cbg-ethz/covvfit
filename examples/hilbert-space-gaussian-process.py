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

# %% [markdown]
# # Hilbert space approximations to Gaussian processes
#
# This notebook serves as a whiteboard/workpad for the approximation to the Covvfit dynamics using a flexible Gaussian process model.
#
# ## Modelling assumptions
#
# Consider variants numbered as $0, ..., V-1$ (with $0$ serving e.g., as the other variant), which abundance should sum up to one: $y_0(t) + ... + y_{V-1}(t) = 1$.
#
# We typically reparameterize it in terms of an auxiliary function $f$:
# $y(t) = \mathrm{softmax}\, f(t)$,
#
# where $f(t) = (f_0(t), ..., f_{V-1}(t))$. To remove aliasing occurring in softmax, which results in a rather bad non-identifiability in the model, we set $f_0(t) = 0$ everywhere.
#
# We can therefore focus on modeling the functions $f_1, ..., f_{V-1}$.
#
#
# The simplest model is the following:
#
# $$f_v(t) = \mu_v(t; a_v, b_v) := a_v t + b_v,$$
#
# which corresponds to the selection dynamics model. However, this model may not be flexible enough to capture some local behaviour. We will therefore consider a more flexible model.
#
# Let
#
# $$g_v \sim \mathrm{GP}(0, k_v)$$
#
# be a random function $g_v = g_v(t)$ sampled from a Gaussian process with mean function $0$ and a stationary kernel $k_v$. For example, we can take the [RBF kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) $k_v(t, t') = k(t, t'; \lambda_v) = \exp\frac{ -(t-t')^2 }{2 \lambda_v^2 }$, parameterized by the lengthscale $\lambda_v$.
# In principle, $g_v$ can be an arbitrarily flexible function, although the used prior encourages is to be around $0$ and change according to the lengthscale $\lambda_v$.
#
# Now, we will model $f_v$ by summing up the linear (selection dynamics; non-stationary) part and the random function $g_v$ (controlling the departure from the model):
#
# $$f_v(t; a_v, b_v, \alpha_v, \lambda_v) = \mu_v(t; a_v, b_v) + \alpha_v\, g_v(t; \lambda_v).$$
#
# This is very similar to a [mixed-effect model](https://en.wikipedia.org/wiki/Mixed_model#Definition): we are interested in $a_v$ and $b_v$ (which have a selection dynamics interpretation if all $g_v = 0$) and the parameters $\alpha_v$ and $\lambda_v$, controlling the departure from the model. The random function $g_v$ does not have a simple interpretation (being a function, rather than a single number), but can be visualised and is useful from the predictive viewpoint.
#
# Note that if $t$ is a time at which we try to evaluate the abundance vector, $f(t)$, for large $t$ (far from the data points in the set, measured in the units of $\lambda_v$), then the model resorts to the simple linear one, $f_v(t) = \mu_v(t)$. Similarly, if $\alpha_v$ is rather small, we do not expect large deviations from the selection dynamics model.
# However, this model can notice departing behaviours and correct for it by incorporating the uncertainty from function $g_v$.
#
# ## Inference and approximations
#
# This is rather a complex problem: we have parameters of interest $a_v, b_v, \alpha_v, \gamma_v$, as well as the whole unknown function $g_v$! Marginalizing $g_v$ analytically is not tractable and we will want to approximate it somehow.
#
# We will use the Hilbert space approximation to the Gaussian process. There are many wonderful resources on the topic:
#
#   - [This blog post](https://juanitorduz.github.io/hsgp_intro/).
#   - [This article](https://link.springer.com/article/10.1007/s11222-019-09886-w), deriving the mathematical background of the approximation.
#   - [This article](https://arxiv.org/abs/2004.11408), detailing how to use this approximation within probabilistic programming languages.
#
# The main idea is to approximate the function $g_v$ on a finite interval $[-L, L]$ as a linear combination of $D$ basis functions $\phi_d = \phi_d(t)$ defined on $[-L, L]$:
#
# $$g_v(t) \approx \sum_{d=1}^D \gamma_{vd}\, s(\lambda_v, d) \, \phi_d(t),$$
#
# where $\phi_d(t)$ are the basis functions (conceptually similar to splines, although the functional forms (and mathematical motivation) are very different!), $s(\lambda_v, d)$ are scaling factors, easy to calculate using lengthscale $\lambda_v$ and the index $d$, as well as the coefficients $\gamma_{vd}$.
# The prior $g_v \sim \mathrm{GP}(0, k_v)$ corresponds then to $\gamma_{vd} \sim \mathrm{Normal}(0, 1)$.
#
# Look how simple this is! Essentially, for a fixed $\lambda_v$, we obtain something computationally as simple as the Bayesian regression with splines.
# However, we have the strengths of the Gaussian process: $\lambda_v$ can be given its own prior to control how much we smooth.
#
# In other words, we inherit the best of both worlds: computationally tractable computation of Bayesian regression, together with the modelling power of a Gaussian process!
#
# If we introduce the likelihood, connecting the abundance function $y(t)$ to the data, we can perform inference with MCMC samplers or variational inference.
# Note that at the same time we perform the inference on all the parameters, including the coefficients $\gamma_{vd}$ (approximating $g_v$), which makes the model much larger than the simple selection dynamics model.

# %% [markdown]
# ## Implementing the approximation
#
# In this part, we will first focus on building the right code infrastructure, focusing on modeling only a single-output timeseries.

# %%
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import covvfit._hsgp as hsgp

# %%
params = hsgp.AmplitudeLengthParams(amplitude=1.0, lengthscale=0.1)

kernels = {
    "Matern 1/2": hsgp.Matern12(params),
    "Matern 3/2": hsgp.Matern32(params),
    "Matern 5/2": hsgp.Matern52(params),
    "RBF": hsgp.RBFKernel(params),
}

# %%
r = jnp.linspace(params.lengthscale * 1e-3, params.lengthscale * 3.5, 301)

for name, kernel in kernels.items():
    k = kernel.evaluate_kernel(jnp.abs(r))
    plt.plot(r, k, label=name)

plt.legend()

# %%
kernel = hsgp.RBFKernel(params)

x = jnp.linspace(0.1, 0.3, 2)
y = jnp.linspace(0, 0.3, 3)

kernel.gram_matrix(x, y)

# %%
hsgp.approximate_gram(
    kernel,
    x,
    y,
    n_basis=30,
    lengthscale=0.4,  # Note: setting lengthscale = 2.0 breaks the approximation with just 30 vectors
)

# %%
jax.vmap(lambda x: x**2)(jnp.arange(5))

# %%
