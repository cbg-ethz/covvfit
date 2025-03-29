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
#
# Let's take a look at the implemented kernels. We use a few kernels, which can be understood in the following fashion:
#
# 1. They are stationary, with covariance between $x$ and $y$ given by $\mathrm{Cov}(x, y) = k(|x-y|)$.
# 2. It is possible to represent them via the spectral density, $S(\omega)$.
# 3. Each of these kernels is parameterized in terms of amplitude (amplitude $a$ scales the covariance by $a^2$) and lengthscale (lengthscale $\ell$ means that we effectively use $|x-y|/\ell$ as the normalized distance within the kernel).
#
# Let's plot them:

# %%
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import covvfit._hsgp as hsgp


params = hsgp.AmplitudeLengthParams(amplitude=1.0, lengthscale=0.1)

kernels = {
    "Matern 1/2": hsgp.Matern12(params),
    "Matern 3/2": hsgp.Matern32(params),
    "Matern 5/2": hsgp.Matern52(params),
    "RBF": hsgp.RBFKernel(params),
}


r = jnp.linspace(params.lengthscale * 1e-3, params.lengthscale * 3.5, 301)

fig, axs = plt.subplots(1, 2, figsize=(2 * 3, 2), dpi=300)

ax = axs[0]

for name, kernel in kernels.items():
    k = kernel.evaluate_kernel(jnp.abs(r))
    ax.plot(r, k, label=name)

ax.legend(frameon=False)
ax.set_xlabel("$r$")
ax.set_ylabel("$k(r)$")


ax = axs[1]
omega = jnp.linspace(1e-3, 30.0, 101)

for name, kernel in kernels.items():
    s = kernel.spectral_density(omega)
    ax.plot(omega, s, label=name)

ax.set_xlabel("$\\omega$")
ax.set_ylabel("$S(\\omega)$")

for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()


# %% [markdown]
# While the above functions define the kernel, it is perhaps hard to visualize it. How do we visualise a sampled function $g \sim \mathrm{GP}(0, k)$?
#
# By discretization: let $x_1, ..., x_m$ be a points on which we want to obtain the function values $g(x_1), \dotsc, g(x_m)$.
# From the definition of a Gaussian process, these values can be sampled by sampling a random vector $(Y_1, ..., Y_m)$, where the covariance matrix is given by $\mathrm{Cov}(Y_i, Y_j) = k(x_i, x_j)$. This is a particular case of the [Gram matrix](https://en.wikipedia.org/wiki/Gram_matrix).
#
# Let's sample a few multivariate normal vectors, then:

# %%
fig, axs = plt.subplots(len(kernels), sharex=True, sharey=True)

key = jax.random.PRNGKey(42)

for ax, (name, kernel) in zip(axs, kernels.items()):
    key, subkey = jax.random.split(key)
    n_points = 81
    jitter = 1e-5
    x = jnp.linspace(-1, 1, n_points)
    K = kernel.gram_matrix(x, x) + jitter * jnp.eye(
        n_points
    )  # Add a small jitter for numerical stability

    n_samples = 5
    for i in range(n_samples):
        new_key = jax.random.fold_in(subkey, i)
        g_x = jax.random.multivariate_normal(
            new_key,
            jnp.zeros_like(x),
            K,
        )
        ax.plot(x, g_x, alpha=1.0)

    ax.set_ylabel(name)

ax = axs[-1]
ax.set_xlabel("$x$")

for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()


# %% [markdown]
# We see that different kernels, even if they have the same amplitude and lengthscale, have quite different smoothness properties. The RBF kernel is smooth, while Matern kernels have only a finite number of derivatives.
#
# We should not that constructing the Gram matrix for $m$ points, required $O(m^2)$ operations. By truncating the domain to the interval $[-L, L]$, we can approximate the Gram matrix using a finite number of basis functions. See Section 4 [here](https://arxiv.org/abs/2004.11408) for the guidelines on how to choose the number of basis function used in the approximation.
#
# As a simple check, we can detect issues by evaluating the Gram matrices (note that it's similar to the convergence checks: this method does not prove that we have enough basis elements, but rather can be used to spot issues):

# %%
kernel = hsgp.RBFKernel(params)

x = jnp.linspace(0.1, 0.3, 2)
y = jnp.linspace(0, 0.3, 3)

print("--- Exact: ---")
print(kernel.gram_matrix(x, y))

print("\n\n--- Approximation: ---")
hsgp.approximate_gram_matrix(
    kernel,
    x,
    y,
    n_basis=50,
    lengthscale=1.5,
)

# %% [markdown]
# Alternatively, we can also look at the covariance function $k(r)$:

# %%
r = jnp.linspace(params.lengthscale * 1e-3, params.lengthscale * 3.5, 5)

print("--- Exact: ---")
print(kernel.evaluate_kernel(r))

print("\n\n--- Approximation: ---")

reference = jnp.array([0.3])
_vals = hsgp.approximate_gram_matrix(
    kernel,
    reference,
    reference + r,
    n_basis=50,
    lengthscale=1.0,
)
print(_vals)

# %% [markdown]
# Let's use this approximation to sample functions from $\mathrm{GP}(0, k)$. In this case, rather than sampling a normal vector from the distribution with $m\times m$ covariance matrix (which may require up to $O(m^3)$ operations), we will sample $D$ independent variables from $\mathrm{Normal}(0, 1)$ vector, which is very cheap.

# %%
fig, axs = plt.subplots(len(kernels), sharex=True, sharey=True)

key = jax.random.PRNGKey(42)

for ax, (name, kernel) in zip(axs, kernels.items()):
    key, subkey = jax.random.split(key)
    n_points = 81
    n_basis = 150

    x = jnp.linspace(-1, 1, n_points)

    predict_fn = hsgp.generate_approximated_prediction_function(
        x, n_basis=n_basis, lengthscale=1.5
    )

    n_samples = 5
    for i in range(n_samples):
        new_key = jax.random.fold_in(subkey, i)
        coeffs = jax.random.normal(new_key, shape=(n_basis,))
        g_x = predict_fn(kernel, coeffs)
        ax.plot(x, g_x, alpha=1.0)

    ax.set_ylabel(name)

ax = axs[-1]
ax.set_xlabel("$x$")

for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

# %% [markdown]
# ## Modelling a simple regression problem
#
# Before we go into modelling variant abundance, which has several complications, we should try to see if our implementation is trustworty on a simple problem.
#
# We will use a simple regression problem, where there is a non-stationary function we are trying to model and some Gaussian noise.

# %%
ts = jnp.linspace(-1, 1, 101)
ts_obs = jnp.linspace(-1, 1, 200)

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)

noise = 0.3


def f(t):
    return 0.5 + 3 * t + jnp.sin(8 * t) - jnp.sin(3 * t)


ys_obs = f(ts_obs) + noise * jax.random.normal(subkey, shape=ts_obs.shape)

fig, ax = plt.subplots()

ax.plot(ts, f(ts), color="darkblue")
ax.scatter(ts_obs, ys_obs, color="black")

# %%
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from numpyro.infer import init_to_mean

N_BASIS = 25
LENGTHSCALE = 1.5
KERNEL_FN = hsgp.Matern52


def model_gp():
    gp_predict = hsgp.generate_approximated_prediction_function(
        ts_obs,
        N_BASIS,
        LENGTHSCALE,
    )

    intercept = numpyro.sample("intercept", dist.Normal(0, 5))
    slope = numpyro.sample("slope", dist.Normal(0, 5))

    gp_amplitude = numpyro.sample("amplitude", dist.Uniform(0.1, 3.0))
    gp_lengthscale = numpyro.sample("lengthscale", dist.Uniform(0.3, 2.0))

    kernel = KERNEL_FN(
        hsgp.AmplitudeLengthParams(amplitude=gp_amplitude, lengthscale=gp_lengthscale)
    )

    coeff = numpyro.sample("z_coeff", dist.Normal(jnp.zeros(N_BASIS), 1.0))

    sigma = numpyro.sample("sigma", dist.HalfNormal(1))

    # Shape the same ys_obs and ts_obs, which is (n_datapoints,)
    predictions = intercept + ts_obs * slope + gp_predict(kernel, coeff)

    with numpyro.plate("data", ts_obs.shape[0]):
        numpyro.sample("obs", dist.Normal(predictions, sigma), obs=ys_obs)


mcmc = MCMC(
    NUTS(model_gp, init_strategy=init_to_mean()),
    num_chains=4,
    num_samples=1000,
    num_warmup=1000,
)
mcmc.run(jax.random.PRNGKey(101))

mcmc.print_summary()

samples = mcmc.get_samples()


# %%
def predict_from_sample(sample):
    gp_predict = hsgp.generate_approximated_prediction_function(
        ts,
        N_BASIS,
        LENGTHSCALE,
    )
    kernel = KERNEL_FN(
        hsgp.AmplitudeLengthParams(
            amplitude=sample["amplitude"], lengthscale=sample["lengthscale"]
        )
    )

    simple_part = sample["slope"] * ts + sample["intercept"]
    gp_part = gp_predict(kernel, sample["z_coeff"])
    return simple_part + gp_part


fig, ax = plt.subplots()

ax.plot(ts, f(ts), color="darkblue")
ax.scatter(ts_obs, ys_obs, color="black")

for index in range(0, len(samples["slope"]), 30):
    vals = predict_from_sample(jax.tree.map(lambda x: x[index], samples))
    ax.plot(ts, vals, color="grey", alpha=0.3)

# %% [markdown]
# ## Modelling variant abundance (on simulated data)
#
# Let's consider the simplest scenario. We have only one city and we observe the competition between two variants (baseline and the new one) over a year.
#
# We assume that the relative fitness of the new variant over the baseline is not a constant, but rather changes as
#
# $$f(t) = f_0 + a_0 \sin( 2\pi t \cdot \nu_0),$$
#
# where $f_0$ is the "baseline" fitness, $a_0$ controls the amplitude of the departures from it and $\nu_0$ controls the frequency of the changes.
# Importantly (because of the Gaussian process regression parameterization), the time of interest will change between $[-1, 1]$. For example, if we monitor the competition over the whole year (so that $2$ is the time unit corresponding to the whole year) and we expect that $f$ should fully oscillate around 3 times in a year, we should set $\nu_0 = 3 / 2 = 1.5$.
#
# monitor the competition over the whole year (so that $1$ time unit is equal to half a year and the whole year is represented by number $2$) and we expect that the "characteristic" timescale is about a month, we expect that $\nu_0 = 0.5 \cdot (1/12)$.
#
# This choice is very convenient, as we can calculate the relative abundance of the new variant analytically. Namely, let
# $$
# \tilde x(t) = \int\limits_{-1}^t f(t') \,\mathrm{d}t' = f_0 \cdot (1+t) + \frac{a_0}{2\pi\nu_0} ( \cos(2\pi \nu_0) - \cos( 2\pi t\nu_0) )
# $$
#
# The relative abundance is then given by
# $$
# x(t) = \frac{ x_\text{start}  \exp \tilde x(t) }{ x_\text{start}  \exp \tilde x(t)  + (1-x_\text{start})} = \frac{1}{1 + \frac{1-x_\text{start}}{x_\text{start}} \exp(-\tilde x(t) ) },
# $$
#
# where $x_\text{start} = x(-1)$.
#
# (See Eq. (1) of [this paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11643185/pdf/nihpp-2024.12.02.24318334v2.pdf).)
#
#
# Let's use the equations above to simulate the ground-truth abundance:

# %%
from typing import NamedTuple


class FitnessParams(NamedTuple):
    f0: float
    a0: float
    nu0: float

    def fitness(self, t):
        return self.f0 + self.a0 * jnp.sin(2 * jnp.pi * t * self.nu0)


def remove_time_dependency(params: FitnessParams) -> FitnessParams:
    """Auxiliary function, creating a simple dynamics (without time dependency)."""
    return FitnessParams(f0=params.f0, a0=0.0, nu0=1.0)


def simulate_tilde_abundance(
    timepoints,
    params: FitnessParams,
):
    f0, a0 = params.f0, params.a0
    two_pi_nu = 2 * jnp.pi * params.nu0
    t = timepoints

    return f0 * (1 + t) + a0 * (jnp.cos(two_pi_nu) - jnp.cos(two_pi_nu * t)) / two_pi_nu


def simulate_abundance(
    timepoints,
    start,
    params: FitnessParams,
):
    x_tilde = simulate_tilde_abundance(timepoints=timepoints, params=params)
    const = jnp.log1p(-start) - jnp.log(start)

    return jnp.reciprocal(1.0 + jnp.exp(const - x_tilde))


timepoints = jnp.linspace(-1, 1, 201)

# params0 = FitnessParams(f0=3.0, a0=4.0, nu0=1 / 2)
params0 = FitnessParams(f0=3.0, a0=0.3, nu0=1 / 2)
start0 = 0.01

abundances = simulate_abundance(timepoints, start0, params0)

abundances_simple = simulate_abundance(
    timepoints,
    start0,
    remove_time_dependency(params0),
)

fig, axs = plt.subplots(1, 2, figsize=(2 * 3, 2), dpi=200)

ax = axs[0]

ax.plot(timepoints, params0.fitness(timepoints), c="darkblue")
ax.plot(timepoints, remove_time_dependency(params0).fitness(timepoints), c="maroon")

ax.set_ylabel("Fitness")
ax.set_xlabel("Time")

ax = axs[1]
ax.plot(timepoints, abundances, label="Varying fitness", c="darkblue")
ax.plot(timepoints, abundances_simple, label="Constant fitness", c="maroon")
ax.legend(frameon=False)
ax.set_ylabel("Abundance")
ax.set_xlabel("Time")

for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()

# %% [markdown]
# Let's now generate observed data points:

# %%
last_observation = 0.5
n_datapoints = 101

t_obs = jnp.linspace(-1.0, last_observation, n_datapoints)
x_obs = simulate_abundance(t_obs, start0, params0)
overdisp = 0.01

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)

y_obs = jnp.clip(
    x_obs
    + overdisp
    * jnp.sqrt(x_obs * (1 - x_obs))
    * jax.random.normal(key, shape=x_obs.shape),
    0.0,
    1.0,
)

fig, ax = plt.subplots()

ax.plot(timepoints, abundances, c="darkblue", alpha=0.5, linestyle="--")
ax.plot(timepoints, abundances_simple, c="maroon", alpha=0.5, linestyle="--")
ax.scatter(t_obs, y_obs, c="k")

ax.axvspan(last_observation * 1.001, 1.0, alpha=0.1, color="grey")
ax.set_xlabel("Time")
ax.set_ylabel("Abundance")
ax.spines[["top", "right"]].set_visible(False)


# %% [markdown]
# ### Fitting the simplest dynamical model
#
# Let's first fit the simplest selection dynamics model (under the constant fitness assumption). We have two free parameters: $f_0$ and $b_0$, with:
#
# $$
# x(t) = \mathrm{logit}^{-1}(f_0 t + b).
# $$
#
# Having observed the abundance $y = y(t)$, when the prediction is $x = x(t)$, we will use the quasi-likelihood:
#
# $$
#     q( y, x ) = \frac{y \log x + (1-y) \log(1-x)}{\psi},
# $$
# where $\psi$ is the overdispersion. We assume (and this is a serious oversimplification) that it is known (in practice, we may need to use the one from the usual fit or some methods to estimate it jointly. But let's leave this issue for another time).


# %%
def quasiloglikelihood_fn(
    predictions,
    overdispersion,
    jitter: float = 1e-3,
):
    predictions = jnp.clip(
        predictions, jitter, 1.0 - jitter
    )  # This is a hack for numerical stability

    qs = y_obs * jnp.log(predictions) + (1 - y_obs) * jnp.log1p(-predictions)
    return jnp.sum(qs) / overdispersion


def simple_pred_fn(fitness, offset):
    return jax.nn.sigmoid(fitness * t_obs + offset)


def model_simple():
    fitness = numpyro.sample("fitness", dist.Normal(0, 10))
    offset = numpyro.sample("offset", dist.Normal(0, 20))

    predictions = simple_pred_fn(fitness, offset)
    numpyro.factor("quasilikelihood", quasiloglikelihood_fn(predictions, overdisp))


mcmc = MCMC(NUTS(model_simple), num_chains=4, num_samples=1000, num_warmup=1000)
mcmc.run(jax.random.PRNGKey(101))

# %%
mcmc.print_summary()

# %%
samples = mcmc.get_samples()

key = jax.random.PRNGKey(134)
n_samples = len(samples["fitness"])
indices = jax.random.choice(key, n_samples, shape=(50,), replace=False)

# %%
fig, axs = plt.subplots(1, 2, figsize=(2 * 4, 3), dpi=150)

ax = axs[0]
ax.set_xlabel("Time")
ax.set_ylabel("Fitness")
ax.plot(timepoints, params0.fitness(timepoints), c="darkblue")
ax.plot(timepoints, remove_time_dependency(params0).fitness(timepoints), c="maroon")

for i in indices:
    f = samples["fitness"][i]
    ax.plot(timepoints, jnp.full_like(timepoints, f), c="black", alpha=0.1)

ax = axs[1]
ax.plot(timepoints, abundances, c="darkblue", alpha=0.5, linestyle="--")
ax.plot(timepoints, abundances_simple, c="maroon", alpha=0.5, linestyle="--")
ax.scatter(t_obs, y_obs, c="k", s=3)

for i in indices:
    f = samples["fitness"][i]
    o = samples["offset"][i]
    ax.plot(timepoints, jax.nn.sigmoid(f * timepoints + o), c="black", alpha=0.05)

ax.axvspan(last_observation * 1.001, 1.0, alpha=0.1, color="grey")
ax.set_xlabel("Time")
ax.set_ylabel("Abundance")

for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# ### Fitting the Gaussian process model

# %%
from numpyro.infer import init_to_mean

N_BASIS = 8
LENGTHSCALE = 1.5
KERNEL_FN = hsgp.RBFKernel


def model_gp():
    gp_predict = hsgp.generate_approximated_prediction_function(
        t_obs,
        N_BASIS,
        LENGTHSCALE,
    )

    fitness = numpyro.sample("fitness", dist.Normal(0, 10))
    offset = numpyro.sample("offset", dist.Normal(0, 20))

    gp_amplitude = numpyro.sample("amplitude", dist.Uniform(0.1, 3.0))
    gp_lengthscale = numpyro.sample("lengthscale", dist.Uniform(0.3, 2.0))

    # gp_amplitude = jnp.clip(gp_amplitude, 0.001)
    # gp_lengthscale = jnp.clip(gp_lengthscale, 0.001)

    # gp_amplitude = numpyro.sample("amplitude", dist.InverseGamma(0.9, 0.2))
    # gp_lengthscale = numpyro.sample("lengthscale", dist.InverseGamma(1.22, 0.05))

    kernel = KERNEL_FN(
        hsgp.AmplitudeLengthParams(amplitude=gp_amplitude, lengthscale=gp_lengthscale)
    )

    coeff = numpyro.sample("z_coeff", dist.Normal(jnp.zeros(N_BASIS), 1.0))

    simple_part = fitness * t_obs + offset
    gp_part = gp_predict(kernel, coeff)

    predictions = jax.nn.sigmoid(simple_part + gp_part)
    numpyro.factor("quasilikelihood", quasiloglikelihood_fn(predictions, overdisp))


mcmc = MCMC(
    NUTS(model_gp, init_strategy=init_to_mean()),
    num_chains=4,
    num_samples=1000,
    num_warmup=1000,
)
mcmc.run(jax.random.PRNGKey(101))

mcmc.print_summary()

samples = mcmc.get_samples()


def predict_from_sample(sample):
    gp_predict = hsgp.generate_approximated_prediction_function(
        timepoints,
        N_BASIS,
        LENGTHSCALE,
    )
    kernel = KERNEL_FN(
        hsgp.AmplitudeLengthParams(
            amplitude=sample["amplitude"], lengthscale=sample["lengthscale"]
        )
    )
    simple_part = sample["fitness"] * timepoints + sample["offset"]
    gp_part = gp_predict(kernel, sample["z_coeff"])
    return jax.nn.sigmoid(simple_part + gp_part)


# %%
fig, axs = plt.subplots(1, 2, figsize=(2 * 4, 3), dpi=150)

ax = axs[1]
ax.plot(timepoints, abundances, c="darkblue", alpha=0.5, linestyle="--")
ax.plot(timepoints, abundances_simple, c="maroon", alpha=0.5, linestyle="--")
ax.scatter(t_obs, y_obs, c="k", s=3)

for i in indices:
    sample = jax.tree.map(lambda x: x[i], samples)
    ax.plot(timepoints, predict_from_sample(sample), c="black", alpha=0.05)

ax.axvspan(last_observation * 1.001, 1.0, alpha=0.1, color="grey")
ax.set_xlabel("Time")
ax.set_ylabel("Abundance")

for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)

# %%
