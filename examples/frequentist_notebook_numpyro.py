# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: jax
#     language: python
#     name: jax
# ---

# +
# import os
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"  # Set this to the number of CPU cores



import jax
import jax.numpy as jnp

import pandas as pd

import numpy as np

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import expit
from scipy.stats import norm

import yaml

import covvfit._frequentist as freq
import covvfit._preprocess_abundances as prec
import covvfit.plotting._timeseries as plot_ts

from covvfit import quasimultinomial as qm

import numpyro

# -


# # Load and preprocess data

# +
DATA_PATH = "../../LolliPop/lollipop_test_noisy/deconvolved.csv"
VAR_DATES_PATH = "../../LolliPop/lollipop_test_noisy/var_dates.yaml"

data = pd.read_csv(DATA_PATH, sep="\t")
data.head()


# Load the YAML file
with open(VAR_DATES_PATH, "r") as file:
    var_dates_data = yaml.safe_load(file)

# Access the var_dates data
var_dates = var_dates_data["var_dates"]
# -

data_wide = data.pivot_table(
    index=["date", "location"], columns="variant", values="proportion", fill_value=0
).reset_index()
data_wide = data_wide.rename(columns={"date": "time", "location": "city"})
data_wide.head()

# +
## Set limit times for modeling

max_date = pd.to_datetime(data_wide["time"]).max()
delta_time = pd.Timedelta(days=240)
start_date = max_date - delta_time


# +
# Convert the keys to datetime objects for comparison
var_dates_parsed = {
    pd.to_datetime(date): variants for date, variants in var_dates.items()
}


# Function to find the latest matching date in var_dates
def match_date(start_date):
    start_date = pd.to_datetime(start_date)
    closest_date = max(date for date in var_dates_parsed if date <= start_date)
    return closest_date, var_dates_parsed[closest_date]


variants_full = match_date(start_date + delta_time)[1]

variants = ["KP.2", "KP.3", "XEC"]

variants_other = [i for i in variants_full if i not in variants]
# -

cities = list(data_wide["city"].unique())

variants2 = ["other"] + variants
data2 = prec.preprocess_df(
    data_wide, cities, variants_full, date_min=start_date, zero_date=start_date
)

# +
data2["other"] = data2[variants_other].sum(axis=1)
data2[variants2] = data2[variants2].div(data2[variants2].sum(axis=1), axis=0)

ts_lst, ys_lst = prec.make_data_list(data2, cities, variants2)
ts_lst, ys_lst2 = prec.make_data_list(data2, cities, variants)

t_max = max([x.max() for x in ts_lst])
t_min = min([x.min() for x in ts_lst])

ts_lst_scaled = [(x - t_min) / (t_max - t_min) for x in ts_lst]
# -


# # fit in jax

# +
# %%time


# no priors
loss = qm.construct_total_loss(
    ys=[
        y.T for y in ys_lst
    ],  # Recall that the input should be (n_timepoints, n_variants)
    ts=ts_lst_scaled,
    average_loss=False,  # Do not average the loss over the data points, so that the covariance matrix shrinks with more and more data added
)

# initial parameters
theta0 = qm.construct_theta0(n_cities=len(cities), n_variants=len(variants2))

# Run the optimization routine
solution = qm.jax_multistart_minimize(loss, theta0, n_starts=10)
# -

# ## Make fitted values and confidence intervals

# +
## compute fitted values
y_fit_lst = qm.fitted_values(
    ts_lst_scaled, theta=solution.x, cities=cities, n_variants=len(variants2)
)

## compute covariance matrix
covariance = qm.get_covariance(loss, solution.x)

## compute overdispersion
# TODO(Pawel, David): Port compute_overdispersion to `qm`
pearson_r_lst, overdisp_list, overdisp_fixed = freq.compute_overdispersion(
    ys_lst2, y_fit_lst, cities
)

## scale covariance by overdisp
covariance_scaled = overdisp_fixed * covariance

## compute standard errors and confidence intervals of the estimates
standard_errors_estimates = qm.get_standard_errors(covariance_scaled)
confints_estimates = qm.get_confidence_intervals(solution.x, standard_errors_estimates)

## compute confidence intervals of the fitted values on the logit scale and back transform
y_fit_lst_confint = qm.get_confidence_bands_logit(
    solution.x, len(variants2), ts_lst_scaled, covariance_scaled
)

## compute predicted values and confidence bands
horizon = 60
ts_pred_lst = [jnp.arange(horizon + 1) + tt.max() for tt in ts_lst]
ts_pred_lst_scaled = [(x - t_min) / (t_max - t_min) for x in ts_pred_lst]
y_pred_lst = qm.fitted_values(
    ts_pred_lst_scaled, theta=solution.x, cities=cities, n_variants=len(variants2)
)
y_pred_lst_confint = qm.get_confidence_bands_logit(
    solution.x, len(variants2), ts_pred_lst_scaled, covariance_scaled
)


# -

# # Numpyro

# +
model = qm.construct_model(
    ys=[
        y.T for y in ys_lst
    ],  # Recall that the input should be (n_timepoints, n_variants)
    ts=ts_lst_scaled,
    overdispersion=overdisp_fixed, #use the overdispersion from the quasilikelihood fit 
    sigma_offset=100.0,
)

from numpyro.infer import MCMC, NUTS

mcmc = MCMC(NUTS(model), num_chains=4, num_samples=2000, num_warmup=2000)
mcmc.run(jax.random.PRNGKey(42))
# -

mcmc.print_summary()

import arviz as az
idata = az.from_numpyro(mcmc)

az.plot_trace(idata)
plt.tight_layout()
plt.show()


overdisp_fixed

# +
np.random.seed(42)
fitted_pred = []
fitted_pred_pred = []

for i in range(100):
    idx = np.random.randint(4000)
    samples = mcmc.get_samples()
    g1 = samples['relative_growths'][idx,:]
    o1 = samples['relative_offsets'][idx,]
    t1 = qm.construct_theta(g1, o1)
    f_lst_1 = qm.fitted_values(
        ts_lst_scaled, theta=t1, cities=cities, n_variants=len(variants2)
    )
    f_lst_2 = qm.fitted_values(
        ts_pred_lst_scaled, theta=t1, cities=cities, n_variants=len(variants2)
    )

    fitted_pred.append(f_lst_1)
    fitted_pred_pred.append(f_lst_2)
# -

y_fit_lst = qm.fitted_values(
    ts_lst_scaled, theta=solution.x, cities=cities, n_variants=len(variants2)
)

# ## Plotting functions

plot_fit = plot_ts.plot_fit
plot_complement = plot_ts.plot_complement
plot_data = plot_ts.plot_data
plot_confidence_bands = plot_ts.plot_confidence_bands

# ## Plot

# +
colors_covsp = plot_ts.colors_covsp
colors = [colors_covsp[var] for var in variants]
fig, axes_tot = plt.subplots(4, 2, figsize=(15, 10))
axes_flat = axes_tot.flatten()

for i, city in enumerate(cities):
    ax = axes_flat[i]
    # plot fitted and predicted values
#     plot_fit(ax, ts_lst[i], f_lst_1[i], variants, colors)
    for f in fitted_pred:
        plot_fit(ax, ts_lst[i], f[i], variants, colors, alpha=.05)
    for f in fitted_pred_pred:
        plot_fit(ax, ts_pred_lst[i], f[i], variants, colors, alpha=.05)
#     plot_fit(ax, ts_pred_lst[i], y_pred_lst[i], variants, colors, linetype="--")

    #     # plot 1-fitted and predicted values
#     plot_complement(ax, ts_lst[i], y_fit_lst[i], variants)
    #     plot_complement(ax, ts_pred_lst[i], y_pred_lst[i], variants, linetype="--")
    # plot raw deconvolved values
    plot_data(ax, ts_lst[i], ys_lst2[i], variants, colors)
#     make confidence bands and plot them
    conf_bands = y_fit_lst_confint[i]
    plot_confidence_bands(
        ax,
        ts_lst[i],
        {"lower": conf_bands[0], "upper": conf_bands[1]},
        variants,
        colors,
    )

    pred_bands = y_pred_lst_confint[i]
    plot_confidence_bands(
        ax,
        ts_pred_lst[i],
        {"lower": pred_bands[0], "upper": pred_bands[1]},
        variants,
        colors,
    )

    # format axes and title
    def format_date(x, pos):
        return plot_ts.num_to_date(x, date_min=start_date)

    date_formatter = ticker.FuncFormatter(format_date)
    ax.xaxis.set_major_formatter(date_formatter)
    tick_positions = [0, 0.5, 1]
    tick_labels = ["0%", "50%", "100%"]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_ylabel("relative abundances")
    ax.set_title(city)

fig.tight_layout()
fig.show()