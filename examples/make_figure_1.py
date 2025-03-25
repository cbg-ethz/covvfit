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

# # Analysis of the full dataset notebook
#
# This notebook shows how to estimate growth advantages by fiting the model within the quasimultinomial framework, on the whole dataset

# +
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import yaml
from pathlib import Path

from covvfit import plot, preprocess
from covvfit import quasimultinomial as qm

plot_ts = plot.timeseries
# -


# ## Load and preprocess data
#
# We start by loading the data:

# +
DATA_DIR = Path("../data/main/")
DATA_PATH = DATA_DIR / "deconvolved.csv"
VAR_DATES_PATH = DATA_DIR / "var_dates.yaml"


data = pd.read_csv(DATA_PATH, sep="\t")
data.head()

# +
# %matplotlib inline
import matplotlib.pyplot as plt

# Set default DPI for high-resolution plots
plt.rcParams["figure.dpi"] = 150  # Adjust to 150, 200, or more for higher resolution

# +
# Load the YAML file
with open(VAR_DATES_PATH, "r") as file:
    var_dates_data = yaml.safe_load(file)

# Access the var_dates data
var_dates = var_dates_data["var_dates"]


data_wide = data.pivot_table(
    index=["date", "location"], columns="variant", values="proportion", fill_value=0
).reset_index()
data_wide = data_wide.rename(columns={"date": "time", "location": "city"})

# Define the list with cities:
cities = [
    "Zürich (ZH)",
    "Altenrhein (SG)",
    "Laupen (BE)",
    "Lugano (TI)",
    "Chur (GR)",
    "Genève (GE)",
]

## Set limit times for modeling

# max_date = pd.to_datetime(data_wide["time"]).max()
max_date = pd.to_datetime("2025-01-10")
# delta_time = pd.Timedelta(days=1250)
start_date = pd.to_datetime("2021-05-01")

# Print the data frame
data_wide.head()
# -

[data_wide.groupby("city").time.min(), data_wide.groupby("city").time.max()]

# Now we look at the variants in the data and define the variants of interest:

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


variants_full = [
    "B.1.1.7",
    "B.1.351",
    #     "P.1",
    "B.1.617.2",
    "BA.1",
    "BA.2",
    "BA.4",
    "BA.5",
    "BA.2.75",
    "BQ.1.1",
    "XBB.1.5",
    "XBB.1.9",
    "XBB.1.16",
    "XBB.2.3",
    "EG.5",
    "BA.2.86",
    "JN.1",
    "KP.2",
    "KP.3",
    "XEC",
]

variants_investigated = [
    # "B.1.1.7",
    "B.1.617.2",
    "BA.1",
    "BA.2",
    "BA.4",
    "BA.5",
    "BA.2.75",
    "BQ.1.1",
    "XBB.1.5",
    "XBB.1.9",
    "XBB.1.16",
    "XBB.2.3",
    "EG.5",
    "BA.2.86",
    "JN.1",
    "KP.2",
    "KP.3",
    "XEC",
]  # Variants found in the data, which we focus on in this analysis
variants_other = [
    i for i in variants_full if i not in variants_investigated
]  # Variants not of interest
# -

# Apart from the variants of interest, we define the "other" variant, which artificially merges all the other variants into one. This allows us to model the data as a compositional time series, i.e., the sum of abundances of all "variants" is normalized to one.

# +
variants_effective = ["other"] + variants_investigated
data_full = preprocess.preprocess_df(
    data_wide,
    cities,
    variants_full,
    date_min=start_date,
    date_max=max_date,
    zero_date=start_date,
)

data_full["other"] = data_full[variants_other].sum(axis=1)
data_full[variants_effective] = data_full[variants_effective].div(
    data_full[variants_effective].sum(axis=1), axis=0
)

# +
ts_lst, ys_effective = preprocess.make_data_list(
    data_full, cities=cities, variants=variants_effective
)

# Scale the time for numerical stability
time_scaler = preprocess.TimeScaler()
ts_lst_scaled = time_scaler.fit_transform(ts_lst)
# -


# ### Count number of samples

print(", ".join(cities))

print(f"Total number of samples used: {data_full.shape[0]}")

for i, city in enumerate(cities):
    print(f"{ts_lst[i].shape[0]} for {city}", end=", ")

print(f"Datapoints: {sum([i.flatten().shape[0] for i in ys_effective])}")

# count the number of samples without filtering for unexplained

# +
data_full2 = preprocess.preprocess_df(
    data_wide,
    cities,
    variants_full,
    date_min=start_date,
    zero_date=start_date,
    undetermined_thresh=1,
)

data_full2["other"] = data_full2[variants_other].sum(axis=1)
data_full2[variants_effective] = data_full2[variants_effective].div(
    data_full2[variants_effective].sum(axis=1), axis=0
)
ts_lst_full, _ = preprocess.make_data_list(
    data_full, cities=cities, variants=variants_effective
)
print(f"Total number of samples used: {data_full2.shape[0]}")
for i, city in enumerate(cities):
    print(f"{ts_lst_full[i].shape[0]} for {city}", end=", ")
# -

# ## Fit the quasimultinomial model
#
# Now we fit the quasimultinomial model, which allows us to find the maximum quasilikelihood estimate of the parameters:

# +
# %%time

# no priors
loss = qm.construct_total_loss(
    ys=ys_effective,
    ts=ts_lst_scaled,
    average_loss=False,  # Do not average the loss over the data points, so that the covariance matrix shrinks with more and more data added
)

n_variants_effective = len(variants_effective)

# initial parameters
theta0 = qm.construct_theta0(n_cities=len(cities), n_variants=n_variants_effective)

# Run the optimization routine
solution = qm.jax_multistart_minimize(loss, theta0, n_starts=1)

theta_star = solution.x  # The maximum quasilikelihood estimate

print(
    f"Relative fitness advantages: \n",
    qm.get_relative_growths(theta_star, n_variants=n_variants_effective),
)
# -

# ## Confidence intervals of the growth advantages
#
# To obtain confidence intervals, we will take into account overdispersion. To do this, we need to compare the predictions with the observed values. Then, we can use overdispersion to attempt to correct the covariance matrix and obtain the confidence intervals.

# +
## compute fitted values
ys_fitted = qm.fitted_values(
    ts_lst_scaled, theta=theta_star, cities=cities, n_variants=n_variants_effective
)

## compute covariance matrix
covariance = qm.get_covariance(loss, theta_star)

overdispersion_tuple = qm.compute_overdispersion(
    observed=ys_effective,
    predicted=ys_fitted,
    epsilon=1e-3,
)

overdisp_fixed = overdispersion_tuple.overall

print(f"Overdispersion factor: {float(overdisp_fixed):.3f}.")
print("Note that values lower than 1 signify underdispersion.")

## scale covariance by overdisp
covariance_scaled = overdisp_fixed * covariance

## compute standard errors and confidence intervals of the estimates
standard_errors_estimates = qm.get_standard_errors(covariance_scaled)
confints_estimates = qm.get_confidence_intervals(
    theta_star, standard_errors_estimates, confidence_level=0.95
)


print("\n\nRelative fitness advantages:")
for variant, m, l, u in zip(
    variants_effective[1:],
    (
        qm.get_relative_growths(theta_star, n_variants=n_variants_effective)
        - time_scaler.t_min
    )
    / (time_scaler.t_max - time_scaler.t_min),
    (
        qm.get_relative_growths(confints_estimates[0], n_variants=n_variants_effective)
        - time_scaler.t_min
    )
    / (time_scaler.t_max - time_scaler.t_min),
    (
        qm.get_relative_growths(confints_estimates[1], n_variants=n_variants_effective)
        - time_scaler.t_min
    )
    / (time_scaler.t_max - time_scaler.t_min),
):
    print(f"  {variant}: {float(m):.2f} ({float(l):.2f} – {float(u):.2f})")
# -


# We can propagate this uncertainty to the observed values. Let's generate confidence bands around the fitted lines and predict the future behaviour.

# +
# %%time

ys_fitted_confint = qm.get_confidence_bands_logit(
    theta_star,
    n_variants=n_variants_effective,
    ts=ts_lst_scaled,
    covariance=covariance_scaled,
    confidence_level=0.95,
)


## compute predicted values and confidence bands
horizon = 90
ts_pred_lst = [jnp.arange(horizon + 1) + tt.max() for tt in ts_lst]
ts_pred_lst_scaled = time_scaler.transform(ts_pred_lst)

ys_pred = qm.fitted_values(
    ts_pred_lst_scaled, theta=theta_star, cities=cities, n_variants=n_variants_effective
)
ys_pred_confint = qm.get_confidence_bands_logit(
    theta_star,
    n_variants=n_variants_effective,
    ts=ts_pred_lst_scaled,
    covariance=covariance_scaled,
)
# -

# ## Plot
#
# Finally, we plot the abundance data and the model predictions. Note that the 0th element in each array corresponds to the artificial "other" variant and we decided to plot only the explicitly defined variants.

# +
colors = [plot_ts.COLORS_COVSPECTRUM[var] for var in variants_investigated]
import matplotlib.dates as mdates


figure_spec = plot.arrange_into_grid(len(cities), axsize=(4, 1.5), dpi=350, wspace=1)


def plot_city(ax, i: int) -> None:
    def remove_0th(arr):
        """We don't plot the artificial 0th variant 'other'."""
        return arr[:, 1:]

    # Plot fits in observed and unobserved time intervals.
    plot_ts.plot_fit(ax, ts_lst[i], remove_0th(ys_fitted[i]), colors=colors)
    plot_ts.plot_fit(
        ax, ts_pred_lst[i], remove_0th(ys_pred[i]), colors=colors, linestyle="--"
    )

    plot_ts.plot_confidence_bands(
        ax,
        ts_lst[i],
        jax.tree.map(remove_0th, ys_fitted_confint[i]),
        colors=colors,
    )
    plot_ts.plot_confidence_bands(
        ax,
        ts_pred_lst[i],
        jax.tree.map(remove_0th, ys_pred_confint[i]),
        colors=colors,
    )

    # Plot the data points
    plot_ts.plot_data(ax, ts_lst[i], remove_0th(ys_effective[i]), colors=colors)

    # Plot the complements
    plot_ts.plot_complement(ax, ts_lst[i], remove_0th(ys_fitted[i]), alpha=0.3)

    plot_ts.AdjustXAxisForTime(start_date)(ax)
    tick_positions = [0, 0.5, 1]
    tick_labels = ["0%", "50%", "100%"]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_ylabel("Relative abundances")
    ax.set_title(cities[i])


figure_spec.map(plot_city, range(len(cities)))
# -

(pd.to_datetime(["2021-08-01", "2025-03-01"]) - start_date).days

# +
import matplotlib.pyplot as plt

# Example data (adjust as needed)
colors = [plot_ts.COLORS_COVSPECTRUM[var] for var in variants_investigated]

# Create the figure and subplots
fig, axes = plt.subplots(3, 2, figsize=(10, 5), sharey="none")
axes = axes.flatten()
for i in range(6):
    plot_city(axes[i], i)  # Assuming plot_city modifies the axes
    axes[i].set_xlim((pd.to_datetime(["2021-09-01", "2025-03-15"]) - start_date).days)

# Create a custom legend
legend_elements = [
    plt.Line2D([0], [0], color=color, lw=4, label=variant)
    for color, variant in zip(colors, variants_investigated)
]

# Add the legend to the right outside the plot
fig.legend(
    handles=legend_elements,
    loc="center left",  # Align the legend to the left of the bounding box
    bbox_to_anchor=(1, 0.5),  # Position it to the right of the figure
    title="Variants",
)

axes[0].set_title("Zurich")
axes[1].set_title("Altenrhein")
axes[2].set_title("Laupen")
axes[3].set_title("Lugano")
axes[4].set_title("Chur")
axes[5].set_title("Geneva")

axes[0].set_ylabel("")
axes[1].set_ylabel("")
axes[3].set_ylabel("")
axes[4].set_ylabel("")
axes[5].set_ylabel("")

# Adjust the layout to make space for the legend

plt.tight_layout()
# Show the plot
plt.show()


# -


# ## Plot matrix of growth advantage estimates


# +
def make_relative_growths(theta_star):
    relative_growths = (
        qm.get_relative_growths(theta_star, n_variants=n_variants_effective)
        - time_scaler.t_min
    ) / (time_scaler.t_max - time_scaler.t_min)
    relative_growths = jnp.concat([jnp.array([0]), relative_growths])
    relative_growths = relative_growths * 7

    pairwise_diff = jnp.expand_dims(relative_growths, axis=1) - jnp.expand_dims(
        relative_growths, axis=0
    )

    return pairwise_diff


pairwise_diffs = make_relative_growths(theta_star)
jacob = jax.jacobian(make_relative_growths)(theta_star)
standerr_relgrowths = qm.get_standard_errors(covariance_scaled, jacob)
relgrowths_confint = qm.get_confidence_intervals(
    make_relative_growths(theta_star), standerr_relgrowths, 0.95
)
# -


relgrowths_confint[0].shape

# +
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable


fitness_df = pd.DataFrame(pairwise_diffs * 100)
# rename columns and rows
fitness_df.columns = variants_effective
fitness_df.index = variants_effective

# format, remove first row and last col, keep only lower triangle and diagonal
fitness_df = fitness_df.iloc[1:, :]
fitness_df = fitness_df.iloc[:, :-1]
mask = np.triu(np.ones(fitness_df.shape), k=1).astype(bool)
fitness_df = fitness_df.mask(mask, np.nan, inplace=False)
fitness_df.iloc[1:, 0] = np.nan
fitness_df.iloc[3:, 1] = np.nan
fitness_df.iloc[3:, 2] = np.nan
fitness_df.iloc[5:, 3] = np.nan
fitness_df.iloc[7:, 4] = np.nan
fitness_df.iloc[9:, 5] = np.nan
fitness_df.iloc[11:, 6] = np.nan
fitness_df.iloc[12:, 7] = np.nan
fitness_df.iloc[14:, 8] = np.nan
fitness_df.iloc[14:, 9] = np.nan
fitness_df.iloc[14:, 10] = np.nan
fitness_df.iloc[14:, 11] = np.nan
fitness_df.iloc[14:, 12] = np.nan
fitness_df.iloc[14:, 13] = np.nan


fitness_df = fitness_df.iloc[1:, 1:]

ax = sns.heatmap(fitness_df, cmap="Reds", annot=True, fmt=".0f", cbar=True)
ax.set_title("Weekly Fitness Advantage (%)")


# +
# Assuming fitness_df and relgrowths_confint are already defined
lower_conf_df = pd.DataFrame(
    relgrowths_confint[0] * 100, columns=variants_effective, index=variants_effective
)
upper_conf_df = pd.DataFrame(
    relgrowths_confint[1] * 100, columns=variants_effective, index=variants_effective
)

# Apply the same masking and slicing as fitness_df
lower_conf_df = lower_conf_df.iloc[1:, :-1]
upper_conf_df = upper_conf_df.iloc[1:, :-1]

mask = np.triu(np.ones(lower_conf_df.shape), k=1).astype(bool)
lower_conf_df = lower_conf_df.mask(mask, np.nan, inplace=False)
upper_conf_df = upper_conf_df.mask(mask, np.nan, inplace=False)

lower_conf_df.iloc[1:, 0] = np.nan
upper_conf_df.iloc[1:, 0] = np.nan
lower_conf_df.iloc[3:, 1] = np.nan
upper_conf_df.iloc[3:, 1] = np.nan
lower_conf_df.iloc[3:, 2] = np.nan
upper_conf_df.iloc[3:, 2] = np.nan
lower_conf_df.iloc[5:, 3] = np.nan
upper_conf_df.iloc[5:, 3] = np.nan
lower_conf_df.iloc[7:, 4] = np.nan
upper_conf_df.iloc[7:, 4] = np.nan
lower_conf_df.iloc[9:, 5] = np.nan
upper_conf_df.iloc[9:, 5] = np.nan
lower_conf_df.iloc[11:, 6] = np.nan
upper_conf_df.iloc[11:, 6] = np.nan
lower_conf_df.iloc[12:, 7] = np.nan
upper_conf_df.iloc[12:, 7] = np.nan
lower_conf_df.iloc[14:, 8] = np.nan
upper_conf_df.iloc[14:, 8] = np.nan
lower_conf_df.iloc[14:, 9] = np.nan
upper_conf_df.iloc[14:, 9] = np.nan
lower_conf_df.iloc[14:, 10] = np.nan
upper_conf_df.iloc[14:, 10] = np.nan
lower_conf_df.iloc[14:, 11] = np.nan
upper_conf_df.iloc[14:, 11] = np.nan
lower_conf_df.iloc[14:, 12] = np.nan
upper_conf_df.iloc[14:, 12] = np.nan
lower_conf_df.iloc[14:, 13] = np.nan
upper_conf_df.iloc[14:, 13] = np.nan

lower_conf_df = lower_conf_df.iloc[1:, 1:]
upper_conf_df = upper_conf_df.iloc[1:, 1:]

# +
# Define number of rows and columns for subplots
num_cols = min(4, fitness_df.shape[1])  # Limit columns per row to 4
num_rows = -(-fitness_df.shape[1] // num_cols)  # Ceiling division to get rows needed

# Create subplots
fig, axes = plt.subplots(
    nrows=num_rows, ncols=num_cols, figsize=(num_cols * 3, num_rows * 3), sharey="none"
)
axes = axes.flatten()  # Flatten in case of multiple rows

for ax, col in zip(axes, fitness_df.columns):
    y_values = fitness_df[col].dropna()
    x_labels = y_values.index
    x_positions = np.arange(len(x_labels))

    lower_bounds = lower_conf_df[col].dropna()
    upper_bounds = upper_conf_df[col].dropna()

    yerr_lower = y_values - lower_bounds
    yerr_upper = upper_bounds - y_values

    ax.errorbar(
        x_positions, y_values, yerr=[yerr_lower, yerr_upper], fmt="o", capsize=5
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_title(f"Reference variant: {col}")
    ax.grid(True)

    ax.set_ylabel("Relative Growth (%)")

# Hide unused subplots if any
for ax in axes[len(fitness_df.columns) :]:
    ax.axis("off")

plt.tight_layout()
plt.show()

# +
fig, axes = plt.subplots(
    ncols=fitness_df.shape[1], figsize=(fitness_df.shape[1] * 3, 6), sharey=True
)

if fitness_df.shape[1] == 1:
    axes = [axes]  # Ensure axes is iterable if there's only one column

for ax, col in zip(axes, fitness_df.columns):
    y_values = fitness_df[col].dropna()
    x_labels = y_values.index
    x_positions = np.arange(len(x_labels))

    lower_bounds = (
        relgrowths_confint[0][x_positions, fitness_df.columns.get_loc(col)] * 100
    )
    upper_bounds = (
        relgrowths_confint[1][x_positions, fitness_df.columns.get_loc(col)] * 100
    )

    yerr_lower = np.clip(y_values - lower_bounds, 0, None)
    yerr_upper = np.clip(upper_bounds - y_values, 0, None)

    ax.errorbar(
        x_positions, y_values, yerr=[yerr_lower, yerr_upper], fmt="o", capsize=5
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_title(col)
    ax.set_xlabel("Variant")

axes[0].set_ylabel("Relative Growth (%)")
plt.tight_layout()
plt.show()


# -

# ## Plot with discrete time model


# +
def make_discrete_growth(theta):
    return jnp.exp(make_relative_growths(theta)) - 1


pairwise_diffs_discrete = make_discrete_growth(theta_star)
jacob_discrete = jax.jacobian(make_discrete_growth)(theta_star)
standerr_relgrowths_discrete = qm.get_standard_errors(covariance_scaled, jacob)
relgrowths_confint_discrete = qm.get_confidence_intervals(
    make_discrete_growth(theta_star), standerr_relgrowths_discrete
)

fitness_df = pd.DataFrame(pairwise_diffs_discrete * 100)
# rename columns and rows
fitness_df.columns = variants_effective
fitness_df.index = variants_effective

# format, remove first row and last col, keep only lower triangle and diagonal
fitness_df = fitness_df.iloc[1:, :]
fitness_df = fitness_df.iloc[:, :-1]
mask = np.triu(np.ones(fitness_df.shape), k=1).astype(bool)
fitness_df = fitness_df.mask(mask, np.nan, inplace=False)
fitness_df.iloc[1:, 0] = np.nan
fitness_df.iloc[3:, 1] = np.nan
fitness_df.iloc[3:, 2] = np.nan
fitness_df.iloc[5:, 3] = np.nan
fitness_df.iloc[7:, 4] = np.nan
fitness_df.iloc[9:, 5] = np.nan
fitness_df.iloc[11:, 6] = np.nan
fitness_df.iloc[12:, 7] = np.nan
fitness_df.iloc[14:, 8] = np.nan
fitness_df.iloc[14:, 9] = np.nan
fitness_df.iloc[14:, 10] = np.nan
fitness_df.iloc[14:, 11] = np.nan
fitness_df.iloc[14:, 12] = np.nan
fitness_df.iloc[14:, 13] = np.nan


fitness_df = fitness_df.iloc[1:, 1:]

ax = sns.heatmap(fitness_df, cmap="Reds", annot=True, fmt=".0f", cbar=True)
ax.set_title("Discrete Time Model Weekly Fitness Advantage (%)")

# -

# ## Look at overdispersion for different thresholds of epsilon
#
# Let's check if our estimate of overdispersion is stable in the vicinity of the selected epsilon=0.001 threshold.

# +
epsilons = np.logspace(-5, -1, 100)

# Compute results for each epsilon
overdispersion_results = [
    qm.compute_overdispersion(ys_effective, ys_fitted, epsilon=eps) for eps in epsilons
]

# Extract overall and cities data
overalls = [res.overall for res in overdispersion_results]
cities_res = np.array([res.cities for res in overdispersion_results])

# Plotting
fig, axes = plt.subplots(1, 1, figsize=(8, 4))
axes = [axes]

# Overall vs Epsilon
axes[0].plot(epsilons, overalls, label="Overall", color="black", linestyle="-")
axes[0].set_title("Overdispersion vs Outlier Threshold")
axes[0].set_xlabel(r"$\kappa$")
axes[0].set_ylabel(r"$\hat\sigma^2$")
axes[0].grid(True)
axes[0].set_xscale("log")
# axes[0].set_yscale('log')


# # Cities vs Epsilon
# for city_idx in range(cities_res.shape[1]):
#     axes[0].plot(epsilons, cities_res[:, city_idx], label=f"{cities[city_idx]}")
# axes[0].legend()

axes[0].set_ylim(0.1, 0.4)

plt.tight_layout()
plt.show()
# -

0.04 / 0.16
