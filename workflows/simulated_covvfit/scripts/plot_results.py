from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

snakemake: Any = globals().get("snakemake")
if snakemake is None:
    raise RuntimeError("This script is intended to be executed by Snakemake.")


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 2:
        return np.nan
    return float(np.corrcoef(x_arr[valid], y_arr[valid])[0, 1])


def _pick_levels(values: list[float], n: int) -> list[float]:
    if len(values) <= n:
        return values
    idx = np.linspace(0, len(values) - 1, n, dtype=int)
    return [values[i] for i in idx]


cfg = snakemake.config
plot_cfg = cfg["plotting"]
jpeg_dpi = int(plot_cfg["jpeg_dpi"])

df = pd.read_csv(str(snakemake.input[0]))
df["index"] = pd.to_datetime(df["index"])
df["mval2"] = df["mval"].round(2)
df["bw2"] = df["bw"].round(0).astype(int)

agg = (
    df.groupby(["index", "variant", "mval", "mval2", "bw", "bw2"], as_index=False)[
        ["ground", "estimate", "estimate2", "mval_day"]
    ]
    .mean()
    .sort_values(["mval", "bw", "variant", "index"])
)

corr = (
    agg.groupby(["mval", "bw"])
    .apply(
        lambda g: pd.Series(
            {
                "cor": _safe_corr(g["estimate"], g["ground"]),
                "cor2": _safe_corr(g["estimate2"], g["ground"]),
            }
        )
    )
    .reset_index()
)
agg = agg.merge(corr, on=["mval", "bw"], how="left")

long_df = agg.melt(
    id_vars=["index", "variant", "mval", "mval2", "bw", "bw2", "cor", "cor2"],
    value_vars=["ground", "estimate", "estimate2"],
    var_name="series",
    value_name="value",
)
long_df["series"] = long_df["series"].map(
    {
        "ground": "ground truth",
        "estimate": "Covvfit (ns=1)",
        "estimate2": "Covvfit (ns=sample size)",
    }
)

sns.set_theme(style="whitegrid", context="talk")

# Full facet plot (analogous to the full R panel figure).
full = sns.relplot(
    data=long_df,
    kind="line",
    x="index",
    y="value",
    hue="variant",
    style="series",
    row="mval2",
    col="bw2",
    facet_kws={"sharex": True, "sharey": True},
    linewidth=1.2,
    height=2.3,
    aspect=1.4,
)
full.set_axis_labels("", "Relative Abundance")
for ax in full.axes.flat:
    ax.tick_params(axis="x", rotation=45)
full.figure.subplots_adjust(bottom=0.08, hspace=0.28, wspace=0.12)
full.figure.savefig(str(snakemake.output[0]), bbox_inches="tight")
full.figure.savefig(str(snakemake.output[1]), bbox_inches="tight", dpi=jpeg_dpi)
plt.close(full.figure)

# Subset panel with R^2 annotation.
subset_mvals = _pick_levels(
    sorted(agg["mval2"].drop_duplicates().to_list()),
    int(plot_cfg["subset_n_missing_rates"]),
)
subset_bws = _pick_levels(
    sorted(agg["bw2"].drop_duplicates().to_list()),
    int(plot_cfg["subset_n_sample_sizes"]),
)
subset = long_df[
    long_df["mval2"].isin(subset_mvals) & long_df["bw2"].isin(subset_bws)
].copy()

subset_corr = agg.groupby(["mval2", "bw2"], as_index=False).first()[
    ["mval2", "bw2", "cor", "cor2"]
]

g = sns.relplot(
    data=subset,
    kind="line",
    x="index",
    y="value",
    hue="variant",
    style="series",
    row="bw2",
    col="mval2",
    facet_kws={"sharex": True, "sharey": True},
    linewidth=1.3,
    height=2.6,
    aspect=1.3,
)
g.set_axis_labels("Date", "Relative Abundance")

for ax in g.axes.flat:
    ax.tick_params(axis="x", rotation=45)

bw_levels = sorted(subset["bw2"].unique())
mval_levels = sorted(subset["mval2"].unique())
for row_idx, col_idx in np.ndindex(len(bw_levels), len(mval_levels)):
    ax = g.axes[row_idx, col_idx]
    row = subset_corr[
        (subset_corr["bw2"] == bw_levels[row_idx])
        & (subset_corr["mval2"] == mval_levels[col_idx])
    ]
    if len(row) == 1:
        r2_1 = (
            float(row["cor"].iloc[0]) ** 2
            if np.isfinite(row["cor"].iloc[0])
            else np.nan
        )
        r2_2 = (
            float(row["cor2"].iloc[0]) ** 2
            if np.isfinite(row["cor2"].iloc[0])
            else np.nan
        )
        ax.text(
            0.03,
            0.95,
            f"R2 ns=1: {r2_1:.3f}\nR2 ns=n: {r2_2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

g.figure.subplots_adjust(bottom=0.1, hspace=0.22, wspace=0.12)
g.figure.savefig(str(snakemake.output[2]), bbox_inches="tight")
plt.close(g.figure)

# Missingness panel.
missing = (
    agg.groupby(["index", "mval2", "bw2"], as_index=False)[["mval_day"]]
    .mean()
    .sort_values(["mval2", "bw2", "index"])
)
miss_plot = sns.relplot(
    data=missing,
    kind="line",
    x="index",
    y="mval_day",
    hue="bw2",
    col="mval2",
    col_wrap=4,
    linewidth=1.5,
    height=2.3,
    aspect=1.4,
)
miss_plot.set_axis_labels("Date", "Missingness rate")
for ax in miss_plot.axes.flat:
    ax.tick_params(axis="x", rotation=45)
miss_plot.figure.subplots_adjust(bottom=0.18)
miss_plot.figure.savefig(str(snakemake.output[3]), bbox_inches="tight")
plt.close(miss_plot.figure)

# Heatmaps of R^2 values.
heat_df = corr.copy()
heat_df["mval2"] = heat_df["mval"].round(2)
heat_df["bw2"] = heat_df["bw"].round(0).astype(int)
heat_df["r2_unweighted"] = heat_df["cor"] ** 2
heat_df["r2_weighted"] = heat_df["cor2"] ** 2

pivot1 = heat_df.pivot(index="mval2", columns="bw2", values="r2_unweighted")
pivot2 = heat_df.pivot(index="mval2", columns="bw2", values="r2_weighted")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
sns.heatmap(pivot1, annot=True, fmt=".3f", cmap="viridis", cbar=True, ax=axs[0])
axs[0].set_title("R2: Covvfit (ns=1)")
axs[0].set_xlabel("Sample size")
axs[0].set_ylabel("Missing rate")

sns.heatmap(pivot2, annot=True, fmt=".3f", cmap="viridis", cbar=True, ax=axs[1])
axs[1].set_title("R2: Covvfit (ns=sample size)")
axs[1].set_xlabel("Sample size")
axs[1].set_ylabel("")

fig.tight_layout()
fig.savefig(str(snakemake.output[4]), bbox_inches="tight")
plt.close(fig)

# Relative fitness advantages panel (vs reference variant).
fit_cols = [
    "mval",
    "bw",
    "mval2",
    "bw2",
    "adv_variant",
    "true_relative_growth",
    "estimated_relative_growth_ns1",
    "ci_low_ns1",
    "ci_high_ns1",
    "estimated_relative_growth_nsn",
    "ci_low_nsn",
    "ci_high_nsn",
]
fit_df = (
    df[fit_cols]
    .dropna(subset=["adv_variant", "true_relative_growth"])
    .drop_duplicates()
    .sort_values(["mval", "bw", "adv_variant"])
)

fit_long = pd.concat(
    [
        fit_df.rename(
            columns={
                "estimated_relative_growth_ns1": "estimate",
                "ci_low_ns1": "ci_low",
                "ci_high_ns1": "ci_high",
            }
        ).assign(method="Covvfit (ns=1)"),
        fit_df.rename(
            columns={
                "estimated_relative_growth_nsn": "estimate",
                "ci_low_nsn": "ci_low",
                "ci_high_nsn": "ci_high",
            }
        ).assign(method="Covvfit (ns=sample size)"),
    ],
    ignore_index=True,
)

mval_levels = sorted(fit_long["mval2"].unique())
bw_levels = sorted(fit_long["bw2"].unique())
variant_levels = sorted(fit_long["adv_variant"].unique())
palette = dict(
    zip(variant_levels, sns.color_palette("tab10", n_colors=len(variant_levels)))
)
marker_map = {"Covvfit (ns=1)": "o", "Covvfit (ns=sample size)": "s"}

x_min = float(fit_long["true_relative_growth"].min())
x_max = float(fit_long["true_relative_growth"].max())
pad = 0.15 * max(1e-6, (x_max - x_min))
x0, x1 = x_min - pad, x_max + pad

fig, axs = plt.subplots(
    len(mval_levels),
    len(bw_levels),
    figsize=(4.2 * len(bw_levels), 3.8 * len(mval_levels)),
    sharex=True,
    sharey=True,
    squeeze=False,
)

for i, mval in enumerate(mval_levels):
    for j, bw in enumerate(bw_levels):
        ax = axs[i, j]
        sub = fit_long[(fit_long["mval2"] == mval) & (fit_long["bw2"] == bw)]
        ax.plot([x0, x1], [x0, x1], linestyle="--", color="gray", linewidth=1.0)

        for _, row in sub.iterrows():
            if not np.all(
                np.isfinite([row["estimate"], row["ci_low"], row["ci_high"]])
            ):
                continue
            yerr = np.array(
                [[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]]
            )
            ax.errorbar(
                row["true_relative_growth"],
                row["estimate"],
                yerr=yerr,
                fmt=marker_map[row["method"]],
                color=palette[row["adv_variant"]],
                markeredgecolor="black",
                markersize=6,
                capsize=3,
                linewidth=1.1,
                alpha=0.95,
            )

        ax.set_title(f"missing={mval:.2f}, n={int(bw)}")
        if i == len(mval_levels) - 1:
            ax.set_xlabel("True relative growth advantage")
        if j == 0:
            ax.set_ylabel("Estimated relative growth advantage")
        ax.spines[["top", "right"]].set_visible(False)

legend_handles = []
for method, marker in marker_map.items():
    legend_handles.append(
        plt.Line2D([0], [0], marker=marker, color="black", linestyle="", label=method)
    )
for variant, color in palette.items():
    legend_handles.append(
        plt.Line2D(
            [0], [0], marker="o", color=color, linestyle="", label=f"Variant {variant}"
        )
    )

fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=min(4, len(legend_handles)),
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
)
fig.tight_layout()
fig.savefig(str(snakemake.output[5]), bbox_inches="tight")
plt.close(fig)
