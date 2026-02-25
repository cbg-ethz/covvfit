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


def _plot_pred_obs(
    data: pd.DataFrame, palette: dict[str, tuple[float, float, float]]
) -> None:
    sns.lineplot(
        data=data,
        x="index",
        y="prediction",
        hue="variant",
        palette=palette,
        linewidth=1.5,
        legend=False,
    )
    obs = data.dropna(subset=["observed"])
    if len(obs) > 0:
        sns.scatterplot(
            data=obs,
            x="index",
            y="observed",
            hue="variant",
            palette=palette,
            s=12,
            alpha=0.65,
            legend=False,
        )


cfg = snakemake.config
plot_cfg = cfg["plotting"]
jpeg_dpi = int(plot_cfg["jpeg_dpi"])

df = pd.read_csv(str(snakemake.input[0]))
df["index"] = pd.to_datetime(df["index"])
df["mval2"] = df["mval"].round(2)
df["bw2"] = df["bw"].round(0).astype(int)
scenario_name = (
    str(df["scenario"].dropna().iloc[0])
    if "scenario" in df.columns and len(df) > 0
    else "scenario"
)

agg = (
    df.groupby(["index", "variant", "mval", "mval2", "bw", "bw2"], as_index=False)[
        ["ground", "prediction", "observed", "mval_day"]
    ]
    .mean()
    .sort_values(["mval", "bw", "variant", "index"])
)

corr = (
    agg.groupby(["mval", "bw"])
    .apply(lambda g: pd.Series({"cor": _safe_corr(g["prediction"], g["ground"])}))
    .reset_index()
)
agg = agg.merge(corr, on=["mval", "bw"], how="left")

variant_levels = sorted(agg["variant"].dropna().unique())
palette = dict(
    zip(variant_levels, sns.color_palette("tab10", n_colors=len(variant_levels)))
)

sns.set_theme(style="whitegrid", context="talk")

# Full facet plot: observed points + Covvfit prediction line.
full = sns.FacetGrid(
    data=agg,
    row="mval2",
    col="bw2",
    sharex=True,
    sharey=True,
    height=2.3,
    aspect=1.4,
    margin_titles=True,
)
full.map_dataframe(lambda data, **_: _plot_pred_obs(data, palette=palette))
full.set_axis_labels("", "Relative Abundance")
for ax in full.axes.flat:
    ax.tick_params(axis="x", rotation=45)

variant_handles = [
    plt.Line2D([0], [0], color=palette[v], lw=2, label=str(v)) for v in variant_levels
]
style_handles = [
    plt.Line2D([0], [0], color="black", lw=1.8, label="Covvfit prediction"),
    plt.Line2D([0], [0], marker="o", linestyle="", color="black", label="Observed"),
]
full.figure.legend(
    handles=style_handles + variant_handles,
    loc="upper center",
    ncol=min(6, len(style_handles) + len(variant_handles)),
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
)
full.figure.subplots_adjust(bottom=0.08, top=0.90, hspace=0.28, wspace=0.12)
full.figure.suptitle(f"Scenario: {scenario_name}", y=0.98)
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
subset = agg[agg["mval2"].isin(subset_mvals) & agg["bw2"].isin(subset_bws)].copy()
subset_corr = subset.groupby(["mval2", "bw2"], as_index=False).first()[
    ["mval2", "bw2", "cor"]
]

g = sns.FacetGrid(
    data=subset,
    row="bw2",
    col="mval2",
    sharex=True,
    sharey=True,
    height=2.6,
    aspect=1.3,
    margin_titles=True,
)
g.map_dataframe(lambda data, **_: _plot_pred_obs(data, palette=palette))
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
        r2 = (
            float(row["cor"].iloc[0]) ** 2
            if np.isfinite(row["cor"].iloc[0])
            else np.nan
        )
        ax.text(
            0.03,
            0.95,
            f"R2: {r2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

g.figure.subplots_adjust(bottom=0.1, top=0.92, hspace=0.22, wspace=0.12)
g.figure.suptitle(f"Scenario: {scenario_name}", y=0.98)
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

# Heatmap of R^2 values for the single prediction setup.
heat_df = corr.copy()
heat_df["mval2"] = heat_df["mval"].round(2)
heat_df["bw2"] = heat_df["bw"].round(0).astype(int)
heat_df["r2"] = heat_df["cor"] ** 2
pivot = heat_df.pivot(index="mval2", columns="bw2", values="r2")

fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar=True, ax=ax)
ax.set_title("R2: Prediction vs ground truth")
ax.set_xlabel("Sample size")
ax.set_ylabel("Missing rate")
fig.tight_layout()
fig.suptitle(f"Scenario: {scenario_name}", y=1.02)
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
    "estimated_relative_growth",
    "ci_low",
    "ci_high",
]
fit_df = (
    df[fit_cols]
    .dropna(subset=["adv_variant", "true_relative_growth"])
    .drop_duplicates()
    .sort_values(["mval", "bw", "adv_variant"])
)

mval_levels = sorted(fit_df["mval2"].unique())
bw_levels = sorted(fit_df["bw2"].unique())
adv_levels = sorted(fit_df["adv_variant"].unique())
adv_palette = dict(
    zip(adv_levels, sns.color_palette("tab10", n_colors=len(adv_levels)))
)

x_min = float(fit_df["true_relative_growth"].min())
x_max = float(fit_df["true_relative_growth"].max())
if np.any(np.isfinite(fit_df["ci_low"])):
    x_min = min(x_min, float(np.nanmin(fit_df["ci_low"])))
if np.any(np.isfinite(fit_df["ci_high"])):
    x_max = max(x_max, float(np.nanmax(fit_df["ci_high"])))
pad = 0.15 * max(1e-6, (x_max - x_min))
x0, x1 = x_min - pad, x_max + pad
tick_start = int(np.floor(x0))
tick_end = int(np.ceil(x1))
axis_ticks = np.arange(tick_start, tick_end + 1, 1, dtype=int)
if axis_ticks.size < 2:
    axis_ticks = np.asarray([tick_start, tick_start + 1], dtype=int)

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
        sub = fit_df[(fit_df["mval2"] == mval) & (fit_df["bw2"] == bw)]
        ax.plot([x0, x1], [x0, x1], linestyle="--", color="gray", linewidth=1.0)
        for _, row in sub.iterrows():
            if not np.all(
                np.isfinite(
                    [row["estimated_relative_growth"], row["ci_low"], row["ci_high"]]
                )
            ):
                continue
            yerr = np.array(
                [
                    [row["estimated_relative_growth"] - row["ci_low"]],
                    [row["ci_high"] - row["estimated_relative_growth"]],
                ]
            )
            ax.errorbar(
                row["true_relative_growth"],
                row["estimated_relative_growth"],
                yerr=yerr,
                fmt="o",
                color=adv_palette[row["adv_variant"]],
                markeredgecolor="black",
                markersize=6,
                capsize=3,
                linewidth=1.1,
                alpha=0.95,
            )
        ax.set_title(f"missing={mval:.2f}, n={int(bw)}")
        if i == len(mval_levels) - 1:
            ax.set_xlabel("Ground truth")
        if j == 0:
            ax.set_ylabel("Estimated")
        ax.set_xlim(x0, x1)
        ax.set_ylim(x0, x1)
        ax.set_xticks(axis_ticks)
        ax.set_yticks(axis_ticks)
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)

fig.legend(
    handles=[
        plt.Line2D([0], [0], marker="o", color=c, linestyle="", label=str(v))
        for v, c in adv_palette.items()
    ],
    loc="upper center",
    title="Variant (relative to the first one)",
    ncol=min(4, len(adv_palette)),
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.suptitle(f"Scenario: {scenario_name}", y=0.99)
fig.savefig(str(snakemake.output[5]), bbox_inches="tight")
plt.close(fig)

# Single-config view by city: smooth predictions and observed points.
target_mval = float(
    plot_cfg.get("single_config_missing_rate", sorted(df["mval2"].unique())[0])
)
target_bw = int(
    plot_cfg.get("single_config_sample_size", sorted(df["bw2"].unique())[0])
)
available_mvals = sorted(df["mval2"].unique())
available_bws = sorted(df["bw2"].unique())
if target_mval not in available_mvals:
    target_mval = min(available_mvals, key=lambda x: abs(float(x) - target_mval))
if target_bw not in available_bws:
    target_bw = min(available_bws, key=lambda x: abs(int(x) - target_bw))

single = df[(df["mval2"] == target_mval) & (df["bw2"] == target_bw)].copy()
city_levels = sorted(single["city"].dropna().unique())
city_plot = sns.FacetGrid(
    data=single,
    col="city",
    col_wrap=2 if len(city_levels) > 1 else 1,
    sharex=True,
    sharey=True,
    height=3.0,
    aspect=1.25,
)
city_plot.map_dataframe(lambda data, **_: _plot_pred_obs(data, palette=palette))
city_plot.set_axis_labels("Date", "Relative\nabundance")
for ax in city_plot.axes.flat:
    ax.tick_params(axis="x", rotation=45)
city_plot.figure.suptitle(
    (
        f"Scenario: {scenario_name} | "
        f"Single config by city (missing={target_mval:.2f}, n={int(target_bw)})"
    ),
    y=1.02,
)
city_plot.figure.legend(
    handles=style_handles + variant_handles,
    loc="upper center",
    ncol=min(6, len(style_handles) + len(variant_handles)),
    frameon=False,
    bbox_to_anchor=(0.5, 1.10),
)
city_plot.figure.subplots_adjust(top=0.82, bottom=0.14, wspace=0.18, hspace=0.25)
city_plot.figure.savefig(str(snakemake.output[6]), bbox_inches="tight")
plt.close(city_plot.figure)
