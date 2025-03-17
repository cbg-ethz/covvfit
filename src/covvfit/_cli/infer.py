"""Script running Covvfit inference on the data."""
import warnings
from pathlib import Path
from typing import Annotated, NamedTuple, Optional

import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import pandas as pd
import pydantic
import typer
import yaml

import covvfit._preprocess_abundances as preprocess
import covvfit._quasimultinomial as qm
import covvfit.plotting as plot

plot_ts = plot.timeseries

_TIME_COL = "time"
_CITY_COL = "city"


class _ProcessedData(NamedTuple):
    dataframe: pd.DataFrame
    cities: list[str]
    variants_effective: list[str]
    start_date: pd.Timestamp


def _process_data(
    *,
    data_path: str,
    data_separator: str,
    variants_investigated: list[str],
    max_days: int,
    variant_col: str,
    proportion_col: str,
    date_col: str,
    location_col: str,
    variant_dates: str = "",  # TODO(Pawel): Remove
) -> _ProcessedData:
    data = pd.read_csv(data_path, sep=data_separator)

    with open(variant_dates) as file:
        var_dates_data = yaml.safe_load(file)
        # Access the var_dates data
        var_dates = var_dates_data["var_dates"]

    data_wide = data.pivot_table(
        index=[date_col, location_col],
        columns=variant_col,
        values=proportion_col,
        fill_value=0,
    ).reset_index()
    data_wide = data_wide.rename(columns={date_col: _TIME_COL, location_col: _CITY_COL})

    # Define the list with cities:
    cities = list(data_wide[_CITY_COL].unique())

    ## Set limit times for modeling

    max_date = pd.to_datetime(data_wide[_TIME_COL]).max()
    delta_time = pd.Timedelta(days=max_days)
    start_date = max_date - delta_time

    var_dates_parsed = {
        pd.to_datetime(date): variants for date, variants in var_dates.items()
    }

    def match_date(start_date):
        """Function to find the latest matching date in var_dates."""
        start_date = pd.to_datetime(start_date)
        closest_date = max(date for date in var_dates_parsed if date <= start_date)
        return closest_date, var_dates_parsed[closest_date]

    variants_full = match_date(start_date + delta_time)[
        1
    ]  # All the variants in this range

    variants_other = [
        i for i in variants_full if i not in variants_investigated
    ]  # Variants not of interest

    variants_effective = ["other"] + variants_investigated
    data_full = preprocess.preprocess_df(
        data_wide, cities, variants_full, date_min=start_date, zero_date=start_date
    )

    data_full["other"] = data_full[variants_other].sum(axis=1)
    data_full[variants_effective] = data_full[variants_effective].div(
        data_full[variants_effective].sum(axis=1), axis=0
    )

    return _ProcessedData(
        dataframe=data_full,
        cities=cities,
        variants_effective=variants_effective,
        start_date=start_date,
    )


def _set_matplotlib_backend(matplotlib_backend: Optional[str]):
    if matplotlib_backend is not None:
        import matplotlib

        matplotlib.use(matplotlib_backend)


class PredictionRegion(pydantic.BaseModel):
    region_color: str = "grey"
    region_alpha: pydantic.confloat(ge=0.0, le=1.0) = 0.1
    linestyle: str = ":"


class PlotDimensions(pydantic.BaseModel):
    panel_width: float = 4.0
    panel_height: float = 1.5
    dpi: int = 350

    wspace: float = pydantic.Field(
        default=1.0, help="Horizontal (width) spacing between figure panels."
    )
    hspace: float = pydantic.Field(
        default=0.5, help="Vertical (height) spacing between figure panels."
    )

    left: float = pydantic.Field(default=1.0, help="Left margin in the figure.")
    right: float = pydantic.Field(default=1.5, help="Right margin in the figure.")
    top: float = pydantic.Field(default=0.7, help="Top margin in the figure.")
    bottom: float = pydantic.Field(default=0.5, help="Bottom margin in the figure.")


class PlotSettings(pydantic.BaseModel):
    dimensions: PlotDimensions = pydantic.Field(default_factory=PlotDimensions)
    prediction: PredictionRegion = pydantic.Field(default_factory=PredictionRegion)
    variant_colors: dict[str, str] = pydantic.Field(
        default_factory=lambda: plot_ts.COLORS_COVSPECTRUM,
        help="Dictionary mapping variants to colors in the plot.",
    )
    time_spacing: pydantic.conint(ge=1) = pydantic.Field(
        default=1, help="Spacing between ticks on the time axis (in months)."
    )
    backend: Optional[str] = pydantic.Field(
        default=None, help="Matplotlib backend to use."
    )


class Config(pydantic.BaseModel):
    variants: list[str] = pydantic.Field(
        default_factory=lambda: [],
        help="List of variants to be included in the analysis.",
    )
    plot: PlotSettings = pydantic.Field(
        default_factory=PlotSettings, help="Plot settings."
    )


def _parse_config(
    config_path: Optional[str],
    variants: Optional[list[str]],
    time_spacing: Optional[int],
) -> Config:
    if config_path is None:
        config = Config()
    else:
        with open(config_path) as fh:
            payload = yaml.safe_load(fh)
        config = Config(**payload)

    if variants is not None:
        config.variants = variants

    if time_spacing is not None:
        config.plot.time_spacing = time_spacing

    if len(config.variants) == 0:
        raise ValueError("No variants have been specified.")

    return config


def infer(
    data: Annotated[
        str, typer.Option("--input", "-i", help="CSV with deconvolved data")
    ],
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory")],
    config: Annotated[
        Optional[str],
        typer.Option(
            "--config", "-c", help="Path to the YAML file with configuration."
        ),
    ] = None,
    var: Annotated[
        Optional[list[str]],
        typer.Option(
            "--var",
            "-v",
            help="Variant names to be included in the analysis. Note: override the settings in the config file (--config).",
        ),
    ] = None,
    data_separator: Annotated[
        str,
        typer.Option(
            "--separator", "-s", help="Separator to be used to read the CSV file."
        ),
    ] = "\t",
    max_days: Annotated[
        int,
        typer.Option(
            "--max-days",
            help="Number of the past dates to which the analysis will be restricted",
        ),
    ] = 240,
    date_min: Annotated[
        str,
        typer.Option(
            "--date-min",
            help="Minimum date to start load data in format YYYY-MM-DD. By default calculated using `--max_days` and `--date-max`.",
        ),
    ] = None,
    date_max: Annotated[
        str,
        typer.Option(
            "--date-max",
            help="Maximum date to finish loading data, provided in format YYYY-MM-DD. By default calculated as the last date in the CSV file.",
        ),
    ] = None,
    horizon: Annotated[
        int,
        typer.Option(
            "--horizon",
            help="Number of future days for which abundance prediction should be generated",
        ),
    ] = 60,
    horizon_date: Annotated[
        str,
        typer.Option(
            "--horizon-date",
            help="Date until when the predictions should occur, provided in format YYYY-MM-DD. By default calculated using `--horizon` and `--date-max`.",
        ),
    ] = None,
    time_spacing: Annotated[
        Optional[int],
        typer.Option(
            "--time-spacing",
            help="Spacing between ticks on the time axis in months",
        ),
    ] = None,
    variant_col: Annotated[
        str,
        typer.Option(
            "--variant-col", help="Name of the column representing observed variant"
        ),
    ] = "variant",
    proportion_col: Annotated[
        str,
        typer.Option(
            "--proportion-col",
            help="Name of the column representing observed proportion",
        ),
    ] = "proportion",
    date_col: Annotated[
        str,
        typer.Option(
            "--date-col", help="Name of the column representing measurement date"
        ),
    ] = "date",
    location_col: Annotated[
        str,
        typer.Option("--location-col", help="Name of the column with spatial location"),
    ] = "location",
    overwrite_output: Annotated[
        bool,
        typer.Option(
            "--overwrite-output",
            help="Allows overwriting the output directory, if it already exists. Note: this may result in unintented loss of data.",
        ),
    ] = False,
    residuals_p1mp: Annotated[
        bool,
        typer.Option(
            "--residuals-p1mp",
            help="If True, to calculate the overdispersion we will use `p_i(1-p_i)` in the denominator."
            "If False (default), we use `p_i` in the denominator.",
        ),
    ] = False,
) -> None:
    """Runs growth advantage inference."""
    # Ignore warnings with JAX converting arrays from 64-bit to 32-bit
    warnings.filterwarnings(
        "ignore",
        message=r"Explicitly requested dtype float64 requested in zeros.*",
        category=UserWarning,
    )

    if var is None and config is None:
        raise ValueError(
            "The variant names are not specified. Use `--config` argument or `-v` to specify them."
        )

    config: Config = _parse_config(
        config_path=config, variants=var, time_spacing=time_spacing
    )

    variants_investigated = config.variants
    _set_matplotlib_backend(config.plot.backend)  # Set matplotlib backend using config.

    bundle = _process_data(
        data_path=data,
        data_separator=data_separator,
        variants_investigated=variants_investigated,
        max_days=max_days,
        variant_col=variant_col,
        proportion_col=proportion_col,
        date_col=date_col,
        location_col=location_col,
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=overwrite_output)

    def pprint(message):
        with open(output / "log.txt", "a") as file:
            file.write(message + "\n")
        print(message)

    cities = bundle.cities
    variants_effective = bundle.variants_effective
    start_date = bundle.start_date

    ts_lst, ys_effective = preprocess.make_data_list(
        bundle.dataframe, cities=cities, variants=variants_effective
    )

    # Scale the time for numerical stability
    time_scaler = preprocess.TimeScaler()
    ts_lst_scaled = time_scaler.fit_transform(ts_lst)

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
    solution = qm.jax_multistart_minimize(loss, theta0, n_starts=10)

    theta_star = solution.x  # The maximum quasilikelihood estimate

    relative_growths = qm.get_relative_growths(
        theta_star, n_variants=n_variants_effective
    )

    DAYS_IN_A_WEEK = 7.0
    relative_growths_per_day = relative_growths / time_scaler.time_unit
    relative_growths_per_week = DAYS_IN_A_WEEK * relative_growths_per_day

    pprint(f"Relative growth advantages (per day): {relative_growths_per_day}")
    pprint(f"Relative growth advantages (per week): {relative_growths_per_week}")

    with open(output / "results.yaml", "w") as fh:
        payload = {
            "relative_growth_advantages_day": relative_growths_per_day.tolist(),
            "relative_growth_advantages_week": relative_growths_per_week.tolist(),
        }
        yaml.safe_dump(payload, fh)

    ## compute fitted values
    ys_fitted = qm.fitted_values(
        ts_lst_scaled, theta=theta_star, cities=cities, n_variants=n_variants_effective
    )

    ## compute covariance matrix
    covariance = qm.get_covariance(loss, theta_star)

    overdispersion_tuple = qm.compute_overdispersion(
        observed=ys_effective,
        predicted=ys_fitted,
        p1mp=residuals_p1mp,
    )

    overdisp_fixed = overdispersion_tuple.overall

    pprint(f"Overdispersion factor: {float(overdisp_fixed):.3f}.")
    pprint("Note that values lower than 1 signify underdispersion.")

    ## scale covariance by overdisp
    covariance_scaled = overdisp_fixed * covariance

    ## compute standard errors and confidence intervals of the estimates
    standard_errors_estimates = qm.get_standard_errors(covariance_scaled)
    confints_estimates = qm.get_confidence_intervals(
        theta_star, standard_errors_estimates, confidence_level=0.95
    )

    pprint("\n\nRelative growth advantages (per day):")
    for variant, m, low, up in zip(
        variants_effective[1:],
        qm.get_relative_growths(theta_star, n_variants=n_variants_effective),
        qm.get_relative_growths(confints_estimates[0], n_variants=n_variants_effective),
        qm.get_relative_growths(confints_estimates[1], n_variants=n_variants_effective),
    ):
        pprint(
            f"  {variant}: {float(m)/ time_scaler.time_unit :.4f} ({float(low) / time_scaler.time_unit:.4f} – {float(up) / time_scaler.time_unit :.4f})"
        )

    pprint("\n\nRelative growth advantages (per week):")
    for variant, m, low, up in zip(
        variants_effective[1:],
        qm.get_relative_growths(theta_star, n_variants=n_variants_effective),
        qm.get_relative_growths(confints_estimates[0], n_variants=n_variants_effective),
        qm.get_relative_growths(confints_estimates[1], n_variants=n_variants_effective),
    ):
        pprint(
            f"  {variant}: {DAYS_IN_A_WEEK * float(m)/ time_scaler.time_unit :.4f} ({DAYS_IN_A_WEEK * float(low) / time_scaler.time_unit:.4f} – {DAYS_IN_A_WEEK * float(up) / time_scaler.time_unit :.4f})"
        )

    # Generate predictions
    ys_fitted_confint = qm.get_confidence_bands_logit(
        theta_star,
        n_variants=n_variants_effective,
        ts=ts_lst_scaled,
        covariance=covariance_scaled,
    )

    ## compute predicted values and confidence bands
    ts_pred_lst = [jnp.arange(horizon + 1) + tt.max() for tt in ts_lst]
    ts_pred_lst_scaled = time_scaler.transform(ts_pred_lst)

    ys_pred = qm.fitted_values(
        ts_pred_lst_scaled,
        theta=theta_star,
        cities=cities,
        n_variants=n_variants_effective,
    )
    ys_pred_confint = qm.get_confidence_bands_logit(
        theta_star,
        n_variants=n_variants_effective,
        ts=ts_pred_lst_scaled,
        covariance=covariance_scaled,
    )

    # Output pairwise fitness advantages

    def make_relative_growths(theta_star):
        relative_growths = (
            qm.get_relative_growths(theta_star, n_variants=n_variants_effective)
            - time_scaler.t_min
        ) / (time_scaler.t_max - time_scaler.t_min)
        relative_growths = jnp.concat([jnp.array([0]), relative_growths])
        relative_growths = relative_growths * DAYS_IN_A_WEEK

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

    df_diffs = (
        pd.DataFrame(
            pairwise_diffs, index=variants_effective, columns=variants_effective
        )
        .reset_index()
        .melt(id_vars="index")
    )
    df_diffs.columns = ["Variant", "Reference_Variant", "Estimate"]

    # Create confidence interval DataFrames
    df_lower = (
        pd.DataFrame(
            relgrowths_confint[0], index=variants_effective, columns=variants_effective
        )
        .reset_index()
        .melt(id_vars="index")
    )
    df_upper = (
        pd.DataFrame(
            relgrowths_confint[1], index=variants_effective, columns=variants_effective
        )
        .reset_index()
        .melt(id_vars="index")
    )

    df_lower.columns = ["Variant", "Reference_Variant", "Lower_CI"]
    df_upper.columns = ["Variant", "Reference_Variant", "Upper_CI"]

    # Merge all data
    df_final = df_diffs.merge(df_lower, on=["Variant", "Reference_Variant"]).merge(
        df_upper, on=["Variant", "Reference_Variant"]
    )
    df_final.to_csv(output / "pairwise_fitnesses.csv", sep=data_separator, index=False)

    pprint("\n\nRelative fitness values:")
    for _, row in df_final.iterrows():
        if row["Variant"] == row["Reference_Variant"]:
            continue
        pprint(
            f"  {row['Variant']} / {row['Reference_Variant']}:\t{row['Estimate']:.3f} ({row['Lower_CI']:.3f} – {row['Upper_CI']:.3f})"
        )

    # Create a plot
    colors = [config.plot.variant_colors[var] for var in variants_investigated]

    plot_dimensions = config.plot.dimensions

    figure_spec = plot.arrange_into_grid(
        len(cities),
        axsize=(plot_dimensions.panel_width, plot_dimensions.panel_height),
        dpi=plot_dimensions.dpi,
        wspace=plot_dimensions.wspace,
        top=plot_dimensions.top,
        bottom=plot_dimensions.bottom,
        left=plot_dimensions.left,
        right=plot_dimensions.right,
        sharex=True,
    )

    def plot_city(ax, i: int) -> None:
        def remove_0th(arr):
            """We don't plot the artificial 0th variant 'other'."""
            return arr[:, 1:]

        # Mark region as predicted
        prediction_region_color = config.plot.prediction.region_color
        prediction_region_alpha = config.plot.prediction.region_alpha
        prediction_linestyle = config.plot.prediction.linestyle
        ax.axvspan(
            jnp.min(ts_pred_lst[i]),
            jnp.max(ts_pred_lst[i]),
            color=prediction_region_color,
            alpha=prediction_region_alpha,
        )

        # Plot fits in observed and unobserved time intervals.
        plot_ts.plot_fit(ax, ts_lst[i], remove_0th(ys_fitted[i]), colors=colors)
        plot_ts.plot_fit(
            ax,
            ts_pred_lst[i],
            remove_0th(ys_pred[i]),
            colors=colors,
            linestyle=prediction_linestyle,
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
        plot_ts.plot_complement(
            ax,
            ts_pred_lst[i],
            remove_0th(ys_pred[i]),
            linestyle=prediction_linestyle,
            alpha=0.3,
        )

        adjust_axis_fn = plot_ts.AdjustXAxisForTime(
            start_date, spacing_months=config.plot.time_spacing
        )
        adjust_axis_fn(ax)

        tick_positions = [0, 0.5, 1]
        tick_labels = ["0%", "50%", "100%"]
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.set_ylabel("Relative abundances")
        ax.set_title(cities[i])

    figure_spec.map(plot_city, range(len(cities)))

    handles = [
        mpatches.Patch(color=col, label=name)
        for name, col in zip(variants_investigated, colors)
    ]
    figure_spec.fig.legend(handles=handles, loc="outside center right", frameon=False)

    figure_spec.fig.savefig(output / "figure.pdf")
    figure_spec.fig.savefig(output / "figure.png")
