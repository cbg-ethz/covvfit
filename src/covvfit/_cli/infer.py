"""Script running Covvfit inference on the data."""
import warnings
from pathlib import Path
from typing import Annotated, NamedTuple, Optional

import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pydantic
import typer
import yaml

import covvfit._preprocess_abundances as preprocess
import covvfit._quasimultinomial as qm
import covvfit.plotting as plot

plot_ts = plot.timeseries


class _InputDates(NamedTuple):
    min_date: Optional[str]
    max_date: Optional[str]
    max_days: int

    horizon: int
    horizon_max_date: Optional[str]


class _ParsedDates(NamedTuple):
    start_date: pd.Timestamp
    max_date: pd.Timestamp
    horizon_date: pd.Timestamp

    @property
    def horizon(self) -> int:
        return (self.horizon_date - self.max_date).days

    @property
    def total_length(self) -> int:
        return (self.horizon_date - self.start_date).days


def _parse_dates(raw: _InputDates, series: pd.Series) -> _ParsedDates:
    """Parses the user-provided dates."""
    series = pd.to_datetime(series)

    # First, infer the `max_date` for the data
    if raw.max_date is None:
        max_date = series.max()
    else:
        max_date = pd.to_datetime(raw.max_date)

    # Next, infer the `start_date`
    if raw.min_date is not None:
        start_date = pd.to_datetime(raw.min_date)
    else:
        start_date = max_date - pd.to_timedelta(raw.max_days, unit="D")

    # Finally, infer the horizon date
    if raw.horizon_max_date is not None:
        horizon_date = pd.to_datetime(raw.horizon_max_date)
    else:
        horizon_date = max_date + pd.to_timedelta(raw.horizon, unit="D")

    parsed = _ParsedDates(
        start_date=start_date, max_date=max_date, horizon_date=horizon_date
    )

    if parsed.horizon <= 0:
        raise ValueError("Inferred horizon is less than 1.")

    return parsed


class _Columns(NamedTuple):
    variant: str
    proportion: str
    date: str
    location: str


class _ProcessedData(NamedTuple):
    # dataframe: pd.DataFrame
    timepoints: list[np.ndarray]
    proportions: list[np.ndarray]

    cities: list[str]
    variants_effective: list[str]

    dates: _ParsedDates


def _process_data(
    *,
    data_path: str,
    data_separator: str,
    other_threshold: Optional[float],
    variants_investigated: list[str],
    locations_investigated: Optional[list[str]],
    dates: _InputDates,
    columns: _Columns,
) -> _ProcessedData:
    # Read the data
    data = pd.read_csv(data_path, sep=data_separator)
    for col in columns:
        if col not in data.columns:
            raise ValueError(
                f"Column {col} not found. Available columns: {data.columns}."
            )

    # If `locations_investigated` is specified, select the data
    if locations_investigated is not None:
        data = data[data[columns.location].isin(locations_investigated)]

    # Define the list with cities
    cities = list(data[columns.location].unique())
    if len(cities) == 0:
        raise ValueError(
            f"Length of cities is 0. Prespecified locations: {locations_investigated}."
        )

    # Now parse the dates and select the data in the right range
    data[columns.date] = pd.to_datetime(data[columns.date])
    parsed_dates = _parse_dates(raw=dates, series=data[columns.date])
    _mask = (data[columns.date] >= parsed_dates.start_date) & (
        data[columns.date] <= parsed_dates.max_date
    )
    data = data[_mask]

    if len(data) == 0:
        raise ValueError("There are no data in the specified range.")

    # Construct the pivot table
    data_wide = data.pivot_table(
        index=[columns.date, columns.location],
        columns=columns.variant,
        values=columns.proportion,
        fill_value=0.0,
    )
    # variants_full = data_wide.columns.tolist()
    data_wide = data_wide.reset_index()

    DAYS_COL = "days_from"
    data_wide[DAYS_COL] = (data_wide[columns.date] - parsed_dates.start_date).dt.days

    # Add the "other" variant
    OTHER_COL = "other"
    variants_effective = [OTHER_COL] + variants_investigated
    data_wide[OTHER_COL] = 1.0 - data_wide[variants_investigated].sum(axis=1)

    # Ensure that the value is not negative (we allow small discrepancy because of float arithmetic)
    _NEGATIVE_THRESHOLD: float = -1e-7
    if data_wide[OTHER_COL].min() < _NEGATIVE_THRESHOLD:
        raise ValueError(f"Negative value encountered, {data_wide[OTHER_COL].min()}.")

    # Finally, remove points whether OTHER_COL has too large value
    if other_threshold is not None:
        data_wide = data_wide[data_wide[OTHER_COL] <= other_threshold]

    timepoints, proportions = preprocess.make_data_list(
        data_wide,
        cities=cities,
        variants=variants_effective,
        city_col=columns.location,
        time_col=DAYS_COL,
        allow_for_undefined_behaviour=False,
    )

    return _ProcessedData(
        timepoints=timepoints,
        proportions=proportions,
        cities=cities,
        variants_effective=variants_effective,
        dates=parsed_dates,
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
    time_spacing: Annotated[int, pydantic.Field(strict=True, ge=1)] = pydantic.Field(
        default=2, help="Spacing between ticks on the time axis (in months)."
    )
    backend: Optional[str] = pydantic.Field(
        default=None, help="Matplotlib backend to use."
    )
    extensions: list[str] = pydantic.Field(
        default_factory=lambda: ["png", "pdf"],
        help="Extensions to which the figure should be exported.",
    )
    dpi: Annotated[int, pydantic.Field(strict=True, ge=1)] = pydantic.Field(
        default=500, help="DPI changes the figure resolution."
    )


class AnalysisSettings(pydantic.BaseModel):
    residuals_p1mp: bool = pydantic.Field(
        default=False, help="Whether to use p(1-p) in the variance formula."
    )
    data_separator: str = pydantic.Field(default="\t", help="Data separator.")
    n_starts: Annotated[int, pydantic.Field(strict=True, ge=1)] = pydantic.Field(
        default=10, help="Number of random starts in the optimization procedure."
    )
    other_threshold: Optional[float] = pydantic.Field(
        default=None,
        help="If the proportion of other variants (not investigated) exceeds this value, the data point will be removed.",
    )


class Config(pydantic.BaseModel):
    variants: list[str] = pydantic.Field(
        default_factory=lambda: [],
        help="List of variants to be included in the analysis.",
    )
    locations: Optional[list[str]] = pydantic.Field(
        default=None,
        help="List of locations to be included in the analysis. If `None`, all locations are used.",
    )
    plot: PlotSettings = pydantic.Field(
        default_factory=PlotSettings, help="Plot settings."
    )
    analysis: AnalysisSettings = pydantic.Field(
        default_factory=AnalysisSettings, help="Analysis settings."
    )


def _make_prediction_csv_entry(
    *,
    time,
    city,
    variant,
    value,
    label, 
) -> dict:
    return {
        "date": time,
        "location": city,
        "variant": variant,
        "value": value,
        "label": label,
    }


def _assemble_prediction_csv(
    *,
    time_observed: list,
    y_observed: list,
    time_retrodict: list,
    time_predict: list,
    y_predict: list,
    y_predict_ci: list,
    y_retrodict: list,
    y_retrodict_ci: list,
    start_date: list,
    variant_names: list[str],
    city_names: list[str],
) -> pd.DataFrame:
    """Assembles data used internally into a CSV
    with predictions for each city."""
    entries = []

    for city_index, city_name in enumerate(city_names):
        # First, add the observed data
        assert len(time_observed[city_index]) == y_observed[city_index].shape[0]

        for date_index, date in enumerate(time_observed[city_index]):
            for variant_index, variant_name in enumerate(variant_names):
                entries.append(
                    _make_prediction_csv_entry(
                        time=start_date + pd.Timedelta(float(date), unit="D"),
                        city=city_name,
                        variant=variant_name,
                        value=y_observed[city_index][date_index, variant_index],
                        label="observed_value",
                    )
                )
        # Then, add the retrodicted values
        assert len(time_retrodict[city_index]) == y_retrodict[city_index].shape[0]

        for date_index, date in enumerate(time_retrodict[city_index]):
            for variant_index, variant_name in enumerate(variant_names):
                # Add the fitted value
                entries.append(
                    _make_prediction_csv_entry(
                        time=start_date + pd.Timedelta(float(date), unit="D"),
                        city=city_name,
                        variant=variant_name,
                        value=y_retrodict[city_index][date_index, variant_index],
                        label="fitted_value",
                    )
                )
                # Add the lower confidence interval
                entries.append(
                    _make_prediction_csv_entry(
                        time=start_date + pd.Timedelta(float(date), unit="D"),
                        city=city_name,
                        variant=variant_name,
                        value=y_retrodict_ci[city_index].lower[
                            date_index, variant_index
                        ],
                        label="fitted_lower_ci",
                    )
                )
                # Add the upper confidence interval
                entries.append(
                    _make_prediction_csv_entry(
                        time=start_date + pd.Timedelta(float(date), unit="D"),
                        city=city_name,
                        variant=variant_name,
                        value=y_retrodict_ci[city_index].upper[
                            date_index, variant_index
                        ],
                        label="fitted_upper_ci",
                    )
                )

        # Finally, add the predicted values
        assert len(time_predict[city_index]) == y_predict[city_index].shape[0]

        for date_index, date in enumerate(time_predict[city_index]):
            for variant_index, variant_name in enumerate(variant_names):
                # Add the value
                entries.append(
                    _make_prediction_csv_entry(
                        time=start_date + pd.Timedelta(float(date), unit="D"),
                        city=city_name,
                        variant=variant_name,
                        value=y_predict[city_index][date_index, variant_index],
                        label="predicted_value",
                    )
                )
                # Add the lower confidence interval
                entries.append(
                    _make_prediction_csv_entry(
                        time=start_date + pd.Timedelta(float(date), unit="D"),
                        city=city_name,
                        variant=variant_name,
                        value=y_predict_ci[city_index].lower[
                            date_index, variant_index
                        ],
                        label="predicted_lower_ci",
                    )
                )
                # Add the upper confidence interval
                entries.append(
                    _make_prediction_csv_entry(
                        time=start_date + pd.Timedelta(float(date), unit="D"),
                        city=city_name,
                        variant=variant_name,
                        value=y_predict_ci[city_index].upper[
                            date_index, variant_index
                        ],
                        label="predicted_upper_ci",
                    )
                )

    return pd.DataFrame(entries)


def _parse_config(
    config_path: Optional[str],
    variants: Optional[list[str]],
    locations: Optional[list[str]],
    time_spacing: Optional[int],
    data_separator: Optional[str],
    residuals_p1mp: bool,
) -> Config:
    if variants is None and config_path is None:
        raise ValueError(
            "The variant names are not specified. Use `--config` argument or `-v` to specify them."
        )

    if config_path is None:
        config = Config()
    else:
        with open(config_path) as fh:
            payload = yaml.safe_load(fh)
        config = Config(**payload)

    # Overwrite variants, if specified
    if variants is not None:
        config.variants = variants
    if len(config.variants) == 0:
        raise ValueError("No variants have been specified.")

    if locations is not None:
        config.locations = locations
    if config.locations is not None and len(config.locations) == 0:
        raise ValueError("No locations have been specified.")

    # Overwrite time spacing, if specified
    if time_spacing is not None:
        config.plot.time_spacing = time_spacing

    # Overwrite the data separator
    if data_separator is not None:
        config.analysis.data_separator = data_separator

    # Overwrite residuals calculation
    if residuals_p1mp is True:
        config.analysis.residuals_p1mp = True

    return config


class _OutputDir(NamedTuple):
    output_path: str
    overwrite: bool

    def create(self) -> Path:
        output = Path(self.output_path)
        output.mkdir(parents=True, exist_ok=self.overwrite)
        return output


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
            help="Variant names to be included in the analysis. Note: overrides the settings in the config file (--config).",
        ),
    ] = None,
    locations: Annotated[
        Optional[list[str]],
        typer.Option(
            "--loc",
            "-l",
            help="Location names to be included in the analysis. Note: overrides the settings in the config file (--config).",
        ),
    ] = None,
    data_separator: Annotated[
        str,
        typer.Option(
            "--separator",
            "-s",
            help="Data separator used to read the input file. "
            "By default read from the config file (if not specified, the TAB character).",
        ),
    ] = None,
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

    # Assemble input information into structured objects

    # --- Parse config and update it with appropriate command-line arguments
    config: Config = _parse_config(
        config_path=config,
        variants=var,
        locations=locations,
        time_spacing=time_spacing,
        data_separator=data_separator,
        residuals_p1mp=residuals_p1mp,
    )
    # --- Parse column specification in the CSV file
    columns = _Columns(
        variant=variant_col,
        proportion=proportion_col,
        date=date_col,
        location=location_col,
    )
    # --- Parse the provided input dates
    input_dates = _InputDates(
        min_date=date_min,
        max_date=date_max,
        max_days=max_days,
        horizon=horizon,
        horizon_max_date=horizon_date,
    )
    # --- Parse the output directory specification
    output_dir = _OutputDir(output_path=output, overwrite=overwrite_output)

    # Call the function processing the structured information
    # to obtain data analysis results
    _main(
        data_path=data,
        config=config,
        columns=columns,
        dates=input_dates,
        output=output_dir,
    )


def _main(
    *,
    data_path: str,
    config: Config,
    dates: _InputDates,
    columns: _Columns,
    output: _OutputDir,
) -> None:
    # Read the variants
    variants_investigated = config.variants
    # Set matplotlib backend
    _set_matplotlib_backend(config.plot.backend)  # Set matplotlib backend using config.

    # Process data
    bundle = _process_data(
        data_path=data_path,
        data_separator=config.analysis.data_separator,
        other_threshold=config.analysis.other_threshold,
        variants_investigated=variants_investigated,
        locations_investigated=config.locations,
        dates=dates,
        columns=columns,
    )
    del columns  # The columns should not be needed anymore in this function

    # Prepare the output path
    output: Path = output.create()

    # Save the config file
    with open(output / "config.yaml", "w") as fh:
        yaml.safe_dump(config.model_dump(), fh)

    def pprint(message):
        # TODO(Pawel): Consider setting up a proper logger.
        with open(output / "log.txt", "a") as file:
            file.write(message + "\n")
        print(message)

    cities = bundle.cities
    variants_effective = bundle.variants_effective
    start_date = bundle.dates.start_date
    horizon: int = bundle.dates.horizon  # The prediction horizon

    ts_lst, ys_effective = bundle.timepoints, bundle.proportions

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
    solution = qm.jax_multistart_minimize(
        loss, theta0, n_starts=config.analysis.n_starts
    )

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
        p1mp=config.analysis.residuals_p1mp,
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

    df_predictions = _assemble_prediction_csv(
        # observed data
        time_observed=ts_lst,
        y_observed=ys_effective,
        # retrodictions (fit to the observed data)
        time_retrodict=ts_lst,
        y_retrodict=ys_fitted,
        y_retrodict_ci=ys_fitted_confint,
        # predictions
        time_predict=ts_pred_lst,
        y_predict=ys_pred,
        y_predict_ci=ys_pred_confint,
        # auxiliary information needed
        start_date=start_date,
        city_names=cities,
        variant_names=variants_effective,
    )
    df_predictions.sort_values(
        by=["location", "date", "variant", "label"], inplace=True
    )
    df_predictions.to_csv(
        output / "predictions.csv",
        sep=config.analysis.data_separator,
        index=False,
    )
    del df_predictions

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
    df_final.to_csv(
        output / "pairwise_fitnesses.csv",
        sep=config.analysis.data_separator,
        index=False,
    )

    pprint("\n\nRelative fitness values:")
    for _, row in df_final.iterrows():
        if row["Variant"] == row["Reference_Variant"]:
            continue
        pprint(
            f"  {row['Variant']} / {row['Reference_Variant']}:\t{row['Estimate']:.3f} ({row['Lower_CI']:.3f} – {row['Upper_CI']:.3f})"
        )

    # Create a plot
    colors = [
        config.plot.variant_colors.get(var, "black") for var in variants_investigated
    ]

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
            bundle.dates.total_length,
            color=prediction_region_color,
            alpha=prediction_region_alpha,
            edgecolor=None,
            linewidth=None,
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

        ax.set_xlim(-0.5, bundle.dates.total_length + 0.5)
        ax.set_ylim(-0.01, 1.01)
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

    for ext in config.plot.extensions:
        figure_spec.fig.savefig(output / f"figure.{ext}", dpi=config.plot.dpi)
