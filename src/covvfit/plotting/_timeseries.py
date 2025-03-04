"""utilities to plot"""

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from jaxtyping import Array, Float

Variant = str
Color = str

COLORS_COVSPECTRUM: dict[Variant, Color] = {
    "B.1.1.7": "#D16666",
    "B.1.351": "#FF6666",
    "P.1": "#FFB3B3",
    "B.1.617.1": "#66C265",
    "B.1.617.2": "#66A366",
    "BA.1": "#A366A3",
    "BA.2": "#CFAFCF",
    "BA.4": "#8a66ff",
    "BA.5": "#585eff",
    "BA.2.75": "#008fe0",
    "BQ.1.1": "#ac00e0",
    "XBB.1.9": "#bb6a33",
    "XBB.1.5": "#ff5656",
    "XBB.1.16": "#e99b30",
    "XBB.2.3": "#f5e424",
    "EG.5": "#b4e80b",
    "BA.2.86": "#FF20E0",
    # TODO(Pawel, David): Use consistent colors with Covspectrum
    "JN.1": "#00e9ff",  # improv
    "KP.2": "#D16666",  # improv
    "KP.3": "#66A366",  # improv
    "XEC": "#A366A3",  # improv
    "undetermined": "#969696",
}


def make_legend(colors: list[Color], variants: list[Variant]) -> list[mpatches.Patch]:
    """make a shared legend for the plot"""
    # Create a patch (i.e., a colored box) for each variant
    variant_patches = [
        mpatches.Patch(color=color, label=variants[i]) for i, color in enumerate(colors)
    ]

    # Create lines for "fitted", "predicted", and "observed" labels
    fitted_line = mlines.Line2D([], [], color="black", linestyle="-", label="fitted")
    predicted_line = mlines.Line2D(
        [], [], color="black", linestyle="--", label="predicted"
    )
    observed_points = mlines.Line2D(
        [], [], color="black", marker="o", linestyle="None", label="daily estimates"
    )
    blank_line = mlines.Line2D([], [], color="white", linestyle="", label="")

    # Combine all the legend handles
    handles = variant_patches + [
        blank_line,
        fitted_line,
        predicted_line,
        observed_points,
    ]

    return handles


class _MonthStartLocator(ticker.Locator):
    """
    A custom Locator that, given a reference 'start_date', places ticks on
    the first day of each month within the data's visible range.
    """

    def __init__(self, start_date: str, time_unit: str) -> None:
        """

        Args:
            start_date: date corresponding to `value=0` timepoint, e.g., "2025-02-01"
            time_unit: time unit, e.g., "D" for days
        """
        super().__init__()
        # Store the reference start_date as a Timestamp
        self.start_date = pd.to_datetime(start_date)
        self.time_unit = time_unit

        # See the todo item in the `__call__` method
        if time_unit != "D":
            raise NotImplementedError(
                "Currently only days (time_unit='D') are supported."
            )

    def __call__(self):
        """
        Return the list of tick positions (as numeric days since start_date)
        that fall on the 1st of each month in the current view interval.
        """
        # Determine the min and max of the current axis range (in numeric days)
        vmin, vmax = self.axis.get_view_interval()
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        # Convert these numeric offsets to actual datetime objects
        dt_min = self.start_date + pd.to_timedelta(vmin, self.time_unit)
        dt_max = self.start_date + pd.to_timedelta(vmax, self.time_unit)

        # Find the 1st-of-the-month that is >= dt_min
        first_of_minmonth = pd.to_datetime(f"{dt_min.year}-{dt_min.month}-01")
        if first_of_minmonth < dt_min:
            # if dt_min is past that 1st, jump to next month
            first_of_minmonth += pd.offsets.MonthBegin(1)

        # Collect offsets for each 1st-of-the-month until we exceed dt_max
        ticks = []
        current = first_of_minmonth
        while current <= dt_max:
            # TODO(Pawel): This depends on the time unit!
            offset_days = (current - self.start_date).days
            ticks.append(offset_days)
            current += pd.offsets.MonthBegin(1)

        return ticks

    def tick_values(self, vmin, vmax):
        # Matplotlib may call tick_values directly; just reuse __call__()
        return self.__call__()


class AdjustXAxisForTime:
    def __init__(
        self,
        start_date: str,
        fmt="%b. '%y",
        time_unit: str = "D",
    ) -> None:
        self.start_date = start_date
        self.fmt = fmt
        self.time_unit = time_unit

    def _num_to_date(self, num: pd.Series | Float[Array, " timepoints"]) -> pd.Series:
        """Convert days number into a date format"""
        date = pd.to_datetime(self.start_date) + pd.to_timedelta(num, self.time_unit)
        return date.strftime(self.fmt)

    def __call__(self, ax: plt.Axes) -> None:
        ax.xaxis.set_major_locator(
            _MonthStartLocator(start_date=self.start_date, time_unit=self.time_unit)
        )
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: self._num_to_date(x))
        )


def num_to_date(
    num: pd.Series | Float[Array, " timepoints"], date_min: str, fmt="%b. '%y"
) -> pd.Series:
    """convert days number into a date format"""
    date = pd.to_datetime(date_min) + pd.to_timedelta(num, "D")
    return date.strftime(fmt)


def plot_fit(
    ax: plt.Axes,
    ts: Float[Array, " timepoints"],
    y_fit: Float[Array, "timepoints variants"],
    *,
    colors: list[Color],
    variants: list[Variant] | None = None,
    linestyle="-",
    **kwargs,
) -> None:
    """
    Function to plot fitted values with customizable line type.

    Parameters:
        ax (matplotlib.axes): The axis to plot on.
        ts (array-like): Time series data.
        y_fit (array-like): Fitted values for each variant.
        variants (list): List of variant names.
        colors (list): List of colors for each variant.
        linestyle (str): Line style for plotting (e.g., '-', '--', '-.', ':').
    """
    sorted_indices = np.argsort(ts)
    n_variants = y_fit.shape[-1]
    if variants is None:
        variants = [""] * n_variants

    for i, variant in enumerate(variants):
        ax.plot(
            ts[sorted_indices],
            y_fit[sorted_indices, i],
            color=colors[i],
            linestyle=linestyle,
            label=variant,
            **kwargs,
        )


def plot_complement(
    ax: plt.Axes,
    ts: Float[Array, " timepoints"],
    y_fit: Float[Array, "timepoints variants"],
    color: str = "grey",
    linestyle: str = "-",
    **kwargs,
) -> None:
    ## function to plot 1-sum(fitted_values) i.e., the other variant(s)
    sorted_indices = np.argsort(ts)
    ax.plot(
        ts[sorted_indices],
        (1 - y_fit.sum(axis=-1))[sorted_indices],
        color=color,
        linestyle=linestyle,
        **kwargs,
    )


def plot_data(
    ax: plt.Axes,
    ts: Float[Array, " timepoints"],
    ys: Float[Array, "timepoints variants"],
    colors: list[Color],
    size: float = 4.0,
    alpha: float = 0.5,
    **kwargs,
) -> None:
    ## function to plot raw values
    for i in range(ys.shape[-1]):
        ax.scatter(
            ts,
            ys[:, i],
            alpha=alpha,
            color=colors[i],
            s=size,
            **kwargs,
        )


def plot_confidence_bands(
    ax: plt.Axes,
    ts: Float[Array, " timepoints"],
    conf_bands,
    *,
    colors: list[Color],
    label: str = "Confidence band",
    alpha: float = 0.2,
    **kwargs,
) -> None:
    """
    Plot confidence intervals for fitted values on a given axis with customizable confidence level.

    Parameters:
        ax: The axis to plot on.
        ts: Time series data.
        conf_bands: confidence bands object. It can be:
            1. A class with attributes `lower` and `upper`, each of which is
               an array of shape `(n_timepoints, n_variants)` and represents
               the lower and upper confidence bands, respectively.
            2. A tuple of two arrays of the specified shape.
            3. A dictionary with keys "lower" and "upper"
        color: Color for the confidence interval.
        label: Label for the confidence band. Default is "Confidence band".
        alpha: Alpha level controling the opacity.
        **kwargs: Additional keyword arguments for `ax.fill_between`.
    """
    # Sort indices for time series
    sorted_indices = np.argsort(ts)

    lower, upper = None, None
    if hasattr(conf_bands, "lower") and hasattr(conf_bands, "upper"):
        lower = conf_bands.lower
        upper = conf_bands.upper
    elif isinstance(conf_bands, dict):
        lower = conf_bands["lower"]
        upper = conf_bands["upper"]
    else:
        lower = conf_bands[0]
        upper = conf_bands[1]

    if lower is None or upper is None:
        raise ValueError("Confidence bands are not in a recognized format.")

    lower = np.asarray(lower)
    upper = np.asarray(upper)

    if lower.ndim != 2 or lower.shape != upper.shape:
        raise ValueError("The shape is wrong.")

    n_variants = lower.shape[-1]

    # Plot the confidence interval
    for i in range(n_variants):
        ax.fill_between(
            ts[sorted_indices],
            lower[sorted_indices, i],
            upper[sorted_indices, i],
            color=colors[i],
            alpha=alpha,
            label=label,
            **kwargs,
        )
