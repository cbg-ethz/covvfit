"""Script deconvolving the data."""
import warnings
from pathlib import Path
from typing import Annotated, Optional

import pydantic
import typer
import yaml

import covvfit._cli.infer as cli_infer


class Config(pydantic.BaseModel):
    variants: list[str] = pydantic.Field(
        default_factory=lambda: [],
        help="List of variants to be included in the analysis.",
    )
    loci: list[str] = pydantic.Field(
        default_factory=[],
        help="List of loci to be included in the analysis.",
    )
    locations: Optional[list[str]] = pydantic.Field(
        default=None,
        help="List of locations to be included in the analysis. If `None`, all locations are used.",
    )


def deconv(
    data: Annotated[
        str, typer.Option("--mutations", "-m", help="CSV with mutation data.")
    ],
    definitions: Annotated[
        str, typer.Option("--definitions", "-d", help="CSV with variant definitions")
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
    mutation_col: Annotated[
        str,
        typer.Option(
            "--locus-col", help="Name of the column representing observed locus."
        ),
    ] = "locus",
    proportion_col: Annotated[
        str,
        typer.Option(
            "--proportion-col",
            help="Name of the column representing observed mutation proportion at specific locus.",
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
) -> None:
    """

    Args:
        data: data frame with columns
    """
    # TODO(Pawel): Remove this warning once the tool has become stable.
    warnings.warn(
        "This tool is still experimental. Breaking changes may be introduced at any time."
    )

    # --- Parse the provided input dates
    input_dates = cli_infer._InputDates(
        min_date=date_min,
        max_date=date_max,
        max_days=max_days,
        horizon=horizon,
        horizon_max_date=horizon_date,
    )
    # --- Parse the output directory specification
    output_dir = cli_infer._OutputDir(output_path=output, overwrite=overwrite_output)

    _main(
        config=None,  # TODO(Pawel): Add
        output=output_dir,
        input_dates=input_dates,
    )


def _main(
    *,
    config,
    output: cli_infer._OutputDir,
    input_dates: cli_infer._InputDates,
):
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
