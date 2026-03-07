import covvfit._deconvolution as deconvolution
import covvfit._dynamics as dynamics
import covvfit._numeric as numeric
import covvfit._quasimultinomial as quasimultinomial
import covvfit._variant_beta_binomial as variant_beta_binomial

try:
    import covvfit.simulation as simulation
except Exception as e:
    import warnings

    warnings.warn(
        f"It is not possible to use `simulation` subpackage due to missing dependencies. Exception raised: {e}"
    )
    simulation = None

import covvfit._preprocess_abundances as preprocess
import covvfit.plotting as plot
from covvfit._splines import create_spline_matrix
from covvfit._variant_beta_binomial import (
    DispersionParams,
    LinearGrowthParams,
    ModelData,
    model_loglik,
    numpyro_model_hierarchical,
    simulate_dataset,
)

VERSION = "0.3.1"


__all__ = [
    "create_spline_matrix",
    "dynamics",
    "deconvolution",
    "load_data",
    "VERSION",
    "numeric",
    "quasimultinomial",
    "variant_beta_binomial",
    "ModelData",
    "DispersionParams",
    "LinearGrowthParams",
    "model_loglik",
    "numpyro_model_hierarchical",
    "simulate_dataset",
    "plot",
    "preprocess",
    "simulation",
]
