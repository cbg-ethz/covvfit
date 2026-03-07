"""Private subpackage for multi-city variant beta-binomial model."""

from covvfit._variant_beta_binomial.data import DispersionParams, ModelData
from covvfit._variant_beta_binomial.dispersion import (
    build_eta_rho,
    rho_to_log_kappa,
    select_rho_per_sample,
    transform_eta_to_rho,
)
from covvfit._variant_beta_binomial.growth import (
    LinearGrowthParams,
    compute_log_pi,
    effective_theta,
    log_f,
)
from covvfit._variant_beta_binomial.likelihood import (
    beta_binomial_logpmf,
    make_beta_binomial_shapes,
    masked_beta_binomial_loglik,
)
from covvfit._variant_beta_binomial.mapping import (
    compute_log_mu,
    compute_mu,
    make_log_A_mask,
)
from covvfit._variant_beta_binomial.model import model_loglik
from covvfit._variant_beta_binomial.numpyro_model import numpyro_model_hierarchical
from covvfit._variant_beta_binomial.simulate import (
    SimulationResult,
    sample_beta_binomial_counts,
    simulate_dataset,
    simulate_log_pi,
    simulate_mu,
)

__all__ = [
    "ModelData",
    "DispersionParams",
    "LinearGrowthParams",
    "SimulationResult",
    "effective_theta",
    "log_f",
    "compute_log_pi",
    "make_log_A_mask",
    "compute_log_mu",
    "compute_mu",
    "build_eta_rho",
    "transform_eta_to_rho",
    "rho_to_log_kappa",
    "select_rho_per_sample",
    "make_beta_binomial_shapes",
    "beta_binomial_logpmf",
    "masked_beta_binomial_loglik",
    "sample_beta_binomial_counts",
    "simulate_log_pi",
    "simulate_mu",
    "simulate_dataset",
    "model_loglik",
    "numpyro_model_hierarchical",
]
