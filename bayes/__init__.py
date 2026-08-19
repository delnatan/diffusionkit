import jax

jax.config.update("jax_enable_x64", True)

from .likelihood import (
    fgn_gamma,
    fgn_covariance,
    noise_covariance,
    displacement_covariance,
    anisotropic_step_covariance,
    anisotropic_displacement_covariance,
)
from .model import (
    normal_diffusion_model,
    anomalous_diffusion_model,
    batched_normal_diffusion_model,
    batched_anomalous_diffusion_model,
)
from .priors import (
    NormalModelPrior,
    AnomalousModelPrior,
    WEAK_NORMAL_PRIOR,
    WEAK_ANOMALOUS_PRIOR,
    sigma_prior_from_localization,
)
from .inference import (
    MAPFit,
    fit_map,
    sample_posterior,
    sample_posterior_table,
    fit_batch_svi,
    fit_all_tracks,
    fit_batch_map,
)
from .simulate import simulate_fbm_tracks
from .api import TrackFit, fit_track, fit_population
from . import anisotropy
from .viz import (
    samples_dict_to_arrays,
    plot_posterior_corner,
    plot_mcmc_trace,
    plot_estimator_scatter,
    plot_D_alpha_joint,
    plot_classic_vs_bayes_joint,
    plot_D_recovery,
    plot_alpha_recovery,
    plot_bias_vs_D_null,
)

__all__ = [
    "fgn_gamma",
    "fgn_covariance",
    "noise_covariance",
    "displacement_covariance",
    "anisotropic_step_covariance",
    "anisotropic_displacement_covariance",
    "normal_diffusion_model",
    "anomalous_diffusion_model",
    "batched_normal_diffusion_model",
    "batched_anomalous_diffusion_model",
    "NormalModelPrior",
    "AnomalousModelPrior",
    "WEAK_NORMAL_PRIOR",
    "WEAK_ANOMALOUS_PRIOR",
    "sigma_prior_from_localization",
    "MAPFit",
    "fit_map",
    "sample_posterior",
    "sample_posterior_table",
    "fit_batch_svi",
    "fit_all_tracks",
    "fit_batch_map",
    "simulate_fbm_tracks",
    "TrackFit",
    "fit_track",
    "fit_population",
    "anisotropy",
    "samples_dict_to_arrays",
    "plot_posterior_corner",
    "plot_mcmc_trace",
    "plot_estimator_scatter",
    "plot_D_alpha_joint",
    "plot_classic_vs_bayes_joint",
    "plot_D_recovery",
    "plot_alpha_recovery",
    "plot_bias_vs_D_null",
]
