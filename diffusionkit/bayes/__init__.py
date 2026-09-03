"""Bayesian pipeline: fit the exact displacement likelihood, no MSD curve.

`fit_track` (one track) and `fit_population` (a whole dataset) are the entry
points; the engines and model pieces they compose are exposed below for
direct use. Anisotropy detection lives in `diffusionkit.bayes.anisotropy`,
and plotting in `diffusionkit.bayes.viz` -- neither is re-exported here, so
importing this package does not pull in matplotlib.

Importing this module enables jax's float64 mode process-wide: the
displacement covariance mixes a motion term and a localization-noise term
that can differ by orders of magnitude, which float32 cannot represent well
enough to factorize reliably.
"""
import jax

jax.config.update("jax_enable_x64", True)

from .api import TrackFit, fit_population, fit_track
from .inference import (
    MAPFit,
    fit_batch_svi,
    fit_map,
    fit_table_map,
    fit_table_nuts,
    fit_table_svi,
    sample_posterior,
)
from .likelihood import (
    anisotropic_displacement_covariance,
    anisotropic_step_covariance,
    displacement_covariance,
    fgn_covariance,
    fgn_gamma,
    noise_covariance,
)
from .model import (
    anomalous_diffusion_model,
    batched_anomalous_diffusion_model,
    batched_normal_diffusion_model,
    normal_diffusion_model,
)
from .priors import (
    WEAK_ANOMALOUS_PRIOR,
    WEAK_NORMAL_PRIOR,
    AnomalousModelPrior,
    NormalModelPrior,
    sigma_prior_from_localization,
)
from .simulate import simulate_fbm_tracks
