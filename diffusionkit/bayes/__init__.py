"""Bayesian pipeline: fit the exact displacement likelihood, no MSD curve.

`fit_track` is the entry point, a per-track diagnostic tool (full NUTS
posterior); the engines and model pieces it composes are exposed below for
direct use. Plotting lives in `diffusionkit.bayes.viz`, not re-exported
here, so importing this package does not pull in matplotlib.

Importing this module enables jax's float64 mode process-wide: the
displacement covariance mixes a motion term and a localization-noise term
that can differ by orders of magnitude, which float32 cannot represent well
enough to factorize reliably.
"""
import jax

jax.config.update("jax_enable_x64", True)

from .api import TrackFit, fit_track
from .inference import fit_batch_svi, fit_table_nuts, fit_table_svi, sample_posterior
from .likelihood import displacement_covariance, fgn_covariance, fgn_gamma, noise_covariance
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
