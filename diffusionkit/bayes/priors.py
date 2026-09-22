"""Prior hyperparameters for the two numpyro models in `model.py`.

D/sigma priors are LogNormal, weakly informative -- the defaults below are a
reasonable starting point for typical particle-tracking D and localization
precision, meant to be overridden with dataset-appropriate values (e.g. from
an ensemble MSD estimate or the localization pipeline's own precision
figure) rather than treated as universal constants. They're anchored to
independent information (a rough D scale, measured localization precision),
never to a particular track's own MSD fit, so using them isn't smuggling MSD
back into the estimator. alpha's prior is Beta(conc, conc) rescaled to
(0, 2): conc=1 is exactly flat (Uniform(0,2)); conc>1 concentrates mildly
around alpha=1 without hard-constraining it -- picking the concentration
that best matches expected motion heterogeneity is a per-study judgment
call. Priors are stated directly on each parameter's own bounded support
(LogNormal for positive scale parameters, Beta for a (0,2)-bounded exponent)
so numpyro's automatic constrained<->unconstrained handling applies without
any hand-written transform.

`WEAK_*_PRIOR` presets (very wide D/sigma, alpha_conc=1 i.e. flat) carry
~no information, so a fit against them is essentially the likelihood argmax
-- a flat-prior MLE, using the same model and inference code as an
informative-prior Bayesian fit rather than a separate code path (see
`inference.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalModelPrior:
    """Prior for the 2-parameter (D, sigma) Brownian model."""

    log_D_mean: float = float(np.log(0.05))       # order-of-magnitude default; override per study
    log_D_sd: float = float(np.log(10) * 2)        # ~2 decades of D within 1 sd -- weakly informative
    log_sigma_mean: float = float(np.log(0.025))   # order-of-magnitude default; override per study
    log_sigma_sd: float = 0.5


@dataclass(frozen=True)
class AnomalousModelPrior:
    """Prior for the 3-parameter (K, sigma, alpha) fGn model."""

    log_D_mean: float = float(np.log(0.05))
    log_D_sd: float = float(np.log(10) * 2)
    log_sigma_mean: float = float(np.log(0.025))
    log_sigma_sd: float = 0.5
    alpha_conc: float = 2.0  # Beta(2,2)/2 -- mildly regularizing toward alpha=1


WEAK_NORMAL_PRIOR = NormalModelPrior(log_D_sd=float(np.log(10) * 4), log_sigma_sd=5.0)
WEAK_ANOMALOUS_PRIOR = AnomalousModelPrior(
    log_D_sd=float(np.log(10) * 4), log_sigma_sd=5.0, alpha_conc=1.0
)


def sigma_prior_from_localization(
    sigma_x_um: np.ndarray, sigma_y_um: np.ndarray, min_sd: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Informative (mean, sd) for log(sigma_loc), from a track's own
    independently-measured MLE localization precision.

    The Bayesian analog of the classic pipeline's
    `analysis.fitting.localization_offset_by_track` diagnostic -- instead of
    hard-subtracting a point estimate of the localization offset from MSD
    after the fact (which overcorrects if that estimate is noisy, since it's
    then treated as exact), it enters as a prior the likelihood can shrink
    toward or away from depending on what the trajectory itself says about
    sigma.

    mean = log(RMS combined x/y precision), averaged over the last axis, so
    a single track's 1D (n_frames,) arrays give scalars and a batch's 2D
    (n_tracks, n_frames) arrays give one (mean, sd) pair per track --
    `inference`'s table builders group tracks by shared track_length and fit
    each group in one batched call (see model.py), so this needs to produce
    a per-track prior array, not just a per-track scalar, without a separate
    code path for the two cases.

    sd reflects how many frames that average rests on (fewer frames ->
    noisier variance estimate -> wider prior), floored at `min_sd` so short
    tracks don't get an overconfident prior on sigma.
    """
    n = sigma_x_um.shape[-1]
    combined_var_um2 = 0.5 * (np.mean(sigma_x_um**2, axis=-1) + np.mean(sigma_y_um**2, axis=-1))
    mean_log_sigma = 0.5 * np.log(combined_var_um2)
    sd = max(min_sd, 1.0 / np.sqrt(2 * n))  # same n_frames for every track in a group -> one scalar sd
    return mean_log_sigma, sd
