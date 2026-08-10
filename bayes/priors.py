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
    """Prior for the 3-parameter (D_alpha, sigma, alpha) fGn model."""

    log_D_mean: float = float(np.log(0.05))
    log_D_sd: float = float(np.log(10) * 2)
    log_sigma_mean: float = float(np.log(0.025))
    log_sigma_sd: float = 0.5
    alpha_conc: float = 2.0  # Beta(2,2)/2 -- mildly regularizing toward alpha=1


@dataclass(frozen=True)
class AnisotropicModelPrior:
    """Prior for the 4-parameter (D_mean, eps, psi, sigma) anisotropic
    Brownian model (`model.anisotropic_diffusion_model`). D_mean/sigma
    follow the same LogNormal convention as `NormalModelPrior`.

    eps = (D_par-D_perp)/(D_par+D_perp) in [0,1) is the anisotropy fraction;
    eps_a/eps_b parameterize a Beta(eps_a, eps_b) prior that by default
    (Beta(1,3)) shrinks toward eps=0 (isotropy) rather than being flat or
    favoring anisotropy -- deliberately asymmetric, since at N=5 the null
    (isotropic) explanation should be favored a priori absent evidence, not
    treated as one of two equally-weighted options. This is the gap the
    original Gemini-drafted note left unaddressed for its analogous kappa
    concentration parameter (no prior was ever specified there).

    psi (orientation, radians in [0, pi) -- a diffusion tensor axis has 180
    degree symmetry) has no free hyperparameters: `model.py` always samples
    it Uniform(0, pi), since there's no reason to prefer one orientation
    over another a priori.

    eps_b=3 is a deliberately conservative default, not a tuned optimum --
    FINDINGS.md ("Anisotropy detection") found that at N=5 no single eps_b
    gives both a low false-positive rate and useful per-track sensitivity
    (the sampling-noise floor of a 4-point 2D covariance estimate is simply
    too large), so this favors suppressing false positives on a truly
    isotropic track over detecting real anisotropy from one track's data
    alone. Read the full `eps` posterior (not a thresholded flag) and see
    FINDINGS.md before relying on this for anything other than a rough,
    per-track, honestly-wide interval -- population-level pooling (not yet
    implemented) is what actually fixes the sensitivity side of this
    trade-off.
    """

    log_D_mean: float = float(np.log(0.05))
    log_D_sd: float = float(np.log(10) * 2)
    eps_a: float = 1.0
    eps_b: float = 3.0
    log_sigma_mean: float = float(np.log(0.025))
    log_sigma_sd: float = 0.5


WEAK_NORMAL_PRIOR = NormalModelPrior(log_D_sd=float(np.log(10) * 4), log_sigma_sd=5.0)
WEAK_ANOMALOUS_PRIOR = AnomalousModelPrior(
    log_D_sd=float(np.log(10) * 4), log_sigma_sd=5.0, alpha_conc=1.0
)
WEAK_ANISOTROPIC_PRIOR = AnisotropicModelPrior(
    log_D_sd=float(np.log(10) * 4), eps_a=1.0, eps_b=1.0, log_sigma_sd=5.0
)


def sigma_prior_from_localization(
    x_std_um: np.ndarray, y_std_um: np.ndarray, min_sd: float = 0.15
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
    `inference.fit_all_tracks` groups tracks by shared track_length and fits
    each group in one batched call (see model.py), so this needs to produce
    a per-track prior array, not just a per-track scalar, without a separate
    code path for the two cases.

    sd reflects how many frames that average rests on (fewer frames ->
    noisier variance estimate -> wider prior), floored at `min_sd` so short
    tracks don't get an overconfident prior on sigma.
    """
    n = x_std_um.shape[-1]
    combined_var_um2 = 0.5 * (np.mean(x_std_um**2, axis=-1) + np.mean(y_std_um**2, axis=-1))
    mean_log_sigma = 0.5 * np.log(combined_var_um2)
    sd = max(min_sd, 1.0 / np.sqrt(2 * n))  # same n_frames for every track in a group -> one scalar sd
    return mean_log_sigma, sd
