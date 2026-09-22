"""Per-track posterior P(alpha | displacements), K marginalized out as a nuisance.

Companion to `gridpost.posterior`'s grid posterior over D: where D probes the
data with the simplest model (alpha=1, ordinary Brownian motion) to ask "how
big are the steps", this asks a genuinely different question -- "how are
consecutive steps correlated" -- by fitting the fBm model and integrating the
generalized diffusion coefficient K out entirely, rather than reporting a
joint (K, alpha) point that inherits their well-known MLE degeneracy. D and
alpha are deliberately two independent 1D measurements of the same track, not
two coordinates of one joint fit.

Model, per axis: m = n - 1 displacements d ~ N(0, K A(alpha) + B), with
A(alpha) the fGn covariance (`likelihood.fgn_motion_covariance`) and B the
known per-frame localization-noise covariance (`likelihood.localization_covariance`,
same as `gridpost.posterior`). For any *fixed* alpha this is linear in K
exactly as `gridpost.posterior`'s model is linear in D, so the same whitening
trick applies -- but A(alpha) itself changes shape with alpha, so (unlike D)
each alpha grid point needs its own eigendecomposition: O(alpha-grid-size)
decompositions per track instead of one. Track lengths here stay short
enough that this is a bounded, linear cost, not a bottleneck.

No exposure-blur model: `motion_covariance`'s closed-form Berglund R average
is specific to alpha=1's linear-motion double integral; no comparably simple
closed form exists at a general alpha. Callers must use `exposure_s=0`
(`diffusionkit.bayes.model`'s NumPyro anomalous model makes the same
simplification for the same reason).

The alpha posterior is the 1D marginal of the 2D (alpha, ln K) log-likelihood
surface: integrating a nuisance parameter out is exactly `logsumexp` over its
axis, using `log_K_prior` as K's own (hand-set, not empirical-Bayes) prior --
reuse `gridpost.posterior.flat`/`log_uniform`/`log_normal` for it, since those
are generic log-scale grid priors, not specific to being called D.

Started as, and still numerically matches, `prototypes/posterior_alpha.py`.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.special import logsumexp

from ..data import Acquisition
from .likelihood import _loglik, _prepared, _whiten_axis, fgn_motion_covariance, localization_covariance
from .posterior import _grid_quantile, _normalize
from .posterior import flat as flat_K  # noqa: F401  (re-exported: a generic log-scale-grid prior)

# alpha grid: avoid the exact 0/2 edges, where the fGn covariance degenerates.
ALPHA = np.linspace(0.05, 1.95, 39)
# ln K grid, K in um^2/s^alpha: 1e-4 to 10, matching gridpost.posterior.U's D grid.
U = np.linspace(np.log(1e-4), np.log(10.0), 251)


def _whiten_alpha(track: pl.DataFrame, acquisition: Acquisition, alpha: float) -> dict:
    """Whitened data for one validated track at a fixed alpha. Requires exposure_s=0."""
    if acquisition.exposure_s > 0:
        raise ValueError("The alpha posterior has no exposure-blur model; requires exposure_s=0")
    positions = track.select("x_um", "y_um").to_numpy()
    sd = track.select("sigma_x_um", "sigma_y_um").to_numpy()
    delta = np.diff(positions, axis=0).T  # (2, m)
    m = delta.shape[1]
    A = fgn_motion_covariance(m, float(acquisition.dt_s), alpha)
    lam, y, logdet_B = [], [], 0.
    for axis, B in enumerate(localization_covariance(sd)):
        values, yy, ld = _whiten_axis(delta[axis], A, B)
        lam.append(values)
        y.append(yy)
        logdet_B += ld
    const = -.5 * logdet_B - m * np.log(2 * np.pi)
    return {"lam": np.array(lam), "y": np.array(y), "const": const}


def track_loglik_given_alpha(track: pl.DataFrame, acquisition: Acquisition, alpha: float,
                             u: np.ndarray = U) -> np.ndarray:
    """(len(u),) log-likelihood of one track's displacements at K = exp(u), fixed alpha."""
    w = _whiten_alpha(_prepared(track, acquisition), acquisition, alpha)
    return _loglik(np.exp(u), w["lam"], w["y"][None], w["const"])


def joint_loglik(track: pl.DataFrame, acquisition: Acquisition, alphas: np.ndarray = ALPHA,
                 u: np.ndarray = U) -> np.ndarray:
    """(len(alphas), len(u)) log-likelihood surface over (alpha, ln K)."""
    return np.array([track_loglik_given_alpha(track, acquisition, a, u) for a in alphas])


# --------------------------------------------------------------------------
# Posterior over alpha: marginalize the nuisance ln K
# --------------------------------------------------------------------------


def flat_alpha(alphas: np.ndarray = ALPHA) -> np.ndarray:
    """Flat over the alpha grid -- the least-informative default."""
    return np.zeros_like(alphas)


def alpha_posterior(joint_ll: np.ndarray, log_K_prior: np.ndarray,
                    log_alpha_prior: np.ndarray | None = None) -> np.ndarray:
    """(len(alphas),) posterior over alpha: add priors, integrate ln K out, normalize.

    `joint_ll` is (len(alphas), len(u)) from `joint_loglik`. Integrating a
    nuisance parameter out is exactly `logsumexp` over its axis (a Riemann
    sum in ln K; the grid step is an additive constant that normalization
    removes).
    """
    log_mass = logsumexp(joint_ll + log_K_prior[None, :], axis=1)
    if log_alpha_prior is not None:
        log_mass = log_mass + log_alpha_prior
    return _normalize(log_mass)


def quantile(p: np.ndarray, q: float, alphas: np.ndarray = ALPHA) -> float:
    """Posterior q-quantile of alpha, interpolating the CDF at cell midpoints."""
    return _grid_quantile(p, alphas, q)


def summary(p: np.ndarray, alphas: np.ndarray = ALPHA, level: float = .9) -> dict[str, float]:
    """Median and equal-tailed credible interval of alpha."""
    return {
        "median": quantile(p, .5, alphas),
        "lo": quantile(p, (1 - level) / 2, alphas),
        "hi": quantile(p, (1 + level) / 2, alphas),
    }


def track_alpha_posterior(track: pl.DataFrame, acquisition: Acquisition, log_K_prior: np.ndarray | None = None,
                          alphas: np.ndarray = ALPHA, u: np.ndarray = U, level: float = .9) -> dict[str, float]:
    """Posterior median and `level` credible interval of alpha for one track.

    `log_K_prior` defaults to `flat_K()` (flat in ln K over the whole grid) --
    the least-informative choice for the nuisance parameter, no
    empirical-Bayes fitting across tracks. Raises if `acquisition.exposure_s`
    is nonzero.
    """
    prior = flat_K(u) if log_K_prior is None else log_K_prior
    ll = joint_loglik(track, acquisition, alphas, u)
    return summary(alpha_posterior(ll, prior), alphas, level)
