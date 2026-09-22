"""Per-track posterior P(D | displacements) for Brownian motion.

Built on `likelihood.py`'s whitening: after whitening, a track's
displacements are independent y_k ~ N(0, 1 + D lam_k). The "1 +" is
localization noise, so no prior on D is conjugate; instead the posterior is
evaluated exactly on a fixed grid in u = ln D, where each point costs O(m)
per track.

The posterior lives on a fixed grid in u = ln D, so a prior that is flat in
u is log-uniform in D (scale-invariant), and the weights are just
exp(ln L + ln prior), normalized. This is the per-track information to
report for D: a track's own uncertainty stays visible as how narrow its
posterior is, rather than being collapsed into a point estimate. A short,
uninformative track producing a wide posterior is an honest answer, not a
defect (see prototypes/posterior_1d.py, prototypes/README.md).
"""
from __future__ import annotations

import numpy as np

from ..data import Acquisition, Track
from .likelihood import _loglik, _prepared, _whiten

# ln D grid, D in um^2/s: 1e-4 to 10 in ~2.3% steps.
U = np.linspace(np.log(1e-4), np.log(10.0), 501)


def track_loglik(track: Track, acquisition: Acquisition, u: np.ndarray = U) -> np.ndarray:
    """(len(u),) log-likelihood of one track's displacements at D = exp(u)."""
    w = _whiten(_prepared(track, acquisition), acquisition)
    return _loglik(np.exp(u), w["lam"], w["y"][None], w["const"])


# --------------------------------------------------------------------------
# Priors: log-density on the grid, up to a constant
# --------------------------------------------------------------------------


def flat(u: np.ndarray = U) -> np.ndarray:
    """Flat in ln D over the whole grid -- the least-informative default."""
    return np.zeros_like(u)


def log_uniform(D_lo: float, D_hi: float, u: np.ndarray = U) -> np.ndarray:
    """Flat in ln D between the limits, impossible outside them."""
    return np.where((u >= np.log(D_lo)) & (u <= np.log(D_hi)), 0.0, -np.inf)


def log_normal(D_lo: float, D_hi: float, u: np.ndarray = U) -> np.ndarray:
    """Log-normal whose central 95% spans [D_lo, D_hi]: the same limits with soft edges."""
    mu, sd = .5 * (np.log(D_lo) + np.log(D_hi)), (np.log(D_hi) - np.log(D_lo)) / (2 * 1.96)
    return -.5 * ((u - mu) / sd) ** 2


# --------------------------------------------------------------------------
# Posterior and summaries
# --------------------------------------------------------------------------


def posterior(ll: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """Grid weights (sum to 1) proportional to likelihood x prior."""
    lp = ll + log_prior
    p = np.exp(lp - lp.max())
    return p / p.sum()


def quantile(p: np.ndarray, q: float, u: np.ndarray = U) -> float:
    """Posterior q-quantile of D, interpolating the CDF at cell midpoints."""
    return float(np.exp(np.interp(q, np.cumsum(p) - p / 2, u)))


def summary(p: np.ndarray, u: np.ndarray = U, level: float = .9) -> dict[str, float]:
    """Median and equal-tailed credible interval of D, in um^2/s."""
    return {
        "median": quantile(p, .5, u),
        "lo": quantile(p, (1 - level) / 2, u),
        "hi": quantile(p, (1 + level) / 2, u),
    }


def track_posterior(track: Track, acquisition: Acquisition, log_prior: np.ndarray | None = None,
                    u: np.ndarray = U, level: float = .9) -> dict[str, float]:
    """Posterior median and `level` credible interval of D for one track.

    `log_prior` defaults to `flat()` -- the least-informative choice, no
    empirical-Bayes fitting across tracks. Pass `log_uniform`/`log_normal`
    (physical-limit bounds) for an informative prior instead.
    """
    prior = flat(u) if log_prior is None else log_prior
    return summary(posterior(track_loglik(track, acquisition, u), prior), u, level)
