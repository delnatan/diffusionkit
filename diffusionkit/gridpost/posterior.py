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
import polars as pl
from scipy.special import logsumexp

from ..data import Acquisition
from .data import GridPostOptions
from .likelihood import _loglik, _prepared, _whiten

# The grid is `GridPostOptions.u_D()`: every function below takes it
# explicitly, so no run can silently use a range other than the one its
# options record.


def track_loglik(track: pl.DataFrame, acquisition: Acquisition, u: np.ndarray) -> np.ndarray:
    """(len(u),) log-likelihood of one track's displacements at D = exp(u)."""
    w = _whiten(_prepared(track, acquisition), acquisition)
    return _loglik(np.exp(u), w["lam"], w["y"][None], w["const"])


# --------------------------------------------------------------------------
# Priors: log-density on the grid, up to a constant
# --------------------------------------------------------------------------


def flat(u: np.ndarray) -> np.ndarray:
    """Flat in ln D over the whole grid -- the least-informative default."""
    return np.zeros_like(u)


def log_uniform(D_lo: float, D_hi: float, u: np.ndarray) -> np.ndarray:
    """Flat in ln D between the limits, impossible outside them."""
    return np.where((u >= np.log(D_lo)) & (u <= np.log(D_hi)), 0.0, -np.inf)


def log_normal(D_lo: float, D_hi: float, u: np.ndarray) -> np.ndarray:
    """Log-normal whose central 95% spans [D_lo, D_hi]: the same limits with soft edges."""
    mu, sd = .5 * (np.log(D_lo) + np.log(D_hi)), (np.log(D_hi) - np.log(D_lo)) / (2 * 1.96)
    return -.5 * ((u - mu) / sd) ** 2


# --------------------------------------------------------------------------
# Posterior and summaries
# --------------------------------------------------------------------------


def _normalize(log_weights: np.ndarray) -> np.ndarray:
    """Grid weights (sum to 1) from log un-normalized weights.

    Shared with `gridpost.posterior_alpha`, whose alpha grid is linear rather
    than log-scale -- this step (exp, shift for stability, sum to 1) doesn't
    care which quantity the grid represents.
    """
    p = np.exp(log_weights - log_weights.max())
    return p / p.sum()


def _grid_quantile(p: np.ndarray, grid: np.ndarray, q: float) -> float:
    """q-quantile of `grid`'s distribution `p`, interpolating the CDF at cell midpoints.

    Also shared with `gridpost.posterior_alpha`; this module's own `quantile`
    below is the D-specific (log-scale grid, exponentiated result) wrapper.
    """
    return float(np.interp(q, np.cumsum(p) - p / 2, grid))


def posterior(ll: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """Grid weights (sum to 1) proportional to likelihood x prior."""
    return _normalize(ll + log_prior)


def log_posterior(ll: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """`posterior`'s weights as logs, computed without the round trip through exp
    (so far-tail cells stay finite instead of underflowing to -inf)."""
    lw = ll + log_prior
    return lw - logsumexp(lw)


def information_bits(log_post: np.ndarray, log_prior: np.ndarray) -> float:
    """What the track taught about D: relative entropy KL(posterior || prior), in bits.

    Both are taken as distributions over the grid points (the prior is
    normalized here), which approximates the continuous relative entropy
    whenever the grid resolves the posterior. 0 means the data left the prior
    unchanged; each further bit is worth about halving the plausible range of
    ln D. It is invariant to reparametrizing D, but relative to the prior, so
    bits are comparable only between runs on the same grid range (a
    localization-limited track, which only bounds D from above, gains
    whatever fraction of the prior below its bound it rules out).
    """
    lq = log_posterior(np.zeros_like(log_prior), log_prior)
    p = np.exp(log_post)
    keep = p > 0
    return float(np.sum(p[keep] * (log_post[keep] - lq[keep])) / np.log(2))


def edge_ratios(p: np.ndarray) -> tuple[float, float]:
    """Posterior weight at the grid's first and last point, each relative to its peak.

    Near 0: the posterior has died out inside the grid. Not small: it is cut
    by the grid edge, i.e. limited there by the prior's support rather than
    by the data, so its median and interval depend on where that edge is.
    (Localization-limited, near-immobile tracks reach the low edge this way:
    the data only bound D from above.)
    """
    peak = p.max()
    return float(p[0] / peak), float(p[-1] / peak)


# `edge_ratios` above this flags a track's D posterior as cut by the grid.
EDGE_RATIO_WARN = .05


def quantile(p: np.ndarray, q: float, u: np.ndarray) -> float:
    """Posterior q-quantile of D, interpolating the CDF at cell midpoints."""
    return float(np.exp(_grid_quantile(p, u, q)))


def summary(p: np.ndarray, u: np.ndarray, level: float = .9) -> dict[str, float]:
    """Median and equal-tailed credible interval of D, in um^2/s."""
    return {
        "median": quantile(p, .5, u),
        "lo": quantile(p, (1 - level) / 2, u),
        "hi": quantile(p, (1 + level) / 2, u),
    }


def track_posterior(track: pl.DataFrame, acquisition: Acquisition, log_prior: np.ndarray | None = None,
                    options: GridPostOptions = GridPostOptions()) -> dict[str, float]:
    """Posterior median and `options.level` credible interval of D for one track.

    Evaluated on `options.u_D()`. `log_prior` (on that grid) defaults to
    `flat` -- the least-informative choice, no empirical-Bayes fitting across
    tracks. Pass `log_uniform`/`log_normal` (physical-limit bounds) for an
    informative prior instead.
    """
    u = options.u_D()
    prior = flat(u) if log_prior is None else log_prior
    return summary(posterior(track_loglik(track, acquisition, u), prior), u, options.level)
