"""D at a second, longer timescale, and how D changes between the two.

The D posterior (`gridpost.posterior`) fits Brownian motion to consecutive
displacements, one frame interval dt apart. Fitting the same model to the
track thinned to every `stride`-th frame gives D at tau = stride * dt. For
Brownian motion the two agree; where they differ, the apparent diffusivity
D(tau) = MSD(tau) / (4 tau) depends on timescale, and the ratio says how,
without committing to a model of why:

  - D(tau) / D(dt) < 1: motion slows at longer times -- confinement, a
    crowded or elastic surrounding (fBm with alpha < 1 gives stride^(alpha-1),
    a particle confined within L gives ~ L^2 / (4 D tau) once tau > L^2 / D);
  - > 1: persistent or directed motion.

Both D's come from the exact likelihood: the thinned track keeps each
retained frame's localization SD, and its blur is the same exposure over a
frame interval stride * dt (`likelihood.motion_covariance`'s Berglund
average, R = exposure / (6 stride dt)).

Thinning at stride k leaves k interleaved sub-tracks (phases 0..k-1), all
of which carry information about D(tau), but they share the underlying path,
so their likelihoods are not independent and multiplying them would
overstate the certainty. `thinned_loglik` averages their log-likelihoods
instead (a composite likelihood with weight 1/k per phase): every frame
counts toward the estimate, while the posterior keeps about one phase's
width. That errs on the conservative side, which
`scripts/validate_D_timescale.py` checks by simulation.

The ratio's posterior (`ratio_posterior`) combines the two D posteriors as
if independent. They come from the same track and are positively
correlated, so that too widens rather than narrows the ratio's interval.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..data import Acquisition
from . import posterior as posterior_mod
from .likelihood import _loglik, _whiten


def thinned_loglik(track: pl.DataFrame, acquisition: Acquisition, u: np.ndarray, stride: int,
                   min_frames: int = 3) -> tuple[np.ndarray | None, int]:
    """(phase-averaged log-likelihood over D = exp(u) at tau = stride * dt, phases used).

    `track` is already validated (`likelihood._prepared`). Each phase p keeps
    frames p, p + stride, ...; phases with fewer than `min_frames` frames are
    skipped, and with none left the log-likelihood is None.
    """
    thinned = Acquisition(stride * float(acquisition.dt_s), float(acquisition.exposure_s))
    lls = []
    for phase in range(stride):
        sub = track[phase::stride]
        if sub.height < min_frames:
            continue
        w = _whiten(sub, thinned)
        lls.append(_loglik(np.exp(u), w["lam"], w["y"][None], w["const"]))
    if not lls:
        return None, 0
    return np.mean(lls, axis=0), len(lls)


def ratio_posterior(log_post_long: np.ndarray, log_post_short: np.ndarray,
                    u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(weights, ln-ratio grid): the posterior of ln(D_long / D_short), taking the two as independent.

    Both are normalized log posteriors on the same, evenly spaced ln D grid
    `u`, so their difference lives on the evenly spaced grid of offsets
    -(n-1) du .. (n-1) du, and its weights are the cross-correlation of the
    two weight vectors.
    """
    du = u[1] - u[0]
    weights = np.correlate(np.exp(log_post_long), np.exp(log_post_short), mode="full")
    weights = np.clip(weights, 0, None)
    grid = (np.arange(len(weights)) - (len(u) - 1)) * du
    return weights / weights.sum(), grid


def ratio_summary(weights: np.ndarray, grid: np.ndarray, level: float = .9) -> dict[str, float]:
    """Median and equal-tailed `level` interval of D_long / D_short, and P(D_long < D_short)."""
    q = [posterior_mod._grid_quantile(weights, grid, p) for p in ((1 - level) / 2, .5, (1 + level) / 2)]
    below = float(weights[grid < 0].sum() + weights[grid == 0].sum() / 2)
    return {"lo": float(np.exp(q[0])), "median": float(np.exp(q[1])), "hi": float(np.exp(q[2])),
            "p_decrease": below}
