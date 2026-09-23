"""Distribution of D across many tracks: nonparametric MLE by EM, Gaussian-smoothed.

Population-level comparator to the per-track posteriors in `posterior.py`, not a
replacement for them -- per-track posteriors stay the primary output of this
package; this reports how D is distributed across a whole table of tracks, the
way an ensemble MSD fit would, but without averaging away each track's own
uncertainty first (it uses each track's likelihood, not its posterior, so the
prior is not counted once per track). See prototypes/posterior_1d.py
(`deconvolve`, the function this module ports) and prototypes/README.md's
"Population level" section for the derivation and calibration checks.

`deconvolve`'s objective, `sum_i log sum_D L_i(D) g(D)`, is only concave, not
strictly, so its unregularized optimum can put mass on a handful of grid
cells; `smooth` (grid cells) guards against that with a Gaussian blur each EM
step -- read a deconvolved peak's width as resolution-limited, not measured,
not as a calibrated uncertainty. `prototypes/maxent_deconvolve.py` explored
choosing this regularization strength from the data itself (Gull & Skilling
evidence) instead of the fixed `smooth` constant, but on synthetic data it
underperformed this simpler method and its evidence curve did not show a
clear interior maximum, so the Gaussian-smoothed EM below remains the one
wired in here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter1d

from ..data import Acquisition
from ..io import validate_table_schema, validated_track_frame
from ..validation import validate_acquisition
from .data import GridPostOptions
from .posterior import flat, posterior, track_loglik


@dataclass(frozen=True)
class PopulationDistribution:
    """Distribution of D across a table of tracks: grid weights (sum to 1) on `u`."""

    u: np.ndarray
    weights: np.ndarray
    n_tracks: int  # tracks that contributed a likelihood
    n_excluded: int  # too short (< options.min_frames) or invalid input
    acquisition: Acquisition


def deconvolve(lls: np.ndarray, log_prior: np.ndarray, iters: int = 500, smooth: float = 0.5) -> np.ndarray:
    """Distribution of D across tracks, as grid weights summing to 1, by EM on the per-track likelihoods.

    Each iteration replaces g by the average of the tracks' posteriors under the current g.
    One iteration from the flat start with smooth=0 is exactly the plain sum of per-track
    posteriors; more iterations remove the blur that sum carries. `smooth` (grid cells) is a
    Gaussian smoothing per iteration, which keeps the nonparametric maximum from going
    spiky. Locations and the mass in each mode are robust to it; peak widths are not
    (unsmoothed EM keeps sharpening toward spikes, smoothing widens), so read a width as
    resolution-limited, not measured. Only the support of log_prior is used, plus its
    shape as the starting point.
    """
    L = np.exp(lls - lls.max(axis=1, keepdims=True))
    support = np.isfinite(log_prior)
    g = posterior(np.zeros(L.shape[1]), log_prior)
    for _ in range(iters):
        r = L * g
        g = (r / r.sum(axis=1, keepdims=True)).mean(axis=0)
        if smooth:
            g = gaussian_filter1d(g, smooth, mode="constant") * support
            g /= g.sum()
    return g


def deconvolve_tracks(
    table: pl.DataFrame,
    acquisition: Acquisition,
    log_prior: np.ndarray | None = None,
    options: GridPostOptions = GridPostOptions(),
    iters: int = 500,
    smooth: float = 0.5,
) -> PopulationDistribution:
    """Distribution of D across every track in `table`, on `options.u_D()`.

    `log_prior` (on that grid) defaults to `flat`, matching `track_posterior`'s own default.
    Tracks shorter than `options.min_frames` or with invalid input (schema,
    non-contiguous frames, non-positive localization SD) are skipped and
    counted in `n_excluded`, mirroring `analyze_tracks`'s tolerance for bad
    individual tracks; a malformed table itself still raises.
    """
    validate_acquisition(acquisition, allow_exposure=True)
    validate_table_schema(table)
    u = options.u_D()
    prior = flat(u) if log_prior is None else log_prior
    groups = table.sort("track_id", "frame").partition_by("track_id", maintain_order=True)

    lls, n_excluded = [], 0
    for group in groups:
        try:
            track = validated_track_frame(group, acquisition, require_localization=True)
            if track.height < options.min_frames:
                n_excluded += 1
                continue
            lls.append(track_loglik(track, acquisition, u))
        except ValueError:
            n_excluded += 1

    if not lls:
        raise ValueError("no track had enough frames and valid input to contribute")

    weights = deconvolve(np.array(lls), prior, iters=iters, smooth=smooth)
    return PopulationDistribution(u, weights, len(lls), n_excluded, acquisition)
