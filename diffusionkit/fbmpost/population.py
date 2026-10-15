"""alpha across tracks: one alpha shared by all (`shared_alpha`) and the distribution of (alpha, D_ref).

One shared alpha is the maximum of the tracks' summed profiles, each track keeping its own amplitude
(complete pooling, `gridpost`'s shared D for alpha). A distribution of alpha cannot be fitted to the
profiles, because they say nothing about how alpha and D_ref vary together across tracks. Here the
amplitude is a second population axis: each track's likelihood on (alpha, log10 D_ref) cells
(`likelihood.cells`), with no prior in it, and `gridpost.deconvolve_2d` -- the smooth log density of
`gridpost.deconvolve` on a 2D grid -- fits g(alpha, log10 D_ref). Alpha's population is g's margin, and the D_ref
population it is marginalized under is the fitted one, not a per-track prior.

Below the floor the tracks cannot tell D_ref values apart, so g's alpha there is mostly the smoothness prior's
continuation of alpha above it. `AlphaDPopulation`'s `above_floor` reads leave that part out.
"""
from __future__ import annotations

import numpy as np

from scipy.interpolate import CubicSpline
from scipy.stats import chi2

from ..gridpost.deconvolve import deconvolve_2d
from .data import AlphaDPopulation, AlphaProfiles, JointLikelihoods, SharedAlpha

# `shared_alpha` interpolates the summed profile at this many points per grid step.
_REFINE = 50
# `informative_range`: a cell is above every track when each track's likelihood there, at its best alpha, is this
# far below its maximum (a 5-frame track loses ~9 nats per decade of D_ref),
_TOP_NATS = 5.
# and below detection when every track's likelihood there is within this (of its maximum, linear scale) of its
# value in the lowest cell. Below the floor the likelihood nears that plateau linearly in K, a tenth per decade, so
# a stricter value only adds cells that tell nothing about alpha.
_PLATEAU = 1e-2
# Cells of margin kept on either side.
_MARGIN = 5


def shared_alpha(profiles: AlphaProfiles, level: float = .95) -> SharedAlpha:
    """One alpha for every track in `profiles` (`analyze_tracks(..., keep_profiles=True).profiles`, or a `rows`
    selection): the maximum of the summed profile and its likelihood-ratio interval at `level`.

    The sum is smooth in alpha and, with many tracks, narrower than a grid step, so it is read off a cubic
    spline through the grid at `_REFINE` points per step.
    """
    if not len(profiles.track_ids):
        raise ValueError("no track to pool")
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    grid = profiles.alphas
    total = profiles.loglik.sum(axis=0)
    fine = np.linspace(grid[0], grid[-1], (len(grid) - 1) * _REFINE + 1)
    curve = CubicSpline(grid, total)(fine)
    top = curve.max()
    inside = np.flatnonzero(curve >= top - chi2.ppf(level, 1) / 2)
    one = float(CubicSpline(grid, total)(1.)) if grid[0] <= 1 <= grid[-1] else np.nan
    dll = max(top - one, 0.)
    return SharedAlpha(float(fine[curve.argmax()]), float(fine[inside[0]]), float(fine[inside[-1]]), dll,
                       float(chi2.sf(2 * dll, 1)), len(profiles.track_ids), level,
                       bool(inside[0] == 0 or inside[-1] == len(fine) - 1), fine, curve - top)


def informative_range(likelihoods: JointLikelihoods) -> slice:
    """The log10 D_ref cells some track tells apart, with `_MARGIN` cells either side.

    Above it every track's likelihood is at least `_TOP_NATS` below its maximum; below it every
    track's is flat to `_PLATEAU` (the noise-only plateau, the same at every alpha), so the lowest
    cell kept stands for all of it.
    """
    ll = likelihoods.loglik
    peak = ll.max(axis=1)  # (n_tracks, n_D): best alpha per D_ref cell
    hi = int(np.flatnonzero((peak > -_TOP_NATS).any(axis=0)).max())
    L = np.exp(ll)
    moved = (np.abs(L - L[:, :, :1]).max(axis=1) > _PLATEAU).any(axis=0)
    lo = int(np.flatnonzero(moved).min()) if moved.any() else 0
    n_d = ll.shape[2]
    lo, hi = max(0, min(lo, hi) - _MARGIN), min(n_d, hi + _MARGIN + 1)
    if hi - lo < 3:
        lo, hi = max(0, hi - 3), max(hi, 3)
    return slice(lo, hi)


def deconvolve_tracks(likelihoods: JointLikelihoods, n_samples: int = 1000,
                      rng: np.random.Generator | None = None) -> AlphaDPopulation:
    """The distribution of (alpha, log10 D_ref) across `likelihoods`' tracks (`analyze_tracks(...,
    keep_likelihoods=True).likelihoods`, or a `rows` selection of it), fitted on `informative_range`.

    Seconds for a thousand tracks; the cost grows with the number of cells the range keeps.
    """
    if not len(likelihoods.track_ids):
        raise ValueError("no track to deconvolve")
    keep = informative_range(likelihoods)
    D = likelihoods.log10_D_ref[keep]
    fit = deconvolve_2d(likelihoods.loglik[:, :, keep], likelihoods.alphas, D, n_samples=n_samples, rng=rng)
    floor = float(np.median(np.log10(likelihoods.D_ref_floor_um2_s)))
    return AlphaDPopulation(likelihoods.alphas, D, fit.weights, fit.samples, fit.lam, len(likelihoods.track_ids),
                            floor)
