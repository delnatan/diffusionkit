"""The distribution of (alpha, D_ref) across tracks, from their joint likelihoods.

A track's alpha curve is a posterior: the amplitude is summed out under its flat ln SNR prior, which near the
localization floor tilts the curve. A population of alpha built from those curves inherits the tilt (docs/fbmpost.md,
"Population"). Here the amplitude is a second population axis instead: each track's likelihood on (alpha, log10
D_ref) cells (`joint_cells`), with no prior in it, and `gridpost.deconvolve_2d` -- the smooth log density of
`gridpost.deconvolve` on a 2D grid -- fits g(alpha, log10 D_ref). Alpha's population is g's margin, and the D_ref
population it is marginalized under is the fitted one, not a per-track prior.

Below the floor the tracks cannot tell D_ref values apart, so g's alpha there is mostly the smoothness prior's
continuation of alpha above it. `AlphaDPopulation`'s `above_floor` reads leave that part out.
"""
from __future__ import annotations

import numpy as np

from ..gridpost.deconvolve import deconvolve_2d
from .data import AlphaDPopulation, FbmPostOptions, JointGrid, JointLikelihoods
from .posterior import TrackGrid, log10_D_ref

# Cells further than this below a track's maximum are stored at it (e^-200: no mixture can tell).
_LOGLIK_FLOOR = -200.
# `informative_range`: a cell is above every track when each track's likelihood there, at its best alpha, is this
# far below its maximum (a 5-frame track loses ~9 nats per decade of D_ref),
_TOP_NATS = 5.
# and below detection when every track's likelihood there is within this (of its maximum, linear scale) of its
# value in the lowest cell. Below the floor the likelihood nears that plateau linearly in K, a tenth per decade, so
# a stricter value only adds cells that tell nothing about alpha.
_PLATEAU = 1e-2
# Cells of margin kept on either side.
_MARGIN = 5


def joint_cells(grid: TrackGrid, options: FbmPostOptions, joint: JointGrid) -> np.ndarray:
    """(n_alpha, n_D) log of a track's likelihood (`posterior.log_likelihood_grid`) averaged over each of
    `joint`'s cells, with its maximum at 0.

    Within a fine alpha row the ln SNR points are evenly spaced in log10 D_ref, so their mean over a cell is
    the cell's average. A cell the row's SNR range does not reach takes the nearest reached cell's value
    (below snr_min the likelihood is the noise-only plateau).
    """
    x = log10_D_ref(grid.ln_K, options)
    centres = joint.log10_D_ref()
    step = joint.d_log10_D_ref
    n_a, n_d = len(joint.alphas(options)), len(centres)
    edges = np.append(centres - step / 2, centres[-1] + step / 2)
    top = grid.loglik.max()
    lin = np.empty((n_a * joint.alpha_group, n_d))
    for j in range(len(lin)):
        b = np.digitize(x[j], edges) - 1
        ok = (b >= 0) & (b < n_d)
        s = np.bincount(b[ok], np.exp(grid.loglik[j, ok] - top), n_d)
        c = np.bincount(b[ok], minlength=n_d)
        reached = np.flatnonzero(c)
        if not len(reached):
            raise ValueError("the track's SNR grid misses every log10 D_ref cell: widen the JointGrid")
        near = reached[np.clip(np.searchsorted(reached, np.arange(n_d)), 0, len(reached) - 1)]
        lin[j] = s[near] / c[near]
    lin = lin.reshape(n_a, joint.alpha_group, n_d).mean(axis=1)
    with np.errstate(divide="ignore"):
        ll = np.log(lin)
    return np.maximum(ll - ll.max(), _LOGLIK_FLOOR)


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
