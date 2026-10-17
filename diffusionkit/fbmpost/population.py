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

import warnings

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
# `deconvolve_tracks`' coarse pass: log10 D_ref cells about this wide (whole multiples of the likelihoods' cells),
_COARSE_D = .2
# its support: alpha rows and D_ref columns cut from either end while their margin's mass stays under this,
_SUPPORT_MASS = 1e-4
# widened by this many coarse cells (D_ref) and alpha cells on each side, and again whenever the fine fit puts
# more than `_EDGE_MASS` on an edge of it that is not an edge of the range.
_SUPPORT_MARGIN = 2
_EDGE_MASS = 1e-3


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


def _support(w: np.ndarray, mass: float) -> tuple[int, int]:
    """[lo, hi) of the entries of the margin `w` left after cutting each end while its cut stays under `mass`."""
    c = np.cumsum(w)
    lo = int(np.searchsorted(c, mass, side="right"))
    hi = len(w) - int(np.searchsorted(np.cumsum(w[::-1]), mass, side="right"))
    return min(lo, hi - 1), max(hi, lo + 1)


def _sheared(g: np.ndarray, alphas: np.ndarray, D: np.ndarray, log10_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """g (..., n_alpha, n_D) on log10 D at one lag carried to log10 D at another, and the new D cell centres.

    D' = D (tau' / tau)^(alpha - 1), so each alpha row moves by (alpha - 1) log10(tau' / tau) along log10 D:
    a cell's mass goes to the two cells of the same lattice it then overlaps, in proportion. Alpha's margin
    is unchanged.
    """
    d = D[1] - D[0]
    shift = (alphas - 1) * log10_ratio / d
    m = np.floor(shift).astype(int)
    f = shift - m
    lo, n_d = int(m.min()), len(D)
    out = np.zeros(g.shape[:-1] + (n_d + int(m.max()) - lo + 1,))
    for i in range(len(alphas)):
        k = m[i] - lo
        out[..., i, k:k + n_d] += (1 - f[i]) * g[..., i, :]
        out[..., i, k + 1:k + n_d + 1] += f[i] * g[..., i, :]
    return out, D[0] + d * (lo + np.arange(out.shape[-1]))


def deconvolve_tracks(likelihoods: JointLikelihoods, n_samples: int = 1000,
                      rng: np.random.Generator | None = None) -> AlphaDPopulation:
    """The distribution of (alpha, log10 D_ref) across `likelihoods`' tracks (`analyze_tracks(...,
    keep_likelihoods=True).likelihoods`, or a `rows` selection of it), fitted on `informative_range`.

    Coarse to fine: `gridpost.deconvolve_2d` first on cells `_COARSE_D` wide (each the likelihood averaged
    over the fine cells it holds, as `likelihood.cells` averages), then on the fine cells, but only over the
    box where the coarse fit puts all but `_SUPPORT_MASS` of its mass, widened by `_SUPPORT_MARGIN` cells
    (and again while the fine fit has mass on an inner edge). Outside the box g is 0. The range a population
    is fitted on is set by the least informative tracks -- a 3-frame track allows almost any D_ref at some
    alpha -- and the cell width by the most informative, so a single fine fit over the range costs ~10x as
    much for the same result.
    """
    if not len(likelihoods.track_ids):
        raise ValueError("no track to deconvolve")
    keep = informative_range(likelihoods)
    D = likelihoods.log10_D_ref[keep]
    ll = likelihoods.loglik[:, :, keep]
    a = likelihoods.alphas
    r = max(1, int(round(_COARSE_D / (D[1] - D[0])))) if len(D) > 1 else 1
    n_c = len(D) // r
    if r > 1 and n_c >= 3 * (_SUPPORT_MARGIN + 1):
        # The coarse pass: whole groups of r fine cells, from the top (the bottom cell stands for everything
        # below detection anyway).
        off = len(D) - n_c * r
        g = ll[:, :, off:].reshape(len(ll), len(a), n_c, r)
        top = g.max(axis=3, keepdims=True)
        coarse = (top + np.log(np.exp(g - top).mean(axis=3, keepdims=True)))[..., 0]
        with warnings.catch_warnings():  # only its support is used: a rough-edge lam there says nothing
            warnings.simplefilter("ignore", UserWarning)
            cfit = deconvolve_2d(coarse, a, D[off:].reshape(n_c, r).mean(axis=1), n_samples=0)
        a0, a1 = _support(cfit.weights.sum(axis=1), _SUPPORT_MASS)
        c0, c1 = _support(cfit.weights.sum(axis=0), _SUPPORT_MASS)
        a0, a1 = max(0, a0 - _SUPPORT_MARGIN), min(len(a), a1 + _SUPPORT_MARGIN)
        d0 = off + (c0 - _SUPPORT_MARGIN) * r if c0 > _SUPPORT_MARGIN else 0
        d1 = min(len(D), off + (c1 + _SUPPORT_MARGIN) * r)
    else:
        a0, a1, d0, d1 = 0, len(a), 0, len(D)
    while True:
        fit = deconvolve_2d(ll[:, a0:a1, d0:d1], a[a0:a1], D[d0:d1], n_samples=n_samples, rng=rng)
        w = fit.weights
        grow = [w[0].sum() > _EDGE_MASS and a0 > 0, w[-1].sum() > _EDGE_MASS and a1 < len(a),
                w[:, 0].sum() > _EDGE_MASS and d0 > 0, w[:, -1].sum() > _EDGE_MASS and d1 < len(D)]
        if not any(grow):
            break
        step = _SUPPORT_MARGIN * r
        a0, a1 = (max(0, a0 - _SUPPORT_MARGIN) if grow[0] else a0), (min(len(a), a1 + _SUPPORT_MARGIN) if grow[1] else a1)
        d0, d1 = (max(0, d0 - step) if grow[2] else d0), (min(len(D), d1 + step) if grow[3] else d1)
    ratio = float(np.log10(likelihoods.tau_ref_s / likelihoods.tau_s))
    weights, D_ref = _sheared(fit.weights, a[a0:a1], D[d0:d1], ratio)
    samples, _ = _sheared(fit.samples, a[a0:a1], D[d0:d1], ratio)
    floor = float(np.median(np.log10(likelihoods.D_ref_floor_um2_s)))
    return AlphaDPopulation(a[a0:a1], D_ref, weights, samples, fit.lam, len(likelihoods.track_ids), floor)
