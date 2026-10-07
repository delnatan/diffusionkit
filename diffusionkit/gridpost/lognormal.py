"""Population models with a formula: one D shared by every track, and a log-normal distribution of D.

Tracks are combined by adding log-likelihoods. For a population g of D across tracks, track i's
likelihood is its own likelihood averaged over g, and the tracks multiply:

    l(theta) = sum_i log int L_i(u) g_theta(u) du,   u = ln D

- **complete pooling** (`fit_shared_D`): g = delta(u - u0), every track has the same D. Then
  l = sum_i log L_i(u0), the summed log-likelihood. Its interval narrows as 1/sqrt(n) however much
  the tracks differ, and it weighs each track by how much it says about D, so long tracks dominate.
- **partial pooling, log-normal** (`fit_lognormal`): ln D ~ N(mu, sigma) across tracks, truncated to
  the grid. exp(mu) is the population's median D and sigma its spread in ln D, with each track's own
  measurement noise taken out by the model rather than averaged in. sigma = 0 is complete pooling,
  so sigma's posterior says whether one shared D describes the tracks.

`deconvolve` is the third member: partial pooling with a smooth g of any shape. What these are not
is an average of the tracks' posteriors (the sum is outside the log there, so it is no likelihood
of anything) or a histogram of their medians (a short track's median sits where the prior puts it).

The likelihood rows are the kept, normalized per-track log-likelihoods on the D grid
(`GridLikelihoods.loglik_D`); a constant per row does not matter. Between grid points a row is
taken as linear in L, so the integral against a normal is exact (`hat_weights`) for every sigma,
sigma = 0 included, and smooth in mu. The posterior over (mu, sigma) is evaluated on a grid, like
the per-track posteriors, with priors flat in mu over the D grid and flat in sigma on
[0, `sigma_max`]. The grid zooms in on where the posterior lives, so its resolution follows the
posterior's width.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from warnings import warn

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp, ndtr

from ..data import Acquisition
from .data import GridDistribution, GridPosteriorAnalysis

SIGMA_MAX = 3.  # sigma's prior bound in ln D: a 90% range of e^(2 * 1.645 * 3), over four decades
N_MU, N_SIGMA = 61, 41  # the (mu, sigma) posterior grid
DROP = 20.  # log-posterior drop that bounds the region the grid must hold (e^-20 of the peak)
MAX_ZOOM = 12  # zoom steps before the grid is taken as it is
TINY = 1e-300
_CHUNK = 5_000_000  # tracks x columns per matrix product, ~40 MB
_SHARED_WINDOW = 40.  # the shared posterior is splined over the cells within this many nats of its peak
_SHARED_REFINE = 64  # spline points per grid cell


def hat_weights(u: np.ndarray, mu, sigma) -> np.ndarray:
    """(len(u), M) weights g with `L @ g` = the integral of L against N(mu, sigma) truncated to the grid.

    L is taken as linear between the grid points (a sum of hat functions), so the integral is exact:
    each column sums to 1 and is a probability on the grid points. sigma = 0 gives the linear
    interpolation weights at mu. Tail cells use survival functions, so a track far from mu gets its
    tiny likelihood rather than a rounding floor. `mu` and `sigma` broadcast to M columns.
    """
    u = np.asarray(u, float)
    mu, sigma = (np.ravel(a).astype(float) for a in np.broadcast_arrays(mu, sigma))
    if np.any(sigma < 0):
        raise ValueError("sigma must be >= 0")
    du = u[1] - u[0]
    G = np.zeros((len(u), len(mu)))
    pos = sigma > 0
    if pos.any():
        m, s = mu[pos][None], sigma[pos][None]
        a = (u[:-1, None] - m) / s
        b = (u[1:, None] - m) / s
        I0 = np.where(a > 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))  # P(X in cell)
        I1 = ((m - u[:-1, None]) * I0 + s * (np.exp(-.5 * a * a) - np.exp(-.5 * b * b)) / np.sqrt(2 * np.pi)) / du
        Gp = np.zeros((len(u), pos.sum()))
        Gp[:-1] += I0 - I1  # the cell's left node: weight 1 - t, t = (x - u_j) / du
        Gp[1:] += I1  # its right node: weight t
        G[:, pos] = Gp / np.maximum(Gp.sum(0, keepdims=True), TINY)  # truncated to the grid
    if (~pos).any():
        x = (np.clip(mu[~pos], u[0], u[-1]) - u[0]) / du
        j = np.minimum(np.floor(x).astype(int), len(u) - 2)
        cols = np.flatnonzero(~pos)
        G[j, cols] = 1 - (x - j)
        G[j + 1, cols] = x - j
    return G


def _scaled(lls: np.ndarray) -> np.ndarray:
    """exp of each row less its maximum: the likelihoods, each to its own constant."""
    lls = np.asarray(lls, float)
    return np.exp(lls - lls.max(axis=1, keepdims=True))


def _loglik(L: np.ndarray, u: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """(M,) sum_i log (L @ g)_i for each (mu, sigma) column, up to the rows' constants."""
    G = hat_weights(u, mu, sigma)
    out = np.empty(G.shape[1])
    step = max(1, _CHUNK // len(L))
    for s in range(0, G.shape[1], step):
        out[s:s + step] = np.log(np.maximum(L @ G[:, s:s + step], TINY)).sum(axis=0)
    return out


def _check(lls: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lls, u = np.asarray(lls, float), np.asarray(u, float)
    if lls.ndim != 2 or lls.shape[1] != len(u):
        raise ValueError(f"lls must be (n_tracks, {len(u)}), got {lls.shape}")
    if not len(lls):
        raise ValueError("no likelihood rows")
    if len(u) < 4:
        raise ValueError("the grid needs at least 4 points")
    du = np.diff(u)
    if not np.allclose(du, du[0], rtol=1e-6):
        raise ValueError("u must be evenly spaced")
    return lls, u


def _grid_quantiles(log_p: np.ndarray, x: np.ndarray, qs) -> np.ndarray:
    """Quantiles of a distribution on grid `x` (log masses), interpolating the CDF at cell midpoints."""
    p = np.exp(log_p - logsumexp(log_p))
    return np.interp(qs, np.cumsum(p) - p / 2, x)


def _interval(values: np.ndarray) -> dict[str, float]:
    lo, med, hi = values
    return {"median": float(med), "lo": float(lo), "hi": float(hi)}


def _qs(level: float) -> list[float]:
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    return [(1 - level) / 2, .5, (1 + level) / 2]


# --------------------------------------------------------------------------
# Complete pooling: one D for every track
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SharedD:
    """The posterior of one D shared by every track (complete pooling), prior flat in ln D on the grid.

    With many tracks it is narrower than a grid cell, so it is kept on a finer stretch of ln D
    around its peak: `u` (not the analysis grid) and `log_post`, normalized log masses on `u`.
    """

    u: np.ndarray
    log_post: np.ndarray
    n_tracks: int
    at_grid_edge: bool  # the summed log-likelihood peaks at a grid end: the summary depends on the grid

    def summary(self, level: float = .9) -> dict[str, float]:
        """Median and equal-tailed `level` interval of the shared D, in um^2/s."""
        return _interval(np.exp(_grid_quantiles(self.log_post, self.u, _qs(level))))


def fit_shared_D(lls: np.ndarray, u: np.ndarray) -> SharedD:
    """Complete pooling: the posterior of one D for all tracks, from per-track log-likelihood rows on `u`.

    The rows are summed. The sum is a smooth function of ln D, close to a parabola near its peak,
    so it is interpolated with a cubic spline through the cells within `_SHARED_WINDOW` nats of the
    peak, at `_SHARED_REFINE` points per cell, which resolves it below the grid step.
    """
    lls, u = _check(lls, u)
    total = (lls - lls.max(axis=1, keepdims=True)).sum(axis=0)
    total -= total.max()
    near = np.flatnonzero(total >= -_SHARED_WINDOW)
    lo, hi = max(near[0] - 2, 0), min(near[-1] + 2, len(u) - 1)
    while hi - lo < 3:  # a cubic needs four points
        lo, hi = max(lo - 1, 0), min(hi + 1, len(u) - 1)
    fine = np.linspace(u[lo], u[hi], (hi - lo) * _SHARED_REFINE + 1)
    log_w = CubicSpline(u[lo:hi + 1], total[lo:hi + 1])(fine)
    peak = int(np.argmax(total))
    at_edge = peak in (0, len(u) - 1)
    if at_edge:
        warn("fit_shared_D: the summed log-likelihood peaks at a grid end; widen the D grid", stacklevel=2)
    return SharedD(fine, log_w - logsumexp(log_w), len(lls), at_edge)


def shared_D_tracks(analysis: GridPosteriorAnalysis) -> SharedD:
    """`fit_shared_D` over the tracks of `analyze_tracks(..., keep_likelihoods=True)`, on its grid."""
    return fit_shared_D(_rows(analysis, "shared_D_tracks"), analysis.options.u_D())


# --------------------------------------------------------------------------
# Partial pooling: log-normal D across tracks
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LogNormal(GridDistribution):
    """A log-normal distribution of D across tracks, ln D ~ N(mu, sigma) truncated to the grid `u`.

    `weights` is the distribution at the posterior mode of (mu, sigma); `samples` are its posterior
    draws, so `band`, `mass` and `cdf_distance` read it as they read a deconvolution. The posterior
    itself is `log_post` (normalized log masses) on the grid `mu` x `sigma`, and `draws` are
    (n_samples, 2) draws of (mu, sigma) from it, each the parameters of the matching `samples` row.
    """

    mu: np.ndarray
    sigma: np.ndarray
    log_post: np.ndarray  # (len(mu), len(sigma))
    draws: np.ndarray  # (n_samples, 2): mu, sigma
    sigma_max: float
    problem: str  # why the posterior depends on its prior bounds, or ""

    def marginal(self, name: str) -> np.ndarray:
        """Normalized log masses of `mu` or `sigma`, the other integrated out."""
        if name not in ("mu", "sigma"):
            raise ValueError(f"name must be 'mu' or 'sigma', got {name!r}")
        return logsumexp(self.log_post, axis=1 if name == "mu" else 0)

    def summary(self, level: float = .9) -> dict[str, dict[str, float]]:
        """Median and equal-tailed `level` interval of the population's median D (exp mu, um^2/s), its
        spread in ln D (sigma), and its mean D (exp(mu + sigma^2 / 2), um^2/s, from the draws).

        The median D and sigma are read off the grid; the mean D, a function of both, off `draws`.
        """
        qs = _qs(level)
        mean = np.exp(self.draws[:, 0] + .5 * self.draws[:, 1] ** 2)
        return {
            "D_median_um2_s": _interval(np.exp(_grid_quantiles(self.marginal("mu"), self.mu, qs))),
            "sigma_ln_D": _interval(_grid_quantiles(self.marginal("sigma"), self.sigma, qs)),
            "D_mean_um2_s": _interval(np.quantile(mean, qs)),
        }


@dataclass(frozen=True)
class LogNormalPopulation(LogNormal):
    """`LogNormal` fitted to an analysis's tracks (`lognormal_tracks`)."""

    n_tracks: int  # tracks that contributed a likelihood
    n_excluded: int  # too short (< options.min_frames) or invalid input
    acquisition: Acquisition | None  # None when combined across experiments with different acquisitions


def _zoom(L: np.ndarray, u: np.ndarray, sigma_max: float):
    """The (mu, sigma) grid that holds the posterior: start on the whole prior, then repeatedly
    fit the grid to the cells within `DROP` of the peak, one cell beyond them on each side, until
    they fill at least a third of it. An edge the region touches is pushed out, unless it is the
    prior's own bound."""
    bounds = np.array([[u[0], u[-1]], [0., sigma_max]])
    window = bounds.copy()
    n = (N_MU, N_SIGMA)
    for _ in range(MAX_ZOOM):
        axes = [np.linspace(*window[k], n[k]) for k in range(2)]
        M, S = np.meshgrid(*axes, indexing="ij")
        lp = _loglik(L, u, M.ravel(), S.ravel()).reshape(M.shape)
        keep = lp >= lp.max() - DROP
        new, done = window.copy(), True
        for k, occupied in enumerate((np.flatnonzero(keep.any(1)), np.flatnonzero(keep.any(0)))):
            step = axes[k][1] - axes[k][0]
            first, last = occupied[0], occupied[-1]
            lo = window[k, 0] - (window[k, 1] - window[k, 0]) if first == 0 else axes[k][first] - step
            hi = window[k, 1] + (window[k, 1] - window[k, 0]) if last == n[k] - 1 else axes[k][last] + step
            new[k] = max(lo, bounds[k, 0]), min(hi, bounds[k, 1])
            open_edge = (first == 0 and window[k, 0] > bounds[k, 0]) or (last == n[k] - 1 and window[k, 1] < bounds[k, 1])
            if open_edge or last - first + 1 < n[k] / 3:
                done = False
        if done or np.allclose(new, window, rtol=0, atol=1e-12):
            break
        window = new
    else:
        warn("fit_lognormal: the (mu, sigma) grid did not settle; the posterior may be cut", stacklevel=3)
    return axes[0], axes[1], lp


def fit_lognormal(lls: np.ndarray, u: np.ndarray, sigma_max: float = SIGMA_MAX, n_samples: int = 1000,
                  rng: np.random.Generator | None = None) -> LogNormal:
    """Partial pooling: the posterior of a log-normal distribution of D across tracks, from per-track
    log-likelihood rows on the evenly spaced ln D grid `u`.

    Priors: mu flat over the grid, sigma flat on [0, `sigma_max`]. A posterior that reaches either
    bound depends on it, and says so in `problem` (with a warning). `rng` (default: seeded) draws
    `samples`. A track with a flat likelihood row leaves the posterior unchanged.
    """
    lls, u = _check(lls, u)
    if not sigma_max > 0:
        raise ValueError(f"sigma_max must be > 0, got {sigma_max}")
    L = _scaled(lls)
    mu, sigma, lp = _zoom(L, u, sigma_max)
    log_post = lp - logsumexp(lp)

    near = log_post >= log_post.max() - DROP
    problems = []
    if near[:, -1].any() and sigma[-1] >= sigma_max:
        problems.append(f"sigma reaches its prior bound sigma_max={sigma_max:g}")
    if (near[0].any() and mu[0] <= u[0]) or (near[-1].any() and mu[-1] >= u[-1]):
        problems.append("mu reaches the D grid's end")
    problem = "; ".join(problems)
    if problem:
        warn(f"fit_lognormal: {problem}", stacklevel=2)

    rng = np.random.default_rng(0) if rng is None else rng
    cells = rng.choice(log_post.size, size=n_samples, p=np.exp(log_post).ravel())
    i, j = np.unravel_index(cells, log_post.shape)
    d_mu, d_sigma = mu[1] - mu[0], sigma[1] - sigma[0]
    draws = np.column_stack([  # uniform within each cell, reflected at the prior's bounds
        _reflect(mu[i] + d_mu * rng.uniform(-.5, .5, n_samples), u[0], u[-1]),
        _reflect(sigma[j] + d_sigma * rng.uniform(-.5, .5, n_samples), 0., sigma_max),
    ])
    mode = np.unravel_index(int(np.argmax(log_post)), log_post.shape)
    weights = hat_weights(u, mu[mode[0]], sigma[mode[1]])[:, 0]
    samples = hat_weights(u, draws[:, 0], draws[:, 1]).T
    return LogNormal(u, weights, samples, mu, sigma, log_post, draws, float(sigma_max), problem)


def lognormal_tracks(analysis: GridPosteriorAnalysis, sigma_max: float = SIGMA_MAX, n_samples: int = 1000,
                     rng: np.random.Generator | None = None) -> LogNormalPopulation:
    """`fit_lognormal` over the tracks of `analyze_tracks(..., keep_likelihoods=True)`, on its grid.

    Tracks without an "ok" likelihood (too short, invalid input) are counted in `n_excluded`.
    """
    lls = _rows(analysis, "lognormal_tracks")
    fit = fit_lognormal(lls, analysis.options.u_D(), sigma_max, n_samples, rng)
    return LogNormalPopulation(**{f.name: getattr(fit, f.name) for f in fields(fit)},
                               n_tracks=len(lls), n_excluded=analysis.fits.height - len(lls),
                               acquisition=analysis.acquisition)


def _reflect(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """x folded back into [lo, hi] at either end (a jitter of at most half a cell crosses one end only)."""
    x = lo + np.abs(x - lo)
    return hi - np.abs(hi - x)


def _rows(analysis: GridPosteriorAnalysis, caller: str) -> np.ndarray:
    if analysis.likelihoods is None:
        raise ValueError(f"{caller} needs the per-track likelihoods: analyze_tracks(..., keep_likelihoods=True)")
    lls = analysis.likelihoods.loglik_D
    if not len(lls):
        raise ValueError("no track had enough frames and valid input to contribute")
    return lls
