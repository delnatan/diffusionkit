"""Per-track posterior P(D | displacements) for Brownian motion. Standalone prototype.

Nothing here imports diffusionkit. Only numpy and scipy; matplotlib for the demo.

Model, per axis. A track of n frames gives m = n - 1 displacements
d = (x_2 - x_1, ..., x_n - x_(n-1)), and

    d ~ N(0, D A + B)

- A (per unit D): Brownian motion averaged over the on-time (Berglund 2010),
  a box shutter. on_time (tau) is how long light is collected within each
  frame: the camera exposure for continuous illumination, the pulse width for
  a strobed laser. Berglund's blur coefficient is R = tau / (6 dt). on_time = 0
  gives 2 dt I; otherwise the diagonal is 2 dt (1 - 2R) and the lag-1
  covariance is +2 dt R, so blur moves variance off the diagonal.
- B: the known per-frame localization variances, B_ii = s_i^2 + s_(i+1)^2 and
  B_(i,i+1) = -s_(i+1)^2. They come from the Gaussian fit's standard errors.

The axes of a 2D track are independent given D, so their log-likelihoods add.

One generalized eigendecomposition A v = lam B v (V' B V = I) whitens the
displacements, y = V' d, after which the log-likelihood over any D grid costs
O(m) per grid point:

    ln L(D) = const - 1/2 sum_k [ ln(1 + D lam_k) + y_k^2 / (1 + D lam_k) ]

The posterior lives on a fixed grid in u = ln D, so a prior that is flat in u
is log-uniform in D (scale-invariant), and the weights are just
exp(ln L + ln prior), normalized.

Sequential updating. Localization noise correlates adjacent displacements, so
multiplying one-displacement likelihoods would double-count and is wrong. The
exact posterior after k displacements conditions on the joint likelihood of
the first k, which is the likelihood of the track truncated to k + 1 frames.
`sequence` does exactly that; it is not an approximation.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d

# ln D grid, D in um^2/s: 1e-4 to 10 in 2.3% steps, wider than any prior below.
U = np.linspace(np.log(1e-4), np.log(10.0), 501)


# --------------------------------------------------------------------------
# Likelihood
# --------------------------------------------------------------------------


def motion_cov(m: int, dt: float, on_time: float = 0.0) -> np.ndarray:
    """(m, m) covariance of m consecutive 1D displacements per unit D.

    Box shutter, on_time <= dt (Berglund 2010): R = on_time / (6 dt), diagonal
    2 dt (1 - 2R), lag-1 covariance 2 dt R, zero beyond lag 1 -- a box shutter
    of at most one frame's duration only correlates adjacent displacements.
    """
    if not 0 <= on_time <= dt:
        raise ValueError("on_time must be between 0 and dt")
    R = on_time / (6 * dt)
    cov = np.diag(np.full(m, 2 * dt * (1 - 2 * R)))
    if m > 1:
        i = np.arange(m - 1)
        cov[i, i + 1] = cov[i + 1, i] = 2 * dt * R
    return cov


def localization_cov(sd) -> np.ndarray:
    """(m, m) displacement covariance from n = m + 1 per-frame position SDs."""
    v = np.asarray(sd, float) ** 2
    B = np.diag(v[:-1] + v[1:])
    i = np.arange(len(v) - 2)
    B[i, i + 1] = B[i + 1, i] = -v[1:-1]
    return B


class Whitened(NamedTuple):
    lam: np.ndarray  # (m,) variance of component k is 1 + D lam_k
    y: np.ndarray  # (m,)
    const: float  # log-likelihood terms that do not depend on D


def whiten(disp, sd, dt: float, on_time: float = 0.0) -> Whitened:
    """One axis: displacements (m,), per-frame position SDs (m + 1,)."""
    disp = np.asarray(disp, float)
    if len(sd) != len(disp) + 1:
        raise ValueError("need one localization SD per frame")
    if np.any(np.asarray(sd) <= 0):
        raise ValueError("localization SDs must be positive")
    B = localization_cov(sd)
    lam, V = eigh(motion_cov(len(disp), dt, on_time), B)
    return Whitened(
        np.maximum(lam, 0.0),
        V.T @ disp,
        -0.5
        * (
            len(disp) * np.log(2 * np.pi)
            + 2 * np.log(np.diag(np.linalg.cholesky(B))).sum()
        ),
    )


def loglik(axes: list[Whitened], u: np.ndarray = U) -> np.ndarray:
    """(len(u),) log-likelihood of the whitened axes at D = exp(u)."""
    total = 0.0
    for lam, y, const in axes:
        d = 1 + np.exp(u)[:, None] * lam
        total = total + const - 0.5 * np.sum(np.log(d) + y**2 / d, axis=1)
    return total


def track_loglik(
    positions, sd, dt: float, on_time: float = 0.0, u: np.ndarray = U
) -> np.ndarray:
    """positions (n,) or (n, axes); sd (n,) or (n, axes) in the same units as positions."""
    x = np.asarray(positions, float)
    x = x[:, None] if x.ndim == 1 else x
    s = np.broadcast_to(
        np.reshape(np.asarray(sd, float), (len(x), -1)), x.shape
    )
    return loglik(
        [
            whiten(np.diff(x[:, a]), s[:, a], dt, on_time)
            for a in range(x.shape[1])
        ],
        u,
    )


# --------------------------------------------------------------------------
# Priors: log-density on the grid, up to a constant
# --------------------------------------------------------------------------


def log_uniform(D_lo: float, D_hi: float, u: np.ndarray = U) -> np.ndarray:
    """Flat in ln D between the limits, impossible outside them."""
    return np.where((u >= np.log(D_lo)) & (u <= np.log(D_hi)), 0.0, -np.inf)


def log_normal(D_lo: float, D_hi: float, u: np.ndarray = U) -> np.ndarray:
    """Log-normal whose central 95% spans [D_lo, D_hi]: the same limits with soft edges."""
    mu, sd = (
        0.5 * (np.log(D_lo) + np.log(D_hi)),
        (np.log(D_hi) - np.log(D_lo)) / (2 * 1.96),
    )
    return -0.5 * ((u - mu) / sd) ** 2


# --------------------------------------------------------------------------
# Simulation (independent of the likelihood: fine-step path, averaged over the on-time)
# --------------------------------------------------------------------------


def simulate(
    D: float,
    sd,
    dt: float,
    on_time: float,
    rng: np.random.Generator,
    axes: int = 2,
    sub: int = 100,
) -> np.ndarray:
    """(n, axes) measured positions: Brownian path in dt/sub steps, mean over each
    on_time window (trapezoid), plus Gaussian localization error with SD sd[i]."""
    sd = np.asarray(sd, float)
    n, h = len(sd), dt / sub
    n_on = round(on_time / h)
    path = np.vstack(
        [
            np.zeros(axes),
            np.cumsum(
                rng.normal(
                    0, np.sqrt(2 * D * h), ((n - 1) * sub + n_on, axes)
                ),
                axis=0,
            ),
        ]
    )
    w = np.ones(n_on + 1)
    w[[0, -1]] = 0.5
    blurred = np.array(
        [
            w @ path[i * sub : i * sub + n_on + 1] / max(n_on, 1)
            if n_on
            else path[i * sub]
            for i in range(n)
        ]
    )
    return blurred + sd[:, None] * rng.standard_normal((n, axes))


# --------------------------------------------------------------------------
# Posterior and summaries
# --------------------------------------------------------------------------


def posterior(ll: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """Grid weights (sum to 1) proportional to likelihood x prior."""
    lp = ll + log_prior
    p = np.exp(lp - lp.max())
    return p / p.sum()


def sequence(
    positions,
    sd,
    dt: float,
    log_prior: np.ndarray,
    on_time: float = 0.0,
    u: np.ndarray = U,
) -> np.ndarray:
    """(m, len(u)); row k - 1 is the exact posterior after the first k displacements."""
    x = np.asarray(positions, float)
    return np.array(
        [
            posterior(
                track_loglik(
                    x[: k + 1], np.asarray(sd)[: k + 1], dt, on_time, u
                ),
                log_prior,
            )
            for k in range(1, len(x))
        ]
    )


# --------------------------------------------------------------------------
# Population level (comparator for an ensemble fit; not per-track inference)
# --------------------------------------------------------------------------


def pooled(lls: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """Posterior of ONE D shared by every track. lls is (n_tracks, len(u)) per-track log-likelihoods.

    They add, and the prior enters once. This is exact if all tracks share D, and its
    interval shrinks like 1/sqrt(n) whether or not they do. With heterogeneous D it
    estimates a mean-like D that no track has, and its interval is too narrow; resample
    tracks (bootstrap) or compare with `deconvolve` to see that. For large n the posterior
    is narrower than U's 2.3% step: recompute lls on a fine local grid around the peak.
    """
    return posterior(np.sum(lls, axis=0), log_prior)


def deconvolve(
    lls: np.ndarray,
    log_prior: np.ndarray,
    iters: int = 500,
    smooth: float = 0.5,
) -> np.ndarray:
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
    L = np.exp(
        lls - lls.max(axis=1, keepdims=True)
    )  # row scaling does not change the E-step
    support = np.isfinite(log_prior)
    g = posterior(np.zeros(L.shape[1]), log_prior)
    for _ in range(iters):
        r = L * g
        g = (r / r.sum(axis=1, keepdims=True)).mean(axis=0)
        if smooth:
            g = gaussian_filter1d(g, smooth, mode="constant") * support
            g /= g.sum()
    return g


def quantile(p: np.ndarray, q: float, u: np.ndarray = U) -> float:
    """Posterior q-quantile of D, interpolating the CDF at cell midpoints."""
    return float(np.exp(np.interp(q, np.cumsum(p) - p / 2, u)))


def summary(
    p: np.ndarray, u: np.ndarray = U, level: float = 0.9
) -> dict[str, float]:
    """Median and equal-tailed credible interval of D, in um^2/s."""
    return {
        "median": quantile(p, 0.5, u),
        "lo": quantile(p, (1 - level) / 2, u),
        "hi": quantile(p, (1 + level) / 2, u),
    }
