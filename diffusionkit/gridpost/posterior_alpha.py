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
A(alpha) the fBm displacement covariance averaged over the camera exposure
(`likelihood.fgn_motion_covariance`, a closed form that reduces to the D
posterior's Berglund box-shutter average at alpha=1) and B the known
per-frame localization-noise covariance (`likelihood.localization_covariance`,
same as `gridpost.posterior`). For any *fixed* alpha this is linear in K
exactly as `gridpost.posterior`'s model is linear in D, so the same whitening
trick applies -- but A(alpha) itself changes shape with alpha, so (unlike D)
each alpha grid point needs its own eigendecomposition. Everything that does
not depend on alpha (validation, B's Cholesky factor, the whitened
displacements) is done once per track, and the alpha grid's
eigendecompositions run as one batched LAPACK call (`_alpha_whitening`).

That is O(m^3) per alpha, which long tracks feel. For them the debiased
Whittle likelihood (`_whittle_joint_loglik`) replaces each eigendecomposition
with an FFT, keeping the exact blurred covariance, the per-frame localization
SDs and the K marginalization; `GridPostOptions.alpha_method` picks between
the two ("auto": exact for short tracks, Whittle from
`alpha_whittle_min_frames` on).

The alpha posterior is the 1D marginal of the 2D (alpha, ln K) log-likelihood
surface: integrating a nuisance parameter out is exactly `logsumexp` over its
axis, using `log_K_prior` as K's own (hand-set, not empirical-Bayes) prior --
reuse `gridpost.posterior.flat`/`log_uniform`/`log_normal` for it, since those
are generic log-scale grid priors, not specific to being called D.

Started as `prototypes/posterior_alpha.py`, and still matches it numerically
at exposure_s=0 (the prototype has no blur model).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from ..data import Acquisition
from .likelihood import _prepared, fgn_autocovariance, fgn_motion_covariance, localization_covariance
from .data import GridPostOptions
from .posterior import _grid_quantile
from .posterior import flat as flat_K  # noqa: F401  (re-exported: a generic log-scale-grid prior)

# The grids are `GridPostOptions.alphas()` and `.u_K()`: every function below
# takes them explicitly, like `gridpost.posterior` does its D grid.

# Upper bound on the float64 elements of one batch of stacked (alpha, axis)
# matrices in `_joint_loglik`: ~32 MB, so a long track's eigendecompositions
# are chunked over alpha rather than allocated all at once.
_BATCH_ELEMENTS = 1 << 22


def _alpha_whitening(track: pl.DataFrame, acquisition: Acquisition, alphas: np.ndarray):
    """Yield (alpha slice, lam, y, const) for a validated track, in alpha chunks.

    lam, y are (n_chunk, 2, m): per alpha and axis, the generalized eigenvalues
    of (A(alpha), B) and the displacements whitened against them, so that
    d ~ N(0, K A + B) has independent components of variance 1 + K lam.
    const (log det B and the 2 pi term) does not depend on alpha.
    """
    positions = track.select("x_um", "y_um").to_numpy()
    sd = track.select("sigma_x_um", "sigma_y_um").to_numpy()
    delta = np.diff(positions, axis=0).T  # (2, m)
    m = delta.shape[1]
    L = np.linalg.cholesky(localization_covariance(sd))  # (2, m, m)
    L_inv = np.stack([solve_triangular(L[a], np.eye(m), lower=True) for a in range(2)])
    z = np.einsum("aij,aj->ai", L_inv, delta)  # (2, m)
    const = -np.sum(np.log(np.diagonal(L, axis1=1, axis2=2))) - m * np.log(2 * np.pi)
    dt, exposure = float(acquisition.dt_s), float(acquisition.exposure_s)
    chunk = max(1, _BATCH_ELEMENTS // (2 * m * m))
    for start in range(0, len(alphas), chunk):
        part = slice(start, start + chunk)
        A = fgn_motion_covariance(m, dt, alphas[part], exposure)  # (n_chunk, m, m)
        M = L_inv[None] @ A[:, None] @ L_inv.transpose(0, 2, 1)[None]  # (n_chunk, 2, m, m)
        lam, Q = np.linalg.eigh((M + M.transpose(0, 1, 3, 2)) / 2)
        y = np.einsum("caji,aj->cai", Q, z)
        yield part, lam, y, const


def _joint_loglik(track: pl.DataFrame, acquisition: Acquisition, alphas: np.ndarray,
                  u: np.ndarray) -> np.ndarray:
    """`joint_loglik` for a track `_prepared` has already validated."""
    alphas = np.asarray(alphas, dtype=float)
    K = np.exp(u)[None, :, None, None]
    out = np.empty((len(alphas), len(u)))
    for part, lam, y, const in _alpha_whitening(track, acquisition, alphas):
        d = 1 + K * lam[:, None]  # (n_chunk, n_K, 2, m)
        out[part] = const - .5 * np.sum(np.log(d) + y[:, None] ** 2 / d, axis=(2, 3))
    return out


def _whittle_joint_loglik(track: pl.DataFrame, acquisition: Acquisition, alphas: np.ndarray,
                          u: np.ndarray) -> np.ndarray:
    """(len(alphas), len(u)) debiased Whittle log-likelihood for a validated track.

    Each axis's m displacements d have periodogram I(w_j) = |FFT(d)_j|^2 / m at
    the Fourier frequencies w_j = 2 pi j / m. The Whittle likelihood treats
    those ordinates as independent, exponential with mean S(w_j), i.e. it
    approximates the Toeplitz covariance's eigenvectors by Fourier vectors:

        -1/2 sum_{axis, j} [ln S(w_j) + I(w_j) / S(w_j)] - m ln 2 pi.

    "Debiased" (Sykulski, Olhede, Guillaumin, Lilly & Early 2019, Biometrika
    106:251) means S is the *expected* periodogram of the exact finite-m
    covariance K A(alpha) + B rather than the process's spectral density, so
    the approximation is only in that independence, not in leakage or
    aliasing. S is linear in K: K S_A(w) with S_A the tapered FFT of the
    exact blurred autocovariance (`likelihood.fgn_autocovariance`), plus S_B
    from the per-frame localization SDs, which is exact even though they
    vary. One O(m log m) FFT per alpha instead of an O(m^3) eigendecomposition.
    """
    positions = track.select("x_um", "y_um").to_numpy()
    sd = track.select("sigma_x_um", "sigma_y_um").to_numpy()
    delta = np.diff(positions, axis=0).T  # (2, m)
    m = delta.shape[1]
    periodogram = np.abs(np.fft.fft(delta, axis=1)) ** 2 / m  # (2, m)
    cos_w = np.cos(2 * np.pi * np.arange(m) / m)
    var = sd.T ** 2  # (2, n)
    # E|FFT(eps_diff)|^2 / m for the tridiagonal B: its diagonal sum, plus
    # 2 cos w times its (negative) first off-diagonal sum.
    S_B = (np.sum(var[:, :-1] + var[:, 1:], axis=1)[:, None]
           - 2 * cos_w[None] * np.sum(var[:, 1:-1], axis=1)[:, None]) / m  # (2, m)
    taper = 1 - np.arange(m) / m
    K = np.exp(u)[None, :, None, None]
    alphas = np.asarray(alphas, dtype=float)
    out = np.empty((len(alphas), len(u)))
    chunk = max(1, _BATCH_ELEMENTS // (len(u) * 2 * m))
    for start in range(0, len(alphas), chunk):
        part = slice(start, start + chunk)
        c = taper * fgn_autocovariance(m, float(acquisition.dt_s), alphas[part], float(acquisition.exposure_s))
        S_A = 2 * np.fft.fft(c, axis=1).real - c[:, :1]  # sum_{|k|<m} (1 - |k|/m) gamma(k) e^{-iwk}
        S = K * S_A[:, None, None, :] + S_B[None, None]  # (n_chunk, n_K, 2, m)
        out[part] = -.5 * np.sum(np.log(S) + periodogram / S, axis=(2, 3)) - m * np.log(2 * np.pi)
    return out


_JOINT_LOGLIK = {"exact": _joint_loglik, "whittle": _whittle_joint_loglik}


def track_loglik_given_alpha(track: pl.DataFrame, acquisition: Acquisition, alpha: float,
                             u: np.ndarray, method: str = "exact") -> np.ndarray:
    """(len(u),) log-likelihood of one track's displacements at K = exp(u), fixed alpha."""
    return joint_loglik(track, acquisition, np.array([alpha]), u, method)[0]


def joint_loglik(track: pl.DataFrame, acquisition: Acquisition, alphas: np.ndarray,
                 u: np.ndarray, method: str = "exact") -> np.ndarray:
    """(len(alphas), len(u)) log-likelihood surface over (alpha, ln K), exposure blur modelled.

    `method` "exact" is the Gaussian likelihood; "whittle" its debiased
    Whittle approximation (`_whittle_joint_loglik`), for long tracks.
    """
    if method not in _JOINT_LOGLIK:
        raise ValueError(f"method must be 'exact' or 'whittle', got {method!r}")
    return _JOINT_LOGLIK[method](_prepared(track, acquisition), acquisition, alphas, u)


# --------------------------------------------------------------------------
# Posterior over alpha: marginalize the nuisance ln K
# --------------------------------------------------------------------------


def flat_alpha(alphas: np.ndarray) -> np.ndarray:
    """Flat over the alpha grid -- the least-informative default."""
    return np.zeros_like(alphas)


def log_alpha_posterior(joint_ll: np.ndarray, log_K_prior: np.ndarray,
                        log_alpha_prior: np.ndarray | None = None) -> np.ndarray:
    """(len(alphas),) normalized log posterior over alpha: add priors, integrate ln K out.

    `joint_ll` is (len(alphas), len(u)) from `joint_loglik`. Integrating a
    nuisance parameter out is exactly `logsumexp` over its axis (a Riemann
    sum in ln K; the grid step is an additive constant that normalization
    removes).
    """
    log_mass = logsumexp(joint_ll + log_K_prior[None, :], axis=1)
    if log_alpha_prior is not None:
        log_mass = log_mass + log_alpha_prior
    return log_mass - logsumexp(log_mass)


def alpha_posterior(joint_ll: np.ndarray, log_K_prior: np.ndarray,
                    log_alpha_prior: np.ndarray | None = None) -> np.ndarray:
    """(len(alphas),) posterior weights over alpha (sum to 1); see `log_alpha_posterior`."""
    return np.exp(log_alpha_posterior(joint_ll, log_K_prior, log_alpha_prior))


def quantile(p: np.ndarray, q: float, alphas: np.ndarray) -> float:
    """Posterior q-quantile of alpha, interpolating the CDF at cell midpoints."""
    return _grid_quantile(p, alphas, q)


def summary(p: np.ndarray, alphas: np.ndarray, level: float = .9) -> dict[str, float]:
    """Median and equal-tailed credible interval of alpha."""
    return {
        "median": quantile(p, .5, alphas),
        "lo": quantile(p, (1 - level) / 2, alphas),
        "hi": quantile(p, (1 + level) / 2, alphas),
    }


def track_alpha_posterior(track: pl.DataFrame, acquisition: Acquisition, log_K_prior: np.ndarray | None = None,
                          options: GridPostOptions = GridPostOptions()) -> dict[str, float]:
    """Posterior median and `options.level` credible interval of alpha for one track.

    Evaluated on `options.alphas()`, with K integrated out over `options.u_K()`,
    by the likelihood `options.alpha_likelihood` picks for the track's length.
    `log_K_prior` (on that K grid) defaults to `flat_K` -- the
    least-informative choice for the nuisance parameter, no empirical-Bayes
    fitting across tracks. `acquisition.exposure_s` is modelled as box-shutter blur.
    """
    alphas, u = options.alphas(), options.u_K()
    prior = flat_K(u) if log_K_prior is None else log_K_prior
    track = _prepared(track, acquisition)
    ll = _JOINT_LOGLIK[options.alpha_likelihood(track.height)](track, acquisition, alphas, u)
    return summary(alpha_posterior(ll, prior), alphas, options.level)
