"""Per-track joint posterior P(alpha, D | displacements) for blurred fBm with known localization SDs.

Per axis, the m = n - 1 displacements are Gaussian with

    Sigma(alpha, K) = K C_alpha + B,

C_alpha the box-shutter-blurred fGn covariance per unit K (`covariance.fgn_motion_covariance`; the
true motion's MSD is 2 K tau^alpha per axis) and B the localization term of `gridpost.likelihood`.
For each alpha, with B = L L^t and L^-1 C_alpha L^-t = Q diag(lam) Q^t, the whitened data
y = Q^t L^-1 delta are independent with variances 1 + K lam_k -- `gridpost`'s whitening, once per
alpha -- so the amplitude axis costs O(m) per grid value and the grid costs one eigendecomposition
per (axis, alpha), not one per cell.

The amplitude axis is ln SNR, SNR = K ||lam_alpha|| with lam_alpha both axes' eigenvalues
(`FbmPostOptions` says why); each cell's K = SNR / ||lam_alpha|| is kept, and what is reported is
D_ref = K tau_ref^(alpha - 1).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import polars as pl
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp

from ..data import Acquisition
from ..gridpost.likelihood import _prepared, localization_covariance
from ..gridpost.posterior import grid_edge
from .covariance import fgn_autocovariance
from .data import FbmPostOptions

# Upper bound on the floats in one alpha chunk's (chunk, n_snr, m) and (chunk, m, m) arrays.
_CHUNK_FLOATS = 4_000_000


class TrackGrid(NamedTuple):
    loglik: np.ndarray  # (n_alpha, n_snr) log-likelihood on options.alphas() x options.u_snr()
    ln_K: np.ndarray    # (n_alpha, n_snr) ln K (um^2 / s^alpha) of every cell


def log_likelihood_grid(track: pl.DataFrame, acquisition: Acquisition, options: FbmPostOptions) -> TrackGrid:
    """One track's log-likelihood on `options.alphas()` x `options.u_snr()`, and each cell's ln K."""
    track = _prepared(track, acquisition)
    delta = np.diff(track.select("x_um", "y_um").to_numpy(), axis=0).T  # (2, m)
    sd = track.select("sigma_x_um", "sigma_y_um").to_numpy()
    m = delta.shape[1]
    alphas, u = options.alphas(), options.u_snr()
    gamma = fgn_autocovariance(m, float(acquisition.dt_s), alphas, float(acquisition.exposure_s))  # per unit K
    lag = np.abs(np.arange(m)[:, None] - np.arange(m)[None, :])
    step = max(1, _CHUNK_FLOATS // (m * max(m, len(u))))
    lam, y = np.zeros((len(alphas), 2 * m)), np.zeros((len(alphas), 2 * m))
    const = 0.
    for axis, B in enumerate(localization_covariance(sd)):
        L = cholesky(B, lower=True)
        Li = solve_triangular(L, np.eye(m), lower=True)
        z = Li @ delta[axis]
        const += -np.log(np.diag(L)).sum() - .5 * m * np.log(2 * np.pi)
        cols = slice(axis * m, (axis + 1) * m)
        for s in range(0, len(alphas), step):
            M = Li @ gamma[s:s + step][:, lag] @ Li.T  # (chunk, m, m)
            values, Q = np.linalg.eigh((M + np.swapaxes(M, 1, 2)) / 2)
            lam[s:s + step, cols] = np.clip(values, 0, None)  # C_alpha is PSD; round-off can dip below 0
            y[s:s + step, cols] = np.einsum("aij,i->aj", Q, z)
    ln_K = u[None, :] - .5 * np.log((lam ** 2).sum(axis=1))[:, None]
    out = np.full((len(alphas), len(u)), const)
    for s in range(0, len(alphas), step // 2 or 1):
        e = s + (step // 2 or 1)
        v = 1 + np.exp(ln_K[s:e, :, None]) * lam[s:e, None, :]  # (chunk, n_snr, 2m)
        out[s:e] -= .5 * (np.log(v).sum(-1) + (y[s:e, None, :] ** 2 / v).sum(-1))
    return TrackGrid(out, ln_K)


def log_posterior(ll: np.ndarray) -> np.ndarray:
    """The flat-prior log posterior on the grid, normalized over all cells."""
    return ll - logsumexp(ll)


def alpha_log_marginal(log_post: np.ndarray) -> np.ndarray:
    """(n_alpha,) normalized log marginal of alpha (the amplitude summed out under its flat ln SNR prior)."""
    return logsumexp(log_post, axis=1)


def _cdf_quantile(p: np.ndarray, x: np.ndarray, q) -> np.ndarray:
    """Quantiles of the distribution `p` over sorted points `x`, interpolating the CDF at cell midpoints
    (as `gridpost.posterior` does)."""
    return np.interp(q, np.cumsum(p) - p / 2, x)


def log10_D_ref(ln_K: np.ndarray, options: FbmPostOptions) -> np.ndarray:
    """log10 D_ref = log10(K tau_ref^(alpha - 1)) of every grid cell."""
    return (ln_K + (options.alphas()[:, None] - 1) * np.log(options.tau_ref_s)) / np.log(10)


def localization_floor(track: pl.DataFrame, options: FbmPostOptions) -> float:
    """The D_ref (um^2/s) at which the true motion's MSD at tau_ref equals the localization noise's
    MSD offset, 2 <s^2> per axis: <s^2> / tau_ref, with <s^2> the mean localization variance over the
    track's frames and both axes. A reference scale to draw next to D_ref, not a threshold."""
    sd = track.select("sigma_x_um", "sigma_y_um").to_numpy()
    return float(np.mean(sd ** 2) / options.tau_ref_s)


def summary(log_post: np.ndarray, ln_K: np.ndarray, options: FbmPostOptions) -> dict:
    """Marginal summaries of alpha and D_ref, and the (alpha, log10 D_ref) moments that draw the ellipse."""
    p = np.exp(log_post)
    alphas = options.alphas()
    pa = p.sum(axis=1)
    x = log10_D_ref(ln_K, options)
    tail = (1 - options.level) / 2
    a_mean = float(pa @ alphas)
    a_sd = float(np.sqrt(max(pa @ (alphas - a_mean) ** 2, 0.)))
    d_mean = float((p * x).sum())
    d_sd = float(np.sqrt(max((p * (x - d_mean) ** 2).sum(), 0.)))
    cov = float((p * (alphas[:, None] - a_mean) * (x - d_mean)).sum())
    order = np.argsort(x, axis=None)
    d_lo, d_hi = _cdf_quantile(p.ravel()[order], x.ravel()[order], [tail, 1 - tail])
    a_lo, a_hi = _cdf_quantile(pa, alphas, [tail, 1 - tail])
    keep = pa > 0
    bits = float(np.sum(pa[keep] * np.log2(pa[keep] * len(pa))))
    return {
        "alpha_post_mean": a_mean, "alpha_post_lo": float(a_lo), "alpha_post_hi": float(a_hi),
        "alpha_post_sd": a_sd, "alpha_post_info_bits": bits,
        "alpha_post_p_below_1": float(pa[alphas < 1].sum() + .5 * pa[np.isclose(alphas, 1)].sum()),
        "D_ref_post_mean_um2_s": float((p * 10. ** x).sum()),
        "D_ref_post_lo_um2_s": float(10. ** d_lo), "D_ref_post_hi_um2_s": float(10. ** d_hi),
        "log10_D_ref_post_mean": d_mean, "log10_D_ref_post_sd": d_sd,
        "alpha_log10_D_ref_corr": cov / (a_sd * d_sd) if a_sd > 0 and d_sd > 0 else float("nan"),
    }


def snr_edge(log_post: np.ndarray) -> str | None:
    """`gridpost.posterior.grid_edge` of the ln SNR margin: which edge of the amplitude prior cuts the posterior."""
    return grid_edge(np.exp(logsumexp(log_post, axis=0)))
