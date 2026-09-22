"""Brownian displacement likelihood with known per-frame localization SDs.

Per axis, the m = n - 1 displacements of one track are Gaussian with

    Sigma(D) = D A + B,

where A is Brownian motion averaged over the exposure (Berglund 2010, box
shutter closed form: R = exposure_s / (6 dt_s), diagonal 2 dt (1 - 2R),
lag-1 covariance +2 dt R, zero beyond lag 1) and B comes from independent
localization errors: B_ii = s_i^2 + s_(i+1)^2, B_(i,i+1) = -s_(i+1)^2.
D >= 0; D = 0 means localization noise alone.

With B = L Lt and L^-1 A L^-t = Q diag(lam) Qt, the whitened data
y = Qt L^-1 delta have independent components of variance 1 + D lam_k, so
the likelihood over any D grid costs O(m) per grid point. This is the
shared whitening engine `gridpost.posterior`'s grid posterior over D (and
`gridpost.posterior_alpha`'s posterior over alpha, via `fgn_motion_covariance`
in place of `motion_covariance`) is built on.
"""
import numpy as np
import polars as pl
from scipy.linalg import cholesky, eigh, solve_triangular

from ..data import Acquisition
from ..io import validated_track_frame


def motion_covariance(n_disp: int, dt_s: float, exposure_s: float = 0.) -> np.ndarray:
    """(n_disp, n_disp) covariance of consecutive displacements per unit D.

    Box shutter, exposure_s <= dt_s (Berglund 2010): R = exposure_s / (6 dt_s),
    diagonal 2 dt (1 - 2R), lag-1 covariance 2 dt R, zero beyond lag 1 -- a
    box shutter of at most one frame's duration only correlates adjacent
    displacements.
    """
    R = exposure_s / (6 * dt_s)
    A = np.diag(np.full(n_disp, 2 * dt_s * (1 - 2 * R)))
    if n_disp > 1:
        i = np.arange(n_disp - 1)
        A[i, i + 1] = A[i + 1, i] = 2 * dt_s * R
    return A


def fgn_motion_covariance(n_disp: int, dt_s: float, alpha: float) -> np.ndarray:
    """(n_disp, n_disp) fGn covariance of consecutive displacements per unit K, no exposure blur.

    gamma(k) = dt^alpha (|k+1|^alpha - 2|k|^alpha + |k-1|^alpha); alpha=1 reduces
    exactly to `motion_covariance(..., exposure_s=0.)` -- ordinary Brownian
    motion, independent increments. No closed-form exposure average exists at
    a general alpha (`motion_covariance`'s Berglund R is specific to alpha=1's
    linear-motion double integral), so callers must use exposure_s=0.
    """
    k = np.arange(n_disp, dtype=float)
    gamma = dt_s**alpha * (np.abs(k + 1) ** alpha - 2 * np.abs(k) ** alpha + np.abs(k - 1) ** alpha)
    i = np.arange(n_disp)
    return gamma[np.abs(i[:, None] - i[None, :])]


def localization_covariance(sd_um: np.ndarray) -> np.ndarray:
    """(2, m, m) displacement covariance from independent per-frame position SDs."""
    var = np.asarray(sd_um, dtype=float).T ** 2  # (2, n)
    m = var.shape[1] - 1
    B = np.zeros((2, m, m))
    i = np.arange(m)
    B[:, i, i] = var[:, :-1] + var[:, 1:]
    B[:, i[:-1], i[1:]] = B[:, i[1:], i[:-1]] = -var[:, 1:-1]
    return B


def _prepared(track: pl.DataFrame, acquisition: Acquisition) -> pl.DataFrame:
    track = validated_track_frame(track, acquisition, require_localization=True)
    if track.height < 3:
        raise ValueError("At least three frames are required")
    sd = track.select("sigma_x_um", "sigma_y_um").to_numpy()
    if np.any(sd <= 0):
        raise ValueError("The grid posterior requires positive localization SDs")
    return track


def _whiten_axis(delta_axis: np.ndarray, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """One axis's (lam, y, logdet_B) for displacements ~ N(0, x A + B), any fixed A.

    Shared by `gridpost.posterior` (A = `motion_covariance`, alpha=1 fixed) and
    `gridpost.posterior_alpha` (A = `fgn_motion_covariance(..., alpha)`, one
    call per alpha grid point) -- the only difference between a D-posterior
    and an alpha-posterior whitening step is which A goes in.
    """
    L = cholesky(B, lower=True)
    LA = solve_triangular(L, A, lower=True)
    M = solve_triangular(L, LA.T, lower=True)  # L^-1 A L^-t
    values, Q = eigh((M + M.T) / 2)
    y = Q.T @ solve_triangular(L, delta_axis, lower=True)
    logdet_B = 2 * np.sum(np.log(np.diag(L)))
    return values, y, logdet_B


def _whiten(track: pl.DataFrame, acquisition: Acquisition) -> dict:
    """Whitened data for one validated track."""
    positions = track.select("x_um", "y_um").to_numpy()
    sd = track.select("sigma_x_um", "sigma_y_um").to_numpy()
    delta = np.diff(positions, axis=0).T  # (2, m)
    m = delta.shape[1]
    A = motion_covariance(m, float(acquisition.dt_s), float(acquisition.exposure_s))
    lam, y, logdet_B = [], [], 0.
    for axis, B in enumerate(localization_covariance(sd)):
        values, yy, ld = _whiten_axis(delta[axis], A, B)
        lam.append(values)
        y.append(yy)
        logdet_B += ld
    const = -.5 * logdet_B - m * np.log(2 * np.pi)
    return {"lam": np.array(lam), "y": np.array(y), "const": const}


def _loglik(D, lam, y, const):
    """D (r,), y (r, 2, m) -> (r,)."""
    d = 1 + D[:, None, None] * lam
    return const - .5 * np.sum(np.log(d) + y**2 / d, axis=(1, 2))


def brownian_log_likelihood(track: pl.DataFrame, acquisition: Acquisition, D_um2_s: float) -> float:
    """Exact Gaussian log-likelihood of both axes' displacements at D >= 0."""
    w = _whiten(_prepared(track, acquisition), acquisition)
    return float(_loglik(np.array([float(D_um2_s)]), w["lam"], w["y"][None], w["const"])[0])
