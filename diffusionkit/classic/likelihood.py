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
shared whitening engine `classic.posterior`'s grid posterior over D is
built on.
"""
import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular

from ..data import Acquisition, Track
from ..validation import validate_acquisition, validated_track


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


def localization_covariance(sd_um: np.ndarray) -> np.ndarray:
    """(2, m, m) displacement covariance from independent per-frame position SDs."""
    var = np.asarray(sd_um, dtype=float).T ** 2  # (2, n)
    m = var.shape[1] - 1
    B = np.zeros((2, m, m))
    i = np.arange(m)
    B[:, i, i] = var[:, :-1] + var[:, 1:]
    B[:, i[:-1], i[1:]] = B[:, i[1:], i[:-1]] = -var[:, 1:-1]
    return B


def _prepared(track: Track, acquisition: Acquisition) -> Track:
    validate_acquisition(acquisition, allow_exposure=True)
    track = validated_track(track, require_localization=True)
    if len(track.frames) < 3:
        raise ValueError("At least three frames are required")
    if np.any(track.localization_sd_um <= 0):
        raise ValueError("The grid posterior requires positive localization SDs")
    return track


def _whiten(track: Track, acquisition: Acquisition) -> dict:
    """Whitened data for one validated track."""
    delta = np.diff(track.positions_um, axis=0).T  # (2, m)
    m = delta.shape[1]
    A = motion_covariance(m, float(acquisition.dt_s), float(acquisition.exposure_s))
    lam, y, logdet_B = [], [], 0.
    for axis, B in enumerate(localization_covariance(track.localization_sd_um)):
        L = cholesky(B, lower=True)
        LA = solve_triangular(L, A, lower=True)
        M = solve_triangular(L, LA.T, lower=True)  # L^-1 A L^-t
        values, Q = eigh((M + M.T) / 2)
        lam.append(values)
        y.append(Q.T @ solve_triangular(L, delta[axis], lower=True))
        logdet_B += 2 * np.sum(np.log(np.diag(L)))
    const = -.5 * logdet_B - m * np.log(2 * np.pi)
    return {"lam": np.array(lam), "y": np.array(y), "const": const}


def _loglik(D, lam, y, const):
    """D (r,), y (r, 2, m) -> (r,)."""
    d = 1 + D[:, None, None] * lam
    return const - .5 * np.sum(np.log(d) + y**2 / d, axis=(1, 2))


def brownian_log_likelihood(track: Track, acquisition: Acquisition, D_um2_s: float) -> float:
    """Exact Gaussian log-likelihood of both axes' displacements at D >= 0."""
    w = _whiten(_prepared(track, acquisition), acquisition)
    return float(_loglik(np.array([float(D_um2_s)]), w["lam"], w["y"][None], w["const"])[0])
