"""Brownian displacement likelihood with known per-frame localization SDs.

Per axis, the m = n - 1 displacements of one track are Gaussian with

    Sigma(D) = D A + B,

where A is Brownian motion averaged over the exposure (Berglund 2010, box
shutter closed form: R = exposure_s / (6 dt_s), diagonal 2 dt (1 - 2R),
lag-1 covariance +2 dt R, zero beyond lag 1) and B comes from independent
localization errors: B_ii = s_i^2 + s_(i+1)^2, B_(i,i+1) = -s_(i+1)^2.
D >= 0; D = 0 means localization noise alone.

Sigma(D) is tridiagonal, so its LDLt factorization is a two-term recursion
along the track, run for every grid point and every track of one length at
once (`brownian_loglik_batch`): O(m) per grid point and track, with no
eigendecomposition.
"""
import numpy as np
import polars as pl

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


def brownian_loglik_batch(xy: np.ndarray, sd: np.ndarray, dt_s: float, exposure_s: float,
                          D: np.ndarray) -> np.ndarray:
    """(T, len(D)) exact log-likelihood of T tracks of one length at each D >= 0.

    `xy` and `sd` are (T, n, 2) positions and per-frame localization SDs (> 0). Per axis,
    Sigma(D) = D A + B is tridiagonal with diagonal D a0 + s_k^2 + s_(k+1)^2 and off-diagonal
    D a1 - s_(k+1)^2 (`motion_covariance`, `localization_covariance`), and its LDLt factorization
    d_k = Sigma_kk - e_k^2 / d_(k-1), with the forward-substituted data z_k, gives
    log L = -1/2 sum_k (ln d_k + z_k^2 / d_k) - (m/2) ln 2 pi per axis.
    """
    delta = np.diff(np.ascontiguousarray(xy, float), axis=1)  # (T, m, 2)
    var = np.ascontiguousarray(sd, float) ** 2  # one memory layout: one round-off, whatever the caller passes
    T, m = delta.shape[:2]
    R = exposure_s / (6 * dt_s)
    a0, a1 = 2 * dt_s * (1 - 2 * R), 2 * dt_s * R
    D = np.asarray(D, float)[None, :]
    out = np.full((T, D.shape[1]), -m * np.log(2 * np.pi))
    for axis in range(2):
        diag = var[:, :-1, axis] + var[:, 1:, axis]  # (T, m)
        off = -var[:, 1:-1, axis]  # (T, m - 1): Sigma_(k-1, k) = D a1 - s_k^2
        d = D * a0 + diag[:, :1]
        z = np.broadcast_to(delta[:, :1, axis], d.shape)
        out -= .5 * (np.log(d) + z * z / d)
        for k in range(1, m):
            e = D * a1 + off[:, k - 1:k]
            ell = e / d
            d = D * a0 + diag[:, k:k + 1] - ell * e
            z = delta[:, k:k + 1, axis] - ell * z
            out -= .5 * (np.log(d) + z * z / d)
    return out


def brownian_log_likelihood(track: pl.DataFrame, acquisition: Acquisition, D_um2_s: float) -> float:
    """Exact Gaussian log-likelihood of both axes' displacements at D >= 0."""
    track = _prepared(track, acquisition)
    xy, sd = (track.select(c).to_numpy()[None] for c in (["x_um", "y_um"], ["sigma_x_um", "sigma_y_um"]))
    return float(brownian_loglik_batch(xy, sd, float(acquisition.dt_s), float(acquisition.exposure_s),
                                       np.array([float(D_um2_s)]))[0, 0])
