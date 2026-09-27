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
-- the same box-shutter average at a general alpha -- in place of
`motion_covariance`) is built on.
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


# Below this exposure/lag ratio `_blurred_abs_power` uses its series (through
# (te/x)^12, truncation ~1e-16 here) rather than the exact second difference,
# which cancels: its relative error is ~ eps (x/te)^2, ~2e-14 at the switch.
_BLUR_SERIES_RATIO = .1
_BLUR_SERIES_TERMS = 6


def _blurred_abs_power(x: np.ndarray, alpha, exposure_s: float) -> np.ndarray:
    """E|x + u - v|^alpha for independent u, v ~ Uniform(0, exposure_s); x and alpha broadcast.

    w = u - v has the triangular density on [-te, te], so the average is
    (H(x+te) - 2 H(x) + H(x-te)) / te^2 with H'' = |x|^alpha, i.e.
    H(x) = |x|^(alpha+2) / ((alpha+1)(alpha+2)). Where te/|x| is small that
    difference cancels, so there the binomial series of |x|^alpha (1 + w/x)^alpha
    is used instead, with the even moments E[w^2k] = 2 te^2k / ((2k+1)(2k+2)).
    """
    x = np.abs(np.asarray(x, dtype=float))
    alpha = np.asarray(alpha, dtype=float)
    if exposure_s == 0:
        return x**alpha
    te = exposure_s
    series = x * _BLUR_SERIES_RATIO > te
    x_series = np.where(series, x, 1.)
    r2 = np.where(series, (te / x_series) ** 2, 0.)
    total, binom, power = 1., 1., 1.
    for k in range(1, _BLUR_SERIES_TERMS + 1):
        n = 2 * k
        binom = binom * (alpha - n + 2) * (alpha - n + 1) / ((n - 1) * n)  # C(alpha, n)
        power = power * r2
        total = total + binom * (2 / ((n + 1) * (n + 2))) * power
    exact = (np.abs(x + te) ** (alpha + 2) - 2 * x ** (alpha + 2) + np.abs(x - te) ** (alpha + 2)) / (
        (alpha + 1) * (alpha + 2) * te**2)
    return np.where(series, x_series**alpha * total, exact)


def fgn_autocovariance(n_disp: int, dt_s: float, alpha, exposure_s: float = 0.) -> np.ndarray:
    """gamma(k), k = 0 .. n_disp-1: `fgn_motion_covariance`'s first row, per unit K.

    `alpha` may be an array: the result has shape `np.shape(alpha) + (n_disp,)`.
    """
    alpha = np.asarray(alpha, dtype=float)[..., None]
    G = _blurred_abs_power(np.arange(-1, n_disp + 1) * dt_s, alpha, exposure_s)  # lags -1 .. n_disp
    return G[..., 2:] - 2 * G[..., 1:-1] + G[..., :-2]


def fgn_motion_covariance(n_disp: int, dt_s: float, alpha, exposure_s: float = 0.) -> np.ndarray:
    """(n_disp, n_disp) fBm displacement covariance per unit K, box-shutter blurred.

    The true motion has structure function E[(z(t+s) - z(t))^2] = 2 K |s|^alpha
    per axis, and each frame records its average over the exposure
    (0 <= exposure_s <= dt_s). With G(x) = E|x + u - v|^alpha over the two
    frames' exposure offsets (`_blurred_abs_power`),

        gamma(k) = G((k+1) dt) - 2 G(k dt) + G((k-1) dt)   (lag k).

    exposure_s=0 gives plain fGn, dt^alpha (|k+1|^alpha - 2|k|^alpha +
    |k-1|^alpha). alpha=1 gives `motion_covariance` (the Berglund box-shutter
    average) at any exposure, so this is its generalization, not a second
    blur model. Blur correlates adjacent displacements positively, which an
    unblurred model reads as a higher alpha. An array `alpha` gives a stack of
    matrices, shape `np.shape(alpha) + (n_disp, n_disp)`.
    """
    gamma = fgn_autocovariance(n_disp, dt_s, alpha, exposure_s)
    i = np.arange(n_disp)
    return gamma[..., np.abs(i[:, None] - i[None, :])]


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
