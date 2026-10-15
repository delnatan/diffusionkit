"""The exact likelihood of blurred fBm with known localization SDs, whitened once per (track, alpha).

Per axis, the m = n - 1 displacements are Gaussian with

    Sigma(alpha, K) = K C_alpha + B,

C_alpha the box-shutter-blurred fGn covariance per unit K (`covariance.fgn_motion_covariance`; the
true motion's MSD is 2 K tau^alpha per axis) and B the localization term of `gridpost.likelihood`.
For each alpha, with B = L L^t and L^-1 C_alpha L^-t = Q diag(lam) Q^t, the whitened data
y = Q^t L^-1 delta are independent with variances 1 + K lam_k. So after one eigendecomposition per
(track, axis, alpha) the likelihood at any K costs O(m) (`loglik_points`), which is what the profile over K
(`profile`) and the population cells (`cells`) evaluate. `whiten_batch` does this for every track
of one length at once.

D_ref = K tau_ref^(alpha - 1) is the D of the true motion at the lag tau_ref (`FbmPostOptions`).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import polars as pl

from ..data import Acquisition
from ..gridpost.likelihood import _prepared
from .covariance import fgn_motion_covariance
from .data import FbmPostOptions, JointGrid

# Upper bound on the floats in one chunk's (tracks, chunk, m, m) and (chunk, points, m) arrays.
_CHUNK_FLOATS = 4_000_000
# Cells further than this below a track's maximum are stored at it (e^-200: no mixture can tell).
_LOGLIK_FLOOR = -200.
# Points per log10 D_ref cell over which `cells` averages the likelihood (a cell is 0.2 decades by default).
_CELL_POINTS = 8


class Whitened(NamedTuple):
    """`whiten_batch`'s output for T tracks on the alphas a: per (track, alpha), y ~ N(0, 1 + K lam) independently."""
    lam: np.ndarray    # (T, n_a, 2m) both axes side by side
    y2: np.ndarray     # (T, n_a, 2m) squared whitened data
    const: np.ndarray  # (T,) -sum ln diag L - m ln(2 pi), over both axes

    def row(self, k: int) -> "Whitened":
        return Whitened(self.lam[k:k + 1], self.y2[k:k + 1], self.const[k:k + 1])


def whiten_batch(xy: np.ndarray, sd: np.ndarray, dt_s: float, exposure_s: float, alphas: np.ndarray) -> Whitened:
    """The per-alpha whitening of T tracks of one length, done together.

    `xy`, `sd` are (T, n, 2) positions and per-frame localization SDs (> 0). The blurred fGn matrices
    C_alpha depend only on the length, so they are built once for all T tracks, and every (track, alpha)
    eigendecomposition runs in one batched call per chunk.
    """
    delta = np.diff(np.ascontiguousarray(xy, float), axis=1)  # (T, m, 2)
    var = np.ascontiguousarray(sd, float) ** 2  # one memory layout: one round-off, whatever the caller passes
    T, m = delta.shape[:2]
    alphas = np.atleast_1d(np.asarray(alphas, float))
    C = fgn_motion_covariance(m, dt_s, alphas, exposure_s)  # (n_a, m, m) per unit K
    lam, y2 = np.zeros((T, len(alphas), 2 * m)), np.zeros((T, len(alphas), 2 * m))
    const = np.zeros(T)
    i = np.arange(m)
    step = max(1, _CHUNK_FLOATS // (T * m * m))
    for axis in range(2):
        B = np.zeros((T, m, m))  # `gridpost.likelihood.localization_covariance`, batched
        B[:, i, i] = var[:, :-1, axis] + var[:, 1:, axis]
        B[:, i[:-1], i[1:]] = B[:, i[1:], i[:-1]] = -var[:, 1:-1, axis]
        L = np.linalg.cholesky(B)
        Li = np.linalg.inv(L)
        z = np.einsum("tij,tj->ti", Li, delta[:, :, axis])
        const += -np.log(np.diagonal(L, axis1=1, axis2=2)).sum(axis=1) - .5 * m * np.log(2 * np.pi)
        cols = slice(axis * m, (axis + 1) * m)
        for s in range(0, len(alphas), step):
            M = Li[:, None] @ C[None, s:s + step] @ np.swapaxes(Li, 1, 2)[:, None]  # (T, chunk, m, m)
            values, Q = np.linalg.eigh((M + np.swapaxes(M, -1, -2)) / 2)
            lam[:, s:s + step, cols] = np.clip(values, 0, None)  # C_alpha is PSD; round-off can dip below 0
            y2[:, s:s + step, cols] = np.einsum("taij,ti->taj", Q, z) ** 2
    return Whitened(lam, y2, const)


def whiten_track(track: pl.DataFrame, acquisition: Acquisition, alphas: np.ndarray) -> Whitened:
    """`whiten_batch` for one validated track's rows."""
    track = _prepared(track, acquisition)
    xy, sd = (track.select(c).to_numpy()[None] for c in (["x_um", "y_um"], ["sigma_x_um", "sigma_y_um"]))
    return whiten_batch(xy, sd, float(acquisition.dt_s), float(acquisition.exposure_s), alphas)


def loglik_points(w: Whitened, ln_K: np.ndarray) -> np.ndarray:
    """The log-likelihood at ln K, (T, n_a, P) for ln K (T, n_a, P); ln K = -inf is K = 0. Chunked, so any
    number of points fits in memory."""
    T, n_a, P = np.shape(ln_K)
    m2 = w.lam.shape[2]
    R = T * n_a
    lam, y2 = w.lam.reshape(R, m2), w.y2.reshape(R, m2)
    const = np.repeat(w.const, n_a)
    lk = np.reshape(ln_K, (R, P))
    out = np.empty((R, P))
    pc = max(1, min(P, _CHUNK_FLOATS // m2))
    rc = max(1, _CHUNK_FLOATS // (m2 * pc))
    for r in range(0, R, rc):
        rows = slice(r, r + rc)
        for p in range(0, P, pc):
            v = 1 + np.exp(lk[rows, p:p + pc, None]) * lam[rows, None, :]
            out[rows, p:p + pc] = const[rows, None] - .5 * (np.log(v) + y2[rows, None, :] / v).sum(-1)
    return out.reshape(T, n_a, P)


def ln_K_of(log10_D_ref: np.ndarray, alphas: np.ndarray, tau_ref_s: float) -> np.ndarray:
    """ln K of D_ref = K tau_ref^(alpha - 1), (n_a, n_x) for alphas (n_a,) and log10 D_ref (n_x,)."""
    return np.log(10) * np.asarray(log10_D_ref)[None, :] - (np.asarray(alphas)[:, None] - 1) * np.log(tau_ref_s)


def surface(w: Whitened, alphas: np.ndarray, log10_D_ref: np.ndarray, tau_ref_s: float) -> np.ndarray:
    """(T, n_a, n_x) log-likelihood on alphas x log10 D_ref, `w` whitened on `alphas`."""
    ln_K = ln_K_of(log10_D_ref, alphas, tau_ref_s)
    return loglik_points(w, np.broadcast_to(ln_K, (len(w.const),) + ln_K.shape))


def log_likelihood_surface(track: pl.DataFrame, acquisition: Acquisition, options: FbmPostOptions,
                           log10_D_ref: np.ndarray) -> np.ndarray:
    """One track's log-likelihood on `options.alphas()` x `log10_D_ref` (um^2/s at `options.tau_ref_s`), (n_a, n_x).

    Free of priors; what a per-track picture of the (alpha, D_ref) trade-off draws.
    """
    alphas = options.alphas()
    return surface(whiten_track(track, acquisition, alphas), alphas, log10_D_ref, options.tau_ref_s)[0]


def cells(w: Whitened, options: FbmPostOptions, joint: JointGrid) -> np.ndarray:
    """(T, n_alpha_cells, n_D_cells) log of each track's likelihood averaged over `joint`'s cells, maximum 0.

    `w` is whitened on `options.alphas()`; a cell averages `joint.alpha_group` alpha rows and `_CELL_POINTS`
    evenly spaced log10 D_ref points, in linear scale, so a likelihood narrower than a cell is not aliased.
    """
    centres = joint.log10_D_ref()
    offsets = joint.d_log10_D_ref * ((np.arange(_CELL_POINTS) + .5) / _CELL_POINTS - .5)
    x = (centres[:, None] + offsets[None, :]).ravel()
    n_a, n_d = len(joint.alphas(options)), len(centres)
    rows = n_a * joint.alpha_group
    ll = surface(Whitened(w.lam[:, :rows], w.y2[:, :rows], w.const), options.alphas()[:rows], x, options.tau_ref_s)
    T = len(ll)
    ll = ll.reshape(T, n_a, joint.alpha_group, n_d, _CELL_POINTS)
    top = ll.max(axis=(1, 2, 3, 4), keepdims=True)
    with np.errstate(divide="ignore"):
        out = np.log(np.exp(ll - top).mean(axis=(2, 4)))
    out -= out.max(axis=(1, 2), keepdims=True)
    return np.maximum(out, _LOGLIK_FLOOR)


def localization_floor(track: pl.DataFrame, options: FbmPostOptions) -> float:
    """The D_ref (um^2/s) at which the true motion's MSD at tau_ref equals the localization noise's
    MSD offset, 2 <s^2> per axis: <s^2> / tau_ref, with <s^2> the mean localization variance over the
    track's frames and both axes. A reference scale to draw next to D_ref, not a threshold."""
    sd = track.select("sigma_x_um", "sigma_y_um").to_numpy()
    return float(np.mean(np.ascontiguousarray(sd) ** 2) / options.tau_ref_s)
