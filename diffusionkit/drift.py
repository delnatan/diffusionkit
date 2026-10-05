"""Drift: a displacement field shared by every track, estimated from the tracks themselves.

Per axis, track i's m displacements, starting at frame f_i, are

    delta_i = S_i dd phi(x_i) + e_i,    e_i ~ N(0, D_i A + B_i)

- dd (n_frames - 1, K): the field's increments between consecutive frames in a polynomial basis phi (Legendre, total
  degree <= `degree`, over the field's bounding box), evaluated at the track's mean position x_i. S_i picks the m
  increments the track spans. The field at x is d(t, x) = cumsum(dd)(t) phi(x), zero at the first frame.
  Degree 0 is one rigid path; 1 adds rotation, shear and stretch; 2 a smooth bend.
- A, B_i: `gridpost.likelihood`'s blurred Brownian and localization covariances. Positions are exposure averages, so
  this is the exposure-averaged drift and subtracting it is exact.
- D_i is unknown. EM over a D grid with a flat prior in ln D: the E-step gives each track's posterior over D (both
  axes), and the M-step is generalized least squares with W_i = E[Sigma_i^-1].
  - Still tracks dominate on their own. A particle with D = 0.3 um^2/s has ~5x a still spot's step SD and gets ~1/25
    of its weight.
  - There is no still/mobile classification and no length cut beyond `min_frames`.
- The inverse normal matrix is the increments' covariance. It is shared by every track: a correlated error, not
  independent per-track noise.

Which degree is enough is a data question. Fit the global field, then ask whether what remains is still shared
between neighbours (`neighbour_correlation`). Motion that neighbours share after a degree-2 field is local
deformation, which no global field removes. Its effect on a track's D is about |v|^2 / (4 dt (1 - 2R)) for a flow of
|v| um per frame, so a ~1 nm/frame flow biases D by ~2e-5 um^2/s and can usually be left alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.polynomial.legendre import legvander
from scipy.linalg import cho_factor, cho_solve, cholesky, eigh, solve_triangular
from scipy.special import logsumexp

from .data import Acquisition
from .gridpost.likelihood import localization_covariance, motion_covariance
from .io import validated_track_frame

__all__ = ["PolyBasis", "Drift", "estimate_drift", "subtract", "neighbour_correlation"]

D_GRID_UM2_S = (1e-5, 10., 121)  # E-step grid: min, max, points, flat in ln D


@dataclass(frozen=True)
class PolyBasis:
    """Legendre polynomials of total degree <= `degree` in (x, y), scaled to [-1, 1] over the box [lo, hi]."""
    degree: int
    lo: tuple[float, float]
    hi: tuple[float, float]

    @classmethod
    def over(cls, tracks: pl.DataFrame, degree: int) -> "PolyBasis":
        lo = (float(tracks["x_um"].min()), float(tracks["y_um"].min()))
        hi = (float(tracks["x_um"].max()), float(tracks["y_um"].max()))
        return cls(degree, lo, hi)

    @property
    def size(self) -> int:
        return (self.degree + 1) * (self.degree + 2) // 2

    def __call__(self, xy) -> np.ndarray:
        """(..., 2) positions in um -> (..., size)."""
        xy = np.asarray(xy, float)
        lo, hi = np.array(self.lo), np.array(self.hi)
        span = np.where(hi > lo, hi - lo, 1.)
        s = (2 * (xy.reshape(-1, 2) - lo) / span - 1).T
        Vx, Vy = legvander(s[0], self.degree), legvander(s[1], self.degree)
        out = np.stack([Vx[:, i] * Vy[:, j] for i in range(self.degree + 1) for j in range(self.degree + 1 - i)],
                       axis=-1)
        return out.reshape(xy.shape[:-1] + (self.size,))


@dataclass
class Drift:
    """d(t, x) = c(t) phi(x), c = cumsum of the per-frame increments, zero at the first frame."""
    frames: np.ndarray  # (n,) frame numbers
    c: np.ndarray  # (2, n, K) field coefficients per axis (x, y) and frame, um
    dd_cov: np.ndarray  # (2, (n-1) K, (n-1) K) covariance of the increments, frame-major
    basis: PolyBasis
    n_tracks: int
    n_iter: int
    converged: bool

    def at(self, xy) -> np.ndarray:
        """Drift path (n, 2) in um at position xy (2,)."""
        return np.einsum("ank,k->na", self.c, self.basis(np.asarray(xy, float)))

    def path_cov(self, xy) -> np.ndarray:
        """(2, n, n) covariance of the drift path at xy."""
        n = len(self.frames)
        C = np.kron(np.tril(np.ones((n, n - 1)), -1), self.basis(np.asarray(xy, float))[None])
        return C @ self.dd_cov @ C.T


def _prepare(tracks: pl.DataFrame, acquisition: Acquisition, min_frames: int, basis: PolyBasis, f0: int):
    """Per track: first increment index, phi at the mean position, and per axis (G, lam, delta) with
    Sigma(D) = L Q diag(1 + D lam) Q^T L^T and G = L^-T Q, so Sigma(D)^-1 = G diag(1 / (1 + D lam)) G^T."""
    out = []
    for group in tracks.partition_by("track_id", maintain_order=True):
        if group.height < min_frames:
            continue
        g = validated_track_frame(group, acquisition)
        xy = g.select("x_um", "y_um").to_numpy()
        delta = np.diff(xy, axis=0).T
        A = motion_covariance(delta.shape[1], acquisition.dt_s, acquisition.exposure_s)
        axes = []
        for B, dl in zip(localization_covariance(g.select("sigma_x_um", "sigma_y_um").to_numpy()), delta):
            L = cholesky(B, lower=True)
            M = solve_triangular(L, solve_triangular(L, A, lower=True).T, lower=True)
            lam, Q = eigh((M + M.T) / 2)
            axes.append((solve_triangular(L, Q, lower=True, trans="T"), lam, dl))
        out.append((int(g["frame"][0]) - f0, basis(xy.mean(0)), axes))
    return out


def estimate_drift(tracks: pl.DataFrame, acquisition: Acquisition, degree: int = 2, min_frames: int = 3,
                   n_iter: int = 50, tol_um: float = 1e-4) -> Drift:
    """The drift field shared by every track of >= `min_frames` frames.

    Columns: track_id, frame, x_um, y_um, sigma_x_um, sigma_y_um (per-frame localization SDs > 0); frames
    consecutive within a track. Converged when no increment of the field, evaluated at the tracks, moves by more
    than `tol_um` between EM iterations (`Drift.converged`).
    """
    if degree < 0:
        raise ValueError("degree must be >= 0")
    tracks = tracks.sort("track_id", "frame")
    f0, f1 = int(tracks["frame"].min()), int(tracks["frame"].max())
    n = f1 - f0 + 1
    if n < 2:
        raise ValueError("drift needs at least two frames")
    basis = PolyBasis.over(tracks, degree)
    K = basis.size
    N = (n - 1) * K
    prepared = _prepare(tracks, acquisition, max(min_frames, 2), basis, f0)
    if not prepared:
        raise ValueError(f"no track has >= {min_frames} frames")
    D = np.geomspace(*D_GRID_UM2_S)
    dd = np.zeros((2, n - 1, K))
    converged = False
    for it in range(1, n_iter + 1):
        F = np.zeros((2, N, N))
        b = np.zeros((2, N))
        for s0, phi, axes in prepared:
            m = len(axes[0][2])
            rows = ((s0 + np.arange(m))[:, None] * K + np.arange(K)[None, :]).ravel()
            ll = np.zeros(len(D))
            for a, (G, lam, dl) in enumerate(axes):
                y = G.T @ (dl - dd[a, s0:s0 + m] @ phi)
                v = 1 + D[:, None] * lam
                ll -= .5 * np.sum(np.log(v) + y ** 2 / v, axis=1)
            p = np.exp(ll - logsumexp(ll))
            pp = np.outer(phi, phi)
            for a, (G, lam, dl) in enumerate(axes):
                W = (G * (p @ (1 / (1 + D[:, None] * lam)))) @ G.T
                F[a][np.ix_(rows, rows)] += np.kron(W, pp)
                b[a, rows] += np.kron(W @ dl, phi)
        try:
            new = np.stack([cho_solve(cho_factor(F[a]), b[a]).reshape(n - 1, K) for a in range(2)])
        except np.linalg.LinAlgError as e:
            raise ValueError("drift is not identified: some frame or basis direction has no data "
                             "(lower the degree, or check that tracks cover every frame)") from e
        moved = max(np.max(np.abs((new[a] - dd[a])[s0:s0 + len(ax[0][2])] @ phi))
                    for s0, phi, ax in prepared for a in range(2))
        dd = new
        if moved < tol_um:
            converged = True
            break
    cov = np.stack([cho_solve(cho_factor(F[a]), np.eye(N)) for a in range(2)])
    c = np.concatenate([np.zeros((2, 1, K)), np.cumsum(dd, axis=1)], axis=1)
    return Drift(np.arange(f0, f1 + 1), c, cov, basis, len(prepared), it, converged)


def subtract(tracks: pl.DataFrame, drift: Drift) -> pl.DataFrame:
    """Positions minus the drift field at each track's mean position. Frames outside the drift's range raise."""
    f = tracks["frame"].to_numpy()
    if f.min() < drift.frames[0] or f.max() > drift.frames[-1]:
        raise ValueError("tracks have frames outside the drift's range")
    mean = tracks.group_by("track_id").agg(pl.col("x_um").mean().alias("_mx"), pl.col("y_um").mean().alias("_my"))
    t = tracks.join(mean, on="track_id", how="left", maintain_order="left")
    phi = drift.basis(t.select("_mx", "_my").to_numpy())
    d = np.einsum("ark,rk->ra", drift.c[:, t["frame"].to_numpy() - drift.frames[0], :], phi)
    return t.with_columns(pl.col("x_um") - d[:, 0], pl.col("y_um") - d[:, 1]).drop("_mx", "_my")


def neighbour_correlation(tracks: pl.DataFrame, radii_um=(5., 8., 12., 20.), min_frames: int = 40,
                          n_null: int = 50, seed: int = 0) -> pl.DataFrame:
    """Is net displacement shared with neighbours? A model-free check for drift left after a correction.

    For every track of >= `min_frames` frames, the net displacement per frame (mean of its last 6 positions minus
    its first 6, over the frames between) is predicted by the median of the other such tracks' within r of its mean
    position (>= 3 of them). Returns per radius the Pearson correlation of prediction and truth over both axes, and
    the same with positions shuffled among the tracks (mean and SD over `n_null` shuffles). Shared drift on a scale
    L gives a correlation well above the null for r < L; a particle's own motion doesn't.
    """
    long = tracks.filter(pl.len().over("track_id") >= max(min_frames, 12)).sort("track_id", "frame")
    P, V = [], []
    for g in long.partition_by("track_id", maintain_order=True):
        xy = g.select("x_um", "y_um").to_numpy()
        f = g["frame"].to_numpy()
        P.append(xy.mean(0))
        V.append((xy[-6:].mean(0) - xy[:6].mean(0)) / (f[-6:].mean() - f[:6].mean()))
    schema = {"radius_um": pl.Float64, "corr": pl.Float64, "null_mean": pl.Float64, "null_sd": pl.Float64,
              "n_tracks": pl.Int64}
    if len(P) < 5:
        return pl.DataFrame(schema=schema)
    P, V = np.array(P), np.array(V)
    rng = np.random.default_rng(seed)

    def corr(pos, r):
        dist = np.linalg.norm(pos[:, None] - pos[None], axis=2)
        np.fill_diagonal(dist, np.inf)
        pred = np.full_like(V, np.nan)
        for i in range(len(pos)):
            m = dist[i] < r
            if m.sum() >= 3:
                pred[i] = np.median(V[m], 0)
        ok = ~np.isnan(pred[:, 0])
        if ok.sum() < 5:
            return np.nan, int(ok.sum())
        return float(np.corrcoef(V[ok].ravel(), pred[ok].ravel())[0, 1]), int(ok.sum())

    rows = []
    for r in radii_um:
        c, n = corr(P, r)
        null = np.array([corr(P[rng.permutation(len(P))], r)[0] for _ in range(n_null)])
        rows.append((float(r), c, float(np.nanmean(null)), float(np.nanstd(null)), n))
    return pl.DataFrame(rows, schema=schema, orient="row")
