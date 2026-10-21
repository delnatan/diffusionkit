"""A smooth map of ln D over the cell: the Besag-York-Mollie (BYM) model on the track graph, fitted by Laplace.

Each track i has its own x_i = ln D_i, split into a spatial part and an independent part:

    x_i = s_i + v_i,   p(s) ∝ exp(-tau_s/2 sum_{i~j} (s_i - s_j)^2 - eps/2 |s|^2),   v_i ~ N(0, 1/tau_v)

- s is an intrinsic CAR field on the symmetric k-nearest-neighbour graph (i ~ j if either lists the other). Its
  prior penalizes only differences between neighbours: nearby tracks have similar D, with no number of classes.
  eps = 1 / (n mean_sd^2) makes it proper along the constant direction, so the overall level of ln D has prior
  SD `mean_sd` around 0 and the evidence is defined.
- v is track-to-track scatter with no spatial structure. The data enter through each track's gridpost
  log-likelihood row, interpolated on the grid.
- The posterior of (x, s) is found by Newton steps and approximated by a Gaussian there (Laplace). tau_s and tau_v
  are chosen by the Laplace evidence, the same empirical-Bayes rule `gridpost.deconvolve` uses for its smoothness.
  tau_s -> inf means there is no map; tau_v -> inf means all the variation is spatial.
- Whether there is a pattern is the evidence of this model over the one with no spatial part (x_i = m + v_i): a
  Bayes factor in nats. Its null comes from the same tracks with shuffled positions (`MapNull`).

x | s is coupled to s only through the identity, so every Newton solve and log-determinant reduces to one sparse
n x n system in s (`_Model`). Besag, York & Mollie 1991, Ann. Inst. Stat. Math. 43:1.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, csr_matrix, diags, identity
from scipy.sparse.linalg import splu
from scipy.stats import norm

from ..gridpost.data import GridLikelihoods, GridPosteriorAnalysis
from .potts import graph

LOG_TAU = (-5., 12.)  # bounds on log tau_s and log tau_v; at the upper bound that part has effectively vanished
GRID_S = np.linspace(-2., 8., 11)  # starting grids for log tau_s and log tau_v
GRID_V = np.linspace(-3., 6., 10)
MIN_CURVATURE = 1e-6  # a track's curvature floor in the Newton steps, so a flat or convex row never stalls them
CHUNK = 256  # columns per sparse solve when reading the posterior variances


def laplacian(xy: np.ndarray, k: int) -> csr_matrix:
    """The graph Laplacian (degree minus adjacency) of the symmetric k-nearest-neighbour graph of `xy`."""
    W = graph(xy, k)
    return csr_matrix(diags(np.asarray(W.sum(axis=1)).ravel()) - W)


class _Rows:
    """Each track's log-likelihood on the grid, with its first and second derivatives, linearly interpolated.
    Outside the grid each row is held at its edge value."""

    def __init__(self, ll: np.ndarray, u: np.ndarray):
        self.u0, self.du, self.n_u = u[0], u[1] - u[0], len(u)
        self.f = ll
        self.g = np.gradient(ll, self.du, axis=1)
        self.h = np.gradient(self.g, self.du, axis=1)
        self.r = np.arange(len(ll))

    def at(self, x: np.ndarray, derivatives: bool = True):
        t = np.clip((x - self.u0) / self.du, 0, self.n_u - 1 - 1e-9)
        j = t.astype(int)
        w = t - j
        lin = lambda A: (1 - w) * A[self.r, j] + w * A[self.r, j + 1]
        return (lin(self.f), lin(self.g), lin(self.h)) if derivatives else lin(self.f)


@dataclass(frozen=True)
class _Fit:
    x: np.ndarray       # posterior mode of ln D per track
    s: np.ndarray       # posterior mode of the spatial part (or of the overall level m)
    sd: np.ndarray      # Laplace posterior SD of ln D per track
    evidence: float     # Laplace log evidence, up to the rows' constants
    log_tau_s: float
    log_tau_v: float


class _Model:
    """x = Z s + v with s ~ N(0, K^-1), v ~ N(0, 1/tau_v), and track i's log-likelihood on x_i.

    Spatial: Z = I and K = tau_s L + eps I. No space: Z = 1 (s is the overall level m) and K = 1 / mean_sd^2.
    With D = tau_v + H (H the tracks' curvatures), the joint precision's Schur complement onto s is
    S = K + Z^T diag(tau_v H / D) Z, which is all a Newton step or a log-determinant needs.
    """

    def __init__(self, rows: _Rows, Z: csr_matrix, K: Callable[[float], csc_matrix]):
        self.rows, self.Z, self.K = rows, csr_matrix(Z), K
        self.ZT = csr_matrix(self.Z.T)

    def fit(self, log_tau_s: float, log_tau_v: float, x0: np.ndarray, s0: np.ndarray, max_iter: int = 50,
            tol: float = 1e-6, variances: bool = False) -> _Fit:
        tau_v = np.exp(log_tau_v)
        K = self.K(np.exp(log_tau_s))
        Z, ZT, rows = self.Z, self.ZT, self.rows

        def objective(x, s):
            e = x - Z @ s
            return rows.at(x, False).sum() - .5 * tau_v * e @ e - .5 * s @ (K @ s)

        x, s = x0.copy(), s0.copy()
        val = objective(x, s)
        for _ in range(max_iter):
            _, g, h = rows.at(x)
            H = np.maximum(-h, MIN_CURVATURE)
            D = tau_v + H
            e = x - Z @ s
            rx, rs = g - tau_v * e, tau_v * (ZT @ e) - K @ s
            S = csc_matrix(K + ZT @ diags(tau_v * H / D) @ Z)
            ds = splu(S).solve(rs + tau_v * (ZT @ (rx / D)))
            dx = (rx + tau_v * (Z @ ds)) / D
            t = 1.
            while True:
                new = objective(x + t * dx, s + t * ds)
                if new >= val - 1e-10 or t < 1e-4:
                    break
                t /= 2
            x, s, old, val = x + t * dx, s + t * ds, val, new
            if abs(val - old) < tol:
                break
        _, _, h = rows.at(x)
        H = np.maximum(-h, MIN_CURVATURE)
        D = tau_v + H
        lu = splu(csc_matrix(K + ZT @ diags(tau_v * H / D) @ Z))
        n = len(x)
        # log|Q| - log|A| = n log tau_v + log|K| - sum log D - log|S|; tau_v cancels into log(1 + H / tau_v).
        evidence = val + .5 * (_logdet(splu(csc_matrix(K))) - np.log1p(H / tau_v).sum() - _logdet(lu))
        sd = np.sqrt(1 / D + (tau_v / D) ** 2 * _diag_ZSinvZT(lu, Z, ZT)) if variances else np.full(n, np.nan)
        return _Fit(x, s, sd, float(evidence), float(log_tau_s), float(log_tau_v))

    def best(self, x0: np.ndarray, s0: np.ndarray, spatial: bool = True, log_tau_v0: float | None = None) -> _Fit:
        """The precisions by Laplace evidence: the best point of a coarse grid, then Nelder-Mead in log space.
        Without space only tau_v is searched. `log_tau_v0` (the no-space fit's) adds the grid point where the
        spatial part has vanished, so the map never scores below the model it contains; a sloppy search would
        otherwise bias the shuffled gains low and make the pattern test anti-conservative."""
        cache: dict[tuple[float, float], _Fit] = {}
        start = [x0, s0]

        def at(ls, lv):
            key = (round(float(ls), 4), round(float(lv), 4))
            if key not in cache:
                cache[key] = f = self.fit(ls, lv, *start)
                start[:] = [f.x, f.s]  # warm start: neighbouring precisions have neighbouring modes
            return cache[key]

        grid_s = GRID_S if spatial else [0.]  # without space tau_s does not enter K
        starts = [(ls, lv) for ls in grid_s for lv in GRID_V]
        if spatial and log_tau_v0 is not None:
            starts.append((LOG_TAU[1], log_tau_v0))
        best = max((at(*c) for c in starts), key=lambda f: f.evidence)
        start[:] = [best.x, best.s]
        if spatial:
            r = minimize(lambda p: -at(*p).evidence, [best.log_tau_s, best.log_tau_v], method="Nelder-Mead",
                         bounds=[LOG_TAU, LOG_TAU], options=dict(xatol=.05, fatol=.01, maxiter=80))
            p = r.x
        else:
            r = minimize(lambda p: -at(0., p[0]).evidence, [best.log_tau_v], method="Nelder-Mead",
                         bounds=[LOG_TAU], options=dict(xatol=.05, fatol=.01, maxiter=40))
            p = [0., r.x[0]]
        f = at(*p)
        return f if f.evidence >= best.evidence else best


def _logdet(lu) -> float:
    """log|M| of a symmetric positive-definite matrix from its sparse LU."""
    return float(np.log(np.abs(lu.U.diagonal())).sum())


def _diag_ZSinvZT(lu, Z: csr_matrix, ZT: csr_matrix) -> np.ndarray:
    """diag(Z S^-1 Z^T) by solves against Z^T's columns, CHUNK at a time."""
    n = Z.shape[0]
    out = np.empty(n)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        cols = ZT[:, a:b].toarray()
        out[a:b] = np.einsum("ij,ij->j", cols, lu.solve(cols))
    return out


def _models(ll: np.ndarray, u: np.ndarray, L: csr_matrix, mean_sd: float) -> tuple[_Model, _Model]:
    n = len(ll)
    rows = _Rows(ll, u)
    eps = 1 / (n * mean_sd ** 2)  # the mean of s then has prior SD mean_sd
    I = identity(n, format="csr")
    spatial = _Model(rows, I, lambda tau_s: csc_matrix(tau_s * L + eps * I))
    plain = _Model(rows, csr_matrix(np.ones((n, 1))), lambda _: csc_matrix([[1 / mean_sd ** 2]]))
    return spatial, plain


@dataclass(frozen=True)
class SmoothMapOptions:
    """Everything a smooth-map run depends on besides the likelihoods and the positions.

    - `k`: neighbours per track in the graph, before it is made symmetric.
    - `n_shuffles`: position shuffles for the pattern test's null; p resolves down to 1 / (1 + n_shuffles).
    - `p_pattern`: the p bound at or below which the map counts as spatially organized.
    - `level`: credible-interval mass of the per-track map columns.
    - `mean_sd`: prior SD of the overall level of ln D around 0.
    """
    k: int = 8
    n_shuffles: int = 99
    p_pattern: float = .05
    level: float = .9
    mean_sd: float = 10.
    seed: int = 0

    def __post_init__(self):
        if self.k < 1:
            raise ValueError(f"k must be at least 1, got {self.k}")
        if self.n_shuffles < 0:
            raise ValueError(f"n_shuffles must be >= 0, got {self.n_shuffles}")
        if not 0 < self.level < 1:
            raise ValueError(f"level must be in (0, 1), got {self.level}")
        if self.mean_sd <= 0:
            raise ValueError(f"mean_sd must be positive, got {self.mean_sd}")


@dataclass(frozen=True)
class MapNull:
    """The evidence gain of the map over no space, against the same tracks with their positions shuffled.
    Shuffling keeps every track's data and destroys only the arrangement; the no-space model has no graph, so its
    evidence is the same for every shuffle."""
    gain: float
    shuffled_gain: np.ndarray  # (n_shuffles,)
    p: float                   # (1 + shuffles with gain >= observed) / (1 + n_shuffles)


@dataclass(frozen=True)
class SmoothMap:
    """One smooth-map run over the tracks `track_ids` (the gridpost "ok" tracks that have a position).

    - `ln_D`, `ln_D_sd`: each track's posterior mode and Laplace SD of ln D under the map.
    - `tau_s`, `tau_v`: the precisions of the spatial and independent parts, chosen by evidence.
    - `evidence`, `evidence_plain`: Laplace log evidence of the map and of the model with no space (a common
      constant aside); `gain` is their difference in nats.
    - `null`: the shuffle null of the gain (None with no shuffles).
    """
    track_ids: np.ndarray
    n_frames: np.ndarray
    xy_um: np.ndarray
    options: SmoothMapOptions
    ln_D: np.ndarray
    ln_D_sd: np.ndarray
    tau_s: float
    tau_v: float
    evidence: float
    evidence_plain: float
    null: MapNull | None

    @property
    def gain(self) -> float:
        return self.evidence - self.evidence_plain

    @property
    def patterned(self) -> bool:
        """Whether the shuffle test finds a spatial pattern (p <= p_pattern); without shuffles, a positive gain."""
        return self.null.p <= self.options.p_pattern if self.null is not None else self.gain > 0

    def columns(self) -> pl.DataFrame:
        """Per track: `track_id`, `x_um`, `y_um`, `D_map_um2_s` (exp of the mode) with its `level` interval
        `D_map_lo_um2_s`, `D_map_hi_um2_s`, and `ln_D_map_sd`."""
        z = norm.ppf(.5 + self.options.level / 2)
        return pl.DataFrame({
            "track_id": self.track_ids,
            "x_um": self.xy_um[:, 0], "y_um": self.xy_um[:, 1],
            "D_map_um2_s": np.exp(self.ln_D),
            "D_map_lo_um2_s": np.exp(self.ln_D - z * self.ln_D_sd),
            "D_map_hi_um2_s": np.exp(self.ln_D + z * self.ln_D_sd),
            "ln_D_map_sd": self.ln_D_sd,
        })

    def record(self) -> dict:
        """The run's scalars as plain Python values, for saving next to `columns()`."""
        return dict(options=asdict(self.options), n_tracks=len(self.track_ids), tau_s=self.tau_s, tau_v=self.tau_v,
                    evidence=self.evidence, evidence_plain=self.evidence_plain, gain=self.gain,
                    null=None if self.null is None else dict(gain=self.null.gain, p=self.null.p,
                                                             shuffled_gain=self.null.shuffled_gain.tolist()))


def _positioned(likelihoods: GridLikelihoods, positions: pl.DataFrame):
    pos = positions.select("track_id", "x_um", "y_um").unique("track_id")
    ids = np.asarray(likelihoods.track_ids)
    keep = np.isin(ids, pos["track_id"].to_numpy())
    if not keep.any():
        raise ValueError("no track in the likelihoods has a position")
    ids = ids[keep]
    xy = (pl.DataFrame({"track_id": ids}).join(pos, on="track_id", how="left")
          .select("x_um", "y_um").to_numpy().astype(float))
    if not np.isfinite(xy).all():
        raise ValueError("track positions must be finite")
    return keep, ids, xy


def smooth_map_likelihoods(likelihoods: GridLikelihoods, u_D: np.ndarray, positions: pl.DataFrame,
                           options: SmoothMapOptions = SmoothMapOptions(), *,
                           map_fn: Callable[..., Iterable] = map,
                           progress: Callable[[int, int], None] | None = None) -> SmoothMap:
    """The smooth map of ln D for the tracks in `likelihoods` (gridpost's kept rows on the grid `u_D`) at
    `positions` (`track_id`, `x_um`, `y_um`; `centroids` makes it from a track table). Tracks without a position
    are left out. `map_fn` runs the shuffles (a pool's `map`); `progress(done, total)` counts fits."""
    keep, ids, xy = _positioned(likelihoods, positions)
    ll = np.asarray(likelihoods.loglik_D, float)[keep]
    ll = ll - ll.max(axis=1, keepdims=True)
    u = np.asarray(u_D, float)
    n = len(ids)
    total, done = 2 + options.n_shuffles, 0

    def tick():
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, total)

    spatial, plain = _models(ll, u, laplacian(xy, options.k), options.mean_sd)
    x0 = np.full(n, np.median(u[ll.argmax(axis=1)]))
    p = plain.best(x0, np.array([x0[0]]), spatial=False)
    tick()
    sp = spatial.best(p.x, p.x.copy(), log_tau_v0=p.log_tau_v)
    sp = spatial.fit(sp.log_tau_s, sp.log_tau_v, sp.x, sp.s, variances=True)
    tick()
    gain = sp.evidence - p.evidence

    null = None
    if options.n_shuffles:
        def shuffled(seed) -> float:
            L = laplacian(xy[np.random.default_rng(seed).permutation(n)], options.k)
            m, _ = _models(ll, u, L, options.mean_sd)
            return m.best(p.x, p.x.copy(), log_tau_v0=p.log_tau_v).evidence - p.evidence

        gains = []
        for g in map_fn(shuffled, np.random.SeedSequence(options.seed).spawn(options.n_shuffles)):
            gains.append(g)
            tick()
        gains = np.asarray(gains, float)
        null = MapNull(float(gain), gains, float((1 + (gains >= gain).sum()) / (1 + len(gains))))

    return SmoothMap(ids, np.asarray(likelihoods.n_frames)[keep], xy, options, sp.x, sp.sd,
                     float(np.exp(sp.log_tau_s)), float(np.exp(sp.log_tau_v)), sp.evidence, p.evidence, null)


def smooth_map(result: GridPosteriorAnalysis, positions: pl.DataFrame,
               options: SmoothMapOptions = SmoothMapOptions(), **kw) -> SmoothMap:
    """`smooth_map_likelihoods` on a gridpost run kept with `keep_likelihoods=True`."""
    if result.likelihoods is None:
        raise ValueError("the gridpost run kept no likelihoods: run it with keep_likelihoods=True")
    return smooth_map_likelihoods(result.likelihoods, result.options.u_D(), positions, options, **kw)
