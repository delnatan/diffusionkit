"""Distribution of D across many tracks: a smooth log density, its smoothness chosen by evidence.

A population g of any smooth shape, the third member beside `lognormal`'s shared D (complete
pooling) and log-normal g. Population-level comparator to the per-track posteriors in `posterior.py`, not a
replacement for them -- per-track posteriors stay the primary output of this
package; this reports how D is distributed across a whole table of tracks, the
way an ensemble MSD fit would, but without averaging away each track's own
uncertainty first (it uses each track's likelihood, not its posterior, so the
prior is not counted once per track).

The grid weights g (summing to 1 on the prior's support) maximize the mixture
log-likelihood `l(g) = sum_i log sum_k L_ik g_k` under a Gaussian smoothness prior
on eta = log g -- the second-order Tikhonov penalty (lam/2) int eta''(u)^2 du, with
g = softmax(eta) (a logistic-Gaussian-process / log-spline density). The penalty
is blind to constants and linear tilts of log g; the tilt gets a negligible proper
prior (precision `EPS`) so the evidence is defined. lam is the maximum of the
Laplace evidence p(tracks | lam) over `LAM_GRID`, refined in log lam.

The returned `samples` are posterior draws of g with lam integrated out by its
evidence weights, so every draw is non-negative and sums to 1. Read any
functional's interval off them: `band` for pointwise or cumulative bands,
`samples @ a` for a mass or mean. They come from Hamiltonian Monte Carlo
preconditioned by the Laplace approximation (unit-normal momenta mapped through
the Cholesky factor of the Hessian at the mode), not from the Laplace Gaussian
itself: where the data rule a grid cell out, its mass at the mode is ~0, so the
local curvature in log g is ~0 while the true posterior has a cliff (mass m there
costs ~n m nats). Gaussian draws then cross the cliff and pile the mass into
empty cells -- on the default 1e-5..10 grid, past any track. The evidence is
less affected: importance sampling put its error at under 1 nat and nearly
constant across the lam that matter.

On simulated populations (1000 tracks of 5-20 frames; a spike, narrow and broad
modes, a log-normal, a mode below the localization floor) this was best or tied
on held-out likelihood against the unregularized NPMLE and against EM with a
fixed or cross-validated Gaussian blur per step: no single
blur width suited every truth, while the evidence moved lam over four decades
to follow them. On the default grid (scripts/validate_deconvolve.py) the 68%/95% bands covered the population CDF
in 69-73%/89-98% of datasets (89% for two modes narrower than a track can
resolve), and the 97.5% quantile of the mass above 1 um^2/s, where no track
was, stayed at or under 0.003. For a spike, the evidence can run to the rough
end of `LAM_GRID`, which warns. Two limits
remain:

- A mode narrower than the per-track resolution comes out as wide as the
  smoothness prior allows, so read a peak's width as resolution-limited.
- Below the localization floor the data cannot tell D values apart; the bands
  widen there, and a mode's location is under-covered (62% at 68%).

`deconvolve_2d` is the same model on a 2D grid, for a population over two parameters at once (alpha and
D_ref in `fbmpost.population`): the penalty is a Kronecker sum of the 1D one along each axis, each axis with
its own lam, so its null space is {1, u, v, u v} and a linear trend of log g along the diagonal (a correlation
between the two parameters) is not smoothed away. Its two lams are integrated out the same way, over a
lattice in (ln lam_u, ln lam_v) around the evidence maximum.

`deconvolve` itself only needs log-likelihood rows on a common 1D grid. A track
with no information about the parameter has a flat likelihood and leaves the fit
and the evidence exactly unchanged (and the bands, up to Monte Carlo error),
where it would pull a histogram of medians toward the prior's median.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable
from warnings import warn

import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse import csr_array
from scipy.optimize import minimize_scalar

from ..data import Acquisition
from .data import GridDistribution, GridPosteriorAnalysis
from .posterior import flat

EPS = 1e-6  # prior precision on log g's unpenalized linear tilt: there only so the evidence is proper
LAM_GRID = np.geomspace(1e4, 1e-8, 37)  # scanned from smooth to rough, flat prior in log lam
N_CHAINS = 50  # HMC chains, split across lam by evidence weight
TINY = 1e-300
# `deconvolve_2d`: the scan of lam_u = lam_v that finds its start; the step in (ln lam_u, ln lam_v) of the lattice
# lam is integrated over; how far below the evidence maximum a lattice point still carries chains (exp(-6),
# 0.25% of the best point's weight); and a cap on the points fitted.
LAM_GRID_2D = np.geomspace(1e3, 1e-6, 10)
LAM_STEP_2D = 0.75  # ~ `LAM_GRID`'s step, 0.77
LAM_CUT_2D = 6.
MAX_LAM_FITS_2D = 80


@dataclass(frozen=True)
class Deconvolution(GridDistribution):
    """Distribution of a parameter across tracks on grid `u`; zero off the prior's support.

    `weights` is the posterior mode at lam; `samples` are draws of the weights, lam integrated out.
    """

    lam: float                # smoothness at the evidence maximum
    lam_grid: np.ndarray      # scanned lam, descending
    log_evidence: np.ndarray  # Laplace log evidence on lam_grid


@dataclass(frozen=True)
class Deconvolution2D:
    """Distribution of two parameters across tracks on the grid `u` x `v`.

    `weights` (len(u), len(v)) is the posterior mode at `lam`; `samples` (n_samples, len(u), len(v)) are
    draws with lam integrated out, from the lattice points `lams` in proportion to `lam_weights`. Each sums
    to 1.
    """

    u: np.ndarray
    v: np.ndarray
    weights: np.ndarray
    samples: np.ndarray
    lam: tuple[float, float]  # (along u, along v) at the evidence maximum
    log_evidence: float
    lams: np.ndarray         # (n, 2): the lam points the draws come from
    lam_weights: np.ndarray  # (n,): their evidence weights, summing to 1


@dataclass(frozen=True)
class DeconvolvedPopulation(Deconvolution):
    """Distribution of D across a table of tracks, on `u` = ln D (`deconvolve_tracks`)."""

    n_tracks: int  # tracks that contributed a likelihood
    n_excluded: int  # too short (< options.min_frames) or invalid input
    acquisition: Acquisition | None  # None when combined across experiments with different acquisitions


@dataclass
class _Mode:
    """Posterior mode of eta at one lam and its Laplace approximation."""

    lam: float
    x: np.ndarray  # eta at the mode
    C: np.ndarray  # lower Cholesky factor of the negative Hessian in tangent coordinates
    log_evidence: float
    problem: str = ""  # why this mode or its evidence is unreliable, if it is


def _second_difference_penalty(K: int, du: float) -> np.ndarray:
    """Omega with eta' Omega eta ~= int eta''(u)^2 du on K evenly spaced cells; null space {1, u}."""
    D2 = np.diff(np.eye(K), 2, axis=0) / du**2
    return D2.T @ D2 * du


class _LogG:
    """l(softmax(eta)) - (1/2) eta' Q eta in coordinates on the sum-zero hyperplane.

    The tangent basis P is the first K - 1 columns of the Householder reflection
    that swaps 1/sqrt(K) with e_K, so P' M P costs O(K^2) (`_tangent`) rather
    than the O(K^3) of a dense basis. Q(lam) is the smoothness prior's
    precision: here lam times the 1D second-difference penalty, plus EPS on
    its unpenalized linear tilt.
    """

    def __init__(self, L: np.ndarray, du: float):
        self._setup(L)
        self.Omega = _second_difference_penalty(self.K, du)
        ev = np.linalg.eigvalsh(self.Omega)
        self.log_pdet_omega = float(np.sum(np.log(ev[2:])))
        k = np.arange(self.K) - (self.K - 1) / 2
        self.tilt = EPS * np.outer(k, k) / (k @ k)

    def _setup(self, L: np.ndarray):
        self.L, (self.n, self.K) = L, L.shape
        v = np.full(self.K, 1 / np.sqrt(self.K))
        v[-1] -= 1
        self._v = v / np.linalg.norm(v)  # reflection I - 2 v v'
        H = np.eye(self.K) - 2 * np.outer(self._v, self._v)
        self.P = H[:, :-1]  # orthonormal, orthogonal to 1: softmax ignores the constant

    def _tangent(self, M: np.ndarray) -> np.ndarray:
        """P' M P for symmetric M, by applying the reflection on both sides."""
        v = self._v
        W = M - 2 * np.outer(v, v @ M)
        W = W - 2 * np.outer(W @ v, v)
        return W[:-1, :-1]

    def Q(self, lam) -> np.ndarray:
        return lam * self.Omega + self.tilt

    def log_prior_norm(self, lam) -> float:
        """log sqrt(pdet Q(lam)) on the tangent space; the 2 pi factors cancel in the evidence."""
        # lam Omega on Omega's K - 2 nonzero eigendirections (Omega scales as du^-3), EPS on the tilt.
        return 0.5 * ((self.K - 2) * np.log(lam) + self.log_pdet_omega + np.log(EPS))

    def value(self, x: np.ndarray, Q: np.ndarray) -> float:
        g = _softmax(x)
        return float(np.sum(np.log(np.maximum(self.L @ g, TINY)))) - 0.5 * x @ Q @ x

    def local(self, x: np.ndarray, Q: np.ndarray):
        """(value, tangent gradient, tangent negative Hessian) at x."""
        g = _softmax(x)
        m = np.maximum(self.L @ g, TINY)
        R = self.L * g / m[:, None]  # responsibilities, rows sum to 1
        r = R.sum(axis=0)
        grad = r - self.n * g - Q @ x
        neg_hess = R.T @ R - np.diag(r) + self.n * (np.diag(g) - np.outer(g, g)) + Q
        F = float(np.sum(np.log(m))) - 0.5 * x @ Q @ x
        return F, self.P.T @ grad, self._tangent(neg_hess)

    def fit(self, lam, x0: np.ndarray, tol: float = 1e-8, maxiter: int = 300) -> _Mode:
        """Mode at fixed lam by damped Newton, and its Laplace log evidence."""
        Q = self.Q(lam)
        x = x0
        F, gt, H = self.local(x, Q)
        problem, mu = "", 0.
        for _ in range(maxiter):
            C, mu = _cholesky_damped(H, mu / 10)
            step = self.P @ solve_triangular(C, solve_triangular(C, gt, lower=True), lower=True, trans="T")
            dec = gt @ (self.P.T @ step)
            if dec < tol * max(1.0, abs(F)):
                break
            t = 1.0
            while t > 1e-12 and self.value(x + t * step, Q) < F + 1e-4 * t * dec:
                t *= 0.5
            x = x + t * step
            F, gt, H = self.local(x, Q)
        else:
            problem = f"Newton did not converge (decrement {dec:.3g})"
        try:
            C = np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            problem = "negative Hessian not positive definite"
            C, _ = _cholesky_damped(H)
        return _Mode(lam, x, C, F - float(np.sum(np.log(np.diag(C)))) + self.log_prior_norm(lam), problem)

    def hmc(self, mode: _Mode, n_chains: int, n_keep: int, rng: np.random.Generator,
            n_warm: int = 30) -> np.ndarray:
        """(n_chains * n_keep, K) posterior draws of g at mode.lam by Hamiltonian Monte Carlo.

        Preconditioned by the Laplace approximation: eta = eta_hat + B w with B = P C^-T, so
        w ~ N(0, I) where Laplace holds and unit-normal momenta suit every direction. Chains
        start at the mode; the step size adapts during `n_warm` iterations, then stays fixed.
        """
        Q = self.Q(mode.lam)
        B = solve_triangular(mode.C, self.P.T, lower=True).T

        def potential(W):
            X = mode.x[:, None] + B @ W
            G = _softmax(X)
            m = np.maximum(self.L @ G, TINY)
            QX = Q @ X
            grad = G * (self.L.T @ (1 / m)) - self.n * G - QX  # d(l - penalty)/d eta
            return -np.sum(np.log(m), axis=0) + 0.5 * np.sum(X * QX, axis=0), -(B.T @ grad), G

        W = np.zeros((self.K - 1, n_chains))
        U, dU, G = potential(W)
        eps, keep = 0.5, []
        for it in range(n_warm + n_keep):
            e = eps * rng.uniform(0.8, 1.2)
            p = rng.standard_normal(W.shape)
            H0 = U + 0.5 * np.sum(p * p, axis=0)
            Wn, pn, dUn = W.copy(), p - 0.5 * e * dU, dU
            n_leap = int(min(50, np.ceil(1.5 / e)))  # trajectory ~ a quarter period of N(0, 1)
            for j in range(n_leap):
                Wn += e * pn
                Un, dUn, Gn = potential(Wn)
                pn -= (e if j < n_leap - 1 else 0.5 * e) * dUn
            H1 = Un + 0.5 * np.sum(pn * pn, axis=0)
            with np.errstate(invalid="ignore", over="ignore"):
                acc = np.isfinite(H1) & (np.log(rng.random(n_chains)) < H0 - H1)
            W[:, acc], U[acc], dU[:, acc], G[:, acc] = Wn[:, acc], Un[acc], dUn[:, acc], Gn[:, acc]
            if it < n_warm:
                eps *= np.exp(acc.mean() - 0.75)
            else:
                keep.append(G.copy())
        return np.hstack(keep).T


class _LogG2(_LogG):
    """`_LogG` on a flattened (n_u, n_v) grid (u major), with lam = (lam_u, lam_v).

    Q(lam) = lam_u Omega_u (x) I dv + lam_v I du (x) Omega_v -- the 1D penalty along each axis,
    integrated over the other -- plus EPS on the null space's three non-constant directions
    {u, v, u v}. Its eigenvalues are sums of the axes' (Kronecker sum), so the evidence's
    normalizer costs nothing per lam.
    """

    def __init__(self, L: np.ndarray, shape: tuple[int, int], du: float, dv: float):
        self._setup(L)
        nu, nv = shape
        Ou, Ov = _second_difference_penalty(nu, du), _second_difference_penalty(nv, dv)
        self.Omegas = (np.kron(Ou, np.eye(nv)) * dv, np.kron(np.eye(nu), Ov) * du)
        self._sparse_omegas = tuple(csr_array(O) for O in self.Omegas)  # banded, for `hmc_mixture`
        eu, ev = np.linalg.eigvalsh(Ou) * dv, np.linalg.eigvalsh(Ov) * du
        eu[:2], ev[:2] = 0., 0.  # the exact null space {1, u} of each axis
        self._eig = (eu, ev)
        a, b = np.arange(nu) - (nu - 1) / 2, np.arange(nv) - (nv - 1) / 2
        null = (np.kron(a, np.ones(nv)), np.kron(np.ones(nu), b), np.kron(a, b))  # orthogonal, sum zero
        self.tilt = EPS * sum(np.outer(w, w) / (w @ w) for w in null)
        self._null = np.column_stack([w / np.linalg.norm(w) for w in null])  # tilt = EPS null null'

    def Q(self, lam) -> np.ndarray:
        return lam[0] * self.Omegas[0] + lam[1] * self.Omegas[1] + self.tilt

    def log_prior_norm(self, lam) -> float:
        ev = lam[0] * self._eig[0][:, None] + lam[1] * self._eig[1][None, :]
        return 0.5 * (float(np.sum(np.log(ev[ev > 0]))) + 3 * np.log(EPS))

    def hmc_mixture(self, modes: list[_Mode], counts, n_keep: int, rng: np.random.Generator,
                    n_warm: int = 30, progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        """(sum(counts) * n_keep, K) draws of g with lam integrated out: `counts[j]` chains at `modes[j]`'s
        lam, each started at that lam's mode, all moved together by one Hamiltonian Monte Carlo.

        As `hmc`, but every chain is preconditioned by the first mode's Laplace factor (pass the most
        weighted first), so one matrix product moves all of them, where a factor per lam costs a full pass
        over a K x K matrix per lam and leapfrog step. The posteriors at nearby lams differ little in
        shape, and any fixed preconditioner leaves each chain's target exact. Each chain's own lam enters
        through its potential, as lam_u Omega_u + lam_v Omega_v (banded) plus the tilt (rank 3).
        `progress(done, total)` counts its iterations.
        """
        B = solve_triangular(modes[0].C, self.P.T, lower=True).T
        counts = np.asarray(counts, int)
        lam = np.repeat(np.array([m.lam for m in modes], float), counts, axis=0).T  # (2, n_chains)
        x0 = np.repeat(np.array([m.x for m in modes]), counts, axis=0).T  # (K, n_chains)
        Ou, Ov = self._sparse_omegas
        null = self._null
        n_chains = int(counts.sum())

        def potential(W):
            X = x0 + B @ W
            G = _softmax(X)
            m = np.maximum(self.L @ G, TINY)
            QX = lam[0] * (Ou @ X) + lam[1] * (Ov @ X) + EPS * (null @ (null.T @ X))
            grad = G * (self.L.T @ (1 / m)) - self.n * G - QX
            return -np.sum(np.log(m), axis=0) + 0.5 * np.sum(X * QX, axis=0), -(B.T @ grad), G

        W = np.zeros((self.K - 1, n_chains))
        U, dU, G = potential(W)
        eps, keep = 0.5, []
        for it in range(n_warm + n_keep):
            e = eps * rng.uniform(0.8, 1.2)
            p = rng.standard_normal(W.shape)
            H0 = U + 0.5 * np.sum(p * p, axis=0)
            Wn, pn, dUn = W.copy(), p - 0.5 * e * dU, dU
            n_leap = int(min(50, np.ceil(1.5 / e)))
            for j in range(n_leap):
                Wn += e * pn
                Un, dUn, Gn = potential(Wn)
                pn -= (e if j < n_leap - 1 else 0.5 * e) * dUn
            H1 = Un + 0.5 * np.sum(pn * pn, axis=0)
            with np.errstate(invalid="ignore", over="ignore"):
                acc = np.isfinite(H1) & (np.log(rng.random(n_chains)) < H0 - H1)
            W[:, acc], U[acc], dU[:, acc], G[:, acc] = Wn[:, acc], Un[acc], dUn[:, acc], Gn[:, acc]
            if it < n_warm:
                eps *= np.exp(acc.mean() - 0.75)
            else:
                keep.append(G.copy())
            if progress is not None:
                progress(it + 1, n_warm + n_keep)
        return np.hstack(keep).T


def _softmax(eta: np.ndarray, axis: int = 0) -> np.ndarray:
    e = np.exp(eta - eta.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _cholesky_damped(H: np.ndarray, mu: float = 0.) -> tuple[np.ndarray, float]:
    """Cholesky factor of H + mu I for the first mu that works, from `mu` up in steps of 10, and that mu.

    Newton passes the previous step's mu over 10: along a path the damping H needs changes
    slowly, and each failed factorization costs as much as a successful one.
    """
    scale = max(np.trace(H) / len(H), 1e-12)
    if mu < 1e-8 * scale:
        mu = 0.
    while True:
        try:
            return np.linalg.cholesky(H + mu * np.eye(len(H)) if mu else H), mu
        except np.linalg.LinAlgError:
            mu = max(10 * mu, 1e-8 * scale)


def deconvolve(lls: np.ndarray, u: np.ndarray, log_prior: np.ndarray | None = None,
               lam_grid: np.ndarray = LAM_GRID, n_samples: int = 1000,
               rng: np.random.Generator | None = None) -> Deconvolution:
    """Distribution of the grid parameter across tracks from per-track log-likelihood rows on `u`.

    `u` must be evenly spaced (lam's scale is set by its spacing). Only the support of
    `log_prior` (default: all of `u`) is used; the weights are zero off it. Warns when the
    evidence maximum sits at an end of `lam_grid`. `rng` (default: seeded) draws `samples`.
    """
    lls, u = np.asarray(lls, float), np.asarray(u, float)
    if lls.ndim != 2 or lls.shape[1] != len(u):
        raise ValueError(f"lls must be (n_tracks, {len(u)}), got {lls.shape}")
    support = np.ones(len(u), bool) if log_prior is None else np.isfinite(log_prior)
    if support.sum() < 3:
        raise ValueError("the prior's support needs at least 3 grid points")
    du = np.diff(u)
    if not np.allclose(du, du[0], rtol=1e-6):
        raise ValueError("u must be evenly spaced")
    ll = lls[:, support]
    model = _LogG(np.exp(ll - ll.max(axis=1, keepdims=True)), float(du[0]))

    lam_grid = np.sort(np.asarray(lam_grid, float))[::-1]
    scan, x = [], np.zeros(model.K)
    for lam in lam_grid:  # smooth to rough, warm-started
        scan.append(model.fit(lam, x))
        x = scan[-1].x
    curve = np.array([m.log_evidence for m in scan])
    i = int(np.argmax(curve))
    best = scan[i]
    if 0 < i < len(lam_grid) - 1:
        cache = {}

        def neg(loglam):
            cache[loglam] = model.fit(float(np.exp(loglam)), scan[i].x)
            return -cache[loglam].log_evidence

        res = minimize_scalar(neg, bounds=(np.log(lam_grid[i + 1]), np.log(lam_grid[i - 1])),
                              method="bounded", options={"xatol": 0.02})
        if -res.fun > best.log_evidence:
            best = cache[res.x]
    else:
        warn(f"deconvolve: evidence maximum at the lam_grid edge (lam={lam_grid[i]:.3g}); widen lam_grid",
             stacklevel=2)

    rng = np.random.default_rng(0) if rng is None else rng
    w = np.exp(curve - curve.max())
    n_keep = -(-n_samples // N_CHAINS)
    chains = rng.multinomial(N_CHAINS, w / w.sum())  # lam integrated out: chains per lam by evidence
    draws = np.vstack([model.hmc(m, c, n_keep, rng) for m, c in zip(scan, chains) if c])[:n_samples]
    for m in {id(m): m for m in [best] + [m for m, c in zip(scan, chains) if c]}.values():
        if m.problem:  # only where it matters: the reported mode or a lam the samples use
            warn(f"deconvolve: {m.problem} at lam={m.lam:.3g}", stacklevel=2)

    def full(g):
        out = np.zeros(g.shape[:-1] + (len(u),))
        out[..., support] = g
        return out

    return Deconvolution(u, full(_softmax(best.x)), full(draws), best.lam, lam_grid, curve)


def deconvolve_2d(lls: np.ndarray, u: np.ndarray, v: np.ndarray, n_samples: int = 1000,
                  lam_grid: np.ndarray = LAM_GRID_2D, rng: np.random.Generator | None = None,
                  lam_start: tuple[float, float] | None = None,
                  lam: tuple[float, float] | None = None,
                  progress: Callable[[str, int, int], None] | None = None) -> Deconvolution2D:
    """Distribution of two grid parameters across tracks from log-likelihoods on `u` x `v`.

    `lls` is (n_tracks, len(u), len(v)); `u` and `v` must each be evenly spaced. lam = (lam_u, lam_v) is
    integrated out under a flat prior in (ln lam_u, ln lam_v), as `deconvolve` integrates its lam: the
    Laplace evidence is evaluated on a lattice of step `LAM_STEP_2D` in ln lam, climbed from its start to a
    maximum and then filled outward over every point within `LAM_CUT_2D` nats of it, and the HMC chains are
    split across those points by evidence weight. The evidence is often nearly flat along a ridge in lam,
    and the draws' spread across it is what the band then carries; a single lam on that ridge sets a band
    that moves with wherever the search stopped. `weights` and `lam` are the mode at the best point.

    The start is the best point of a scan of lam_u = lam_v over `lam_grid` (warning when that is an end of
    it) or, given `lam_start`, that point without the scan -- e.g. the lam of a fit to more of the same
    tracks, or of the same tracks on a coarser grid: the penalty approximates the integral of the squared
    second derivative, so lam means the same at any cell width. With `lam`, there is no search: the mode at
    that lam, and draws at it. `n_samples=0` skips the draws and the fill (the mode is all it returns).
    `rng` (default: seeded) draws `samples`. `progress(stage, done, total)` reports each Laplace fit as
    ("smoothness", fits so far, 0) -- their number is not known ahead -- and the draws as ("sampling",
    iterations, of how many).
    """
    lls, u, v = np.asarray(lls, float), np.asarray(u, float), np.asarray(v, float)
    if lls.ndim != 3 or lls.shape[1:] != (len(u), len(v)):
        raise ValueError(f"lls must be (n_tracks, {len(u)}, {len(v)}), got {lls.shape}")
    if len(u) < 3 or len(v) < 3:
        raise ValueError("each axis needs at least 3 grid points")
    steps = []
    for axis in (u, v):
        d = np.diff(axis)
        if not np.allclose(d, d[0], rtol=1e-6):
            raise ValueError("u and v must be evenly spaced")
        steps.append(float(d[0]))
    flat = lls.reshape(len(lls), -1)
    model = _LogG2(np.exp(flat - flat.max(axis=1, keepdims=True)), (len(u), len(v)), *steps)
    if progress is not None:
        n_fits = 0
        fit = model.fit

        def counted(*args, **kwargs):
            nonlocal n_fits
            mode = fit(*args, **kwargs)
            n_fits += 1
            progress("smoothness", n_fits, 0)
            return mode

        model.fit = counted

    if lam is not None:
        points = {(0, 0): model.fit(tuple(float(x) for x in lam), np.zeros(model.K))}
    else:
        points = _lam_lattice(model, lam_grid, lam_start, fill=n_samples > 0)
    top = max(m.log_evidence for m in points.values())
    used = [m for m in points.values() if m.log_evidence > top - LAM_CUT_2D]
    best = max(used, key=lambda m: m.log_evidence)
    w = np.exp(np.array([m.log_evidence for m in used]) - top)
    w /= w.sum()

    rng = np.random.default_rng(0) if rng is None else rng
    if n_samples:
        n_keep = -(-n_samples // N_CHAINS)
        chains = rng.multinomial(N_CHAINS, w)  # lam integrated out: chains per lattice point by evidence
        sampled = sorted(((m, c) for m, c in zip(used, chains) if c), key=lambda mc: -mc[1])
        draws = model.hmc_mixture([m for m, _ in sampled], [c for _, c in sampled], n_keep, rng,
                                  progress=None if progress is None else lambda d, t: progress("sampling", d, t))
        draws = draws[:n_samples]
        sampled = [m for m, _ in sampled]
    else:
        draws, sampled = np.empty((0, model.K)), []
    for m in {id(m): m for m in [best] + sampled}.values():
        if m.problem:  # only where it matters: the reported mode or a lam the samples use
            warn(f"deconvolve_2d: {m.problem} at lam={m.lam}", stacklevel=2)
    shape = (len(u), len(v))
    return Deconvolution2D(u, v, _softmax(best.x).reshape(shape), draws.reshape(-1, *shape),
                           (float(best.lam[0]), float(best.lam[1])), float(best.log_evidence),
                           np.array([m.lam for m in used], float), w)


def _lam_lattice(model: "_LogG2", lam_grid: np.ndarray, lam_start: tuple[float, float] | None,
                 fill: bool) -> dict[tuple[int, int], _Mode]:
    """The modes `deconvolve_2d` integrates lam over, by lattice point (i, j): (ln lam_u, ln lam_v) =
    origin + `LAM_STEP_2D` (i, j). Each fit is warm-started from the nearest point already fitted."""
    points: dict[tuple[int, int], _Mode] = {}
    if lam_start is None:
        lam_grid = np.sort(np.asarray(lam_grid, float))[::-1]
        scan, x = [], np.zeros(model.K)
        for value in lam_grid:  # lam_u = lam_v, smooth to rough
            scan.append(model.fit((value, value), x))
            x = scan[-1].x
        i = int(np.argmax([m.log_evidence for m in scan]))
        if i in (0, len(lam_grid) - 1):
            warn(f"deconvolve_2d: evidence maximum at the lam_grid edge (lam={lam_grid[i]:.3g}); widen lam_grid",
                 stacklevel=3)
        origin = np.log([lam_grid[i]] * 2)
        points[(0, 0)] = scan[i]
    else:
        origin = np.log(np.asarray(lam_start, float))

    def at(ij: tuple[int, int]) -> _Mode:
        if ij not in points:
            near = min(points, key=lambda k: abs(k[0] - ij[0]) + abs(k[1] - ij[1]), default=None)
            x0 = np.zeros(model.K) if near is None else points[near].x
            points[ij] = model.fit(tuple(np.exp(origin + LAM_STEP_2D * np.array(ij))), x0)
        return points[ij]

    def neighbours(ij):
        return [(ij[0] + 1, ij[1]), (ij[0] - 1, ij[1]), (ij[0], ij[1] + 1), (ij[0], ij[1] - 1)]

    here = (0, 0)
    at(here)
    while len(points) < MAX_LAM_FITS_2D:  # climb: to the best neighbour while one is better
        step = max(neighbours(here), key=lambda ij: at(ij).log_evidence)
        if points[step].log_evidence <= points[here].log_evidence:
            break
        here = step
    if fill:
        queue = [here]
        while queue and len(points) < MAX_LAM_FITS_2D:
            top = max(m.log_evidence for m in points.values())
            ij = queue.pop(0)
            for n in neighbours(ij):
                if n not in points and at(n).log_evidence > top - LAM_CUT_2D:
                    queue.append(n)
    if len(points) >= MAX_LAM_FITS_2D:
        warn(f"deconvolve_2d: stopped at {MAX_LAM_FITS_2D} lam points; the evidence may not be bounded there",
             stacklevel=3)
    return points


def deconvolve_tracks(
    analysis: GridPosteriorAnalysis,
    log_prior: np.ndarray | None = None,
    n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> DeconvolvedPopulation:
    """Distribution of D across the tracks of `analyze_tracks(..., keep_likelihoods=True)`, on its grid.

    Built from the per-track log-likelihood rows the analysis kept; their per-row constant is
    ignored. `log_prior` (on `analysis.options.u_D()`) only sets the support, and defaults to the
    whole grid. Tracks without an "ok" likelihood (too short, invalid input) are counted in
    `n_excluded`.
    """
    if analysis.likelihoods is None:
        raise ValueError("deconvolve_tracks needs the per-track likelihoods: analyze_tracks(..., keep_likelihoods=True)")
    lls = analysis.likelihoods.loglik_D
    if not len(lls):
        raise ValueError("no track had enough frames and valid input to contribute")
    u = analysis.options.u_D()
    prior = flat(u) if log_prior is None else log_prior
    fit = deconvolve(lls, u, prior, n_samples=n_samples, rng=rng)
    return DeconvolvedPopulation(**{f.name: getattr(fit, f.name) for f in fields(fit)},
                                  n_tracks=len(lls), n_excluded=analysis.fits.height - len(lls),
                                  acquisition=analysis.acquisition)
