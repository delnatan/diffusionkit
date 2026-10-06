"""Distribution of D across many tracks: a smooth log density, its smoothness chosen by evidence.

Population-level comparator to the per-track posteriors in `posterior.py`, not a
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
fixed or cross-validated Gaussian blur per step, which it replaces: no single
blur width suited every truth, while the evidence moved lam over four decades
to follow them. On the default grid (scripts/validate_deconvolve.py,
audit/deconvolve_validation.json) the 68%/95% bands covered the population CDF
in 69-73%/89-98% of datasets (89% for two modes narrower than a track can
resolve), and the 97.5% quantile of the mass above 1 um^2/s, where no track
was, stayed at or under 0.003. For a spike, the evidence can run to the rough
end of `LAM_GRID`, which warns. Two limits
remain:

- A mode narrower than the per-track resolution comes out as wide as the
  smoothness prior allows, so read a peak's width as resolution-limited.
- Below the localization floor the data cannot tell D values apart; the bands
  widen there, and a mode's location is under-covered (62% at 68%).

`deconvolve` itself only needs log-likelihood rows on a common 1D grid. A track
with no information about the parameter has a flat likelihood and leaves the fit
and the evidence exactly unchanged (and the bands, up to Monte Carlo error),
where it would pull a histogram of medians toward the prior's median.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from warnings import warn

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar

from ..data import Acquisition
from .data import GridPosteriorAnalysis
from .posterior import flat

EPS = 1e-6  # prior precision on log g's unpenalized linear tilt: there only so the evidence is proper
LAM_GRID = np.geomspace(1e4, 1e-8, 37)  # scanned from smooth to rough, flat prior in log lam
N_CHAINS = 50  # HMC chains, shared across lam by evidence weight
TINY = 1e-300


@dataclass(frozen=True)
class Deconvolution:
    """Distribution of a parameter across tracks on grid `u`; zero off the prior's support."""

    u: np.ndarray
    weights: np.ndarray       # posterior mode at lam, sums to 1
    samples: np.ndarray       # (n_samples, len(u)) draws of the weights, lam integrated out
    lam: float                # smoothness at the evidence maximum
    lam_grid: np.ndarray      # scanned lam, descending
    log_evidence: np.ndarray  # Laplace log evidence on lam_grid

    def band(self, level: float = .68, cumulative: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise (lower, upper) equal-tailed band of the weights, or of their cumulative sum."""
        s = np.cumsum(self.samples, axis=1) if cumulative else self.samples
        q = (1 - level) / 2
        lo, hi = np.quantile(s, [q, 1 - q], axis=0)
        return lo, hi

    def mass(self, lo: float = 0., hi: float = np.inf) -> np.ndarray:
        """(n_samples,) draws of the mass with lo <= D < hi (the parameter's own units, not ln), whole grid cells."""
        with np.errstate(divide="ignore"):
            inside = (self.u >= np.log(lo)) & (self.u < np.log(hi))
        return self.samples[:, inside].sum(axis=1)


@dataclass(frozen=True)
class PopulationDistribution(Deconvolution):
    """Distribution of D across a table of tracks, on `u` = ln D."""

    n_tracks: int  # tracks that contributed a likelihood
    n_excluded: int  # too short (< options.min_frames) or invalid input
    acquisition: Acquisition | None  # None when pooled across experiments with different acquisitions


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


def deconvolve_tracks(
    analysis: GridPosteriorAnalysis,
    log_prior: np.ndarray | None = None,
    n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> PopulationDistribution:
    """Distribution of D across the tracks of `analyze_tracks(..., keep_posteriors=True)`, on its grid.

    Built from the per-track posteriors the analysis kept: their prior is flat in ln D, so each
    row is the track's likelihood up to a constant, which the fit ignores. `log_prior` (on
    `analysis.options.u_D()`) only sets the support, and defaults to the whole grid. Tracks
    without an "ok" posterior (too short, invalid input) are counted in `n_excluded`.
    """
    if analysis.posteriors is None:
        raise ValueError("deconvolve_tracks needs the per-track posteriors: analyze_tracks(..., keep_posteriors=True)")
    lls = analysis.posteriors.log_post_D
    if not len(lls):
        raise ValueError("no track had enough frames and valid input to contribute")
    u = analysis.options.u_D()
    prior = flat(u) if log_prior is None else log_prior
    fit = deconvolve(lls, u, prior, n_samples=n_samples, rng=rng)
    return PopulationDistribution(**{f.name: getattr(fit, f.name) for f in fields(fit)},
                                  n_tracks=len(lls), n_excluded=analysis.fits.height - len(lls),
                                  acquisition=analysis.acquisition)
