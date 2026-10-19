"""Classes of D on a track graph: a hidden Potts model fitted by mean-field EM.

Each track i has a log-likelihood row L_i(u) on the ln D grid u (`gridpost.GridLikelihoods`). Class c is a
normal density p_c(u) on the grid with weight w_c, so track i's evidence for it is e_ic = sum_u L_i(u) p_c(u).
Labels z_i follow a Potts prior on the k-nearest-neighbour graph of the track centroids:

    P(z_i = c | neighbours) ∝ w_c exp(beta * fraction of i's neighbours in c)

- beta = 0 is the ordinary mixture; beta > 0 means neighbouring tracks tend to share a class.
- Mean-field EM: the E-step replaces the neighbours' labels by their current class probabilities q_j, and the M-step
  refits each class's normal from the tracks' grid posteriors under it. w maximizes the mean-field
  pseudo-likelihood, and beta how well each track's neighbours predict its own data (`_beta`).
- Whether there is a pattern at all is not read from beta: when neighbours carry nothing, beta and w trade off
  along a flat ridge. `workflow` tests the neighbour gain (`PottsFit.score` over the plain mixture's) against
  shuffled positions instead.
- The class densities only assign tracks. What a class's tracks look like is read from them as a group with the
  population models (`gridpost.deconvolve`, `fbmpost.deconvolve_tracks`), never from p_c.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

BETA_MAX = 30.  # beta's bound; near it beta is weakly identified, which is why the pattern test uses the gain
MIN_SD = .1  # added in quadrature to a class's sd in ln D, so a class never collapses onto one grid cell
SWEEPS = 5  # mean-field sweeps per E-step
BETA_STEP = 1.  # beta moves at most this far per iteration between scans of its whole range
SCAN_EVERY = 10  # iterations between those scans
TINY = 1e-300


def knn(xy: np.ndarray, k: int) -> np.ndarray:
    """(n, k) indices of each point's k nearest other points."""
    xy = np.asarray(xy, float)
    if len(xy) <= k:
        raise ValueError(f"need more than k = {k} tracks, got {len(xy)}")
    _, idx = cKDTree(xy).query(xy, k + 1)
    out = np.empty((len(xy), k), int)
    for i, row in enumerate(idx):  # the point itself is usually first, but not when positions coincide
        others = row[row != i]
        out[i] = others[:k]
    return out


def averaging(neighbours: np.ndarray) -> csr_matrix:
    """(n, n) sparse matrix whose product with q is each track's mean over its neighbours, `q[neighbours].mean(1)`."""
    n, k = neighbours.shape
    return csr_matrix((np.full(n * k, 1 / k), neighbours.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))


def densities(u: np.ndarray, centre: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(len(u), C) normal densities on the grid, each column summing to 1."""
    z = (u[:, None] - centre[None, :]) / sd[None, :]
    p = np.exp(-.5 * z ** 2)
    return p / p.sum(axis=0, keepdims=True)


def _softmax(lp: np.ndarray) -> np.ndarray:
    p = np.exp(lp - lp.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def conditional(neighbour_q: np.ndarray, weight: np.ndarray, beta: float) -> np.ndarray:
    """P(class | neighbours) from the neighbours' mean class probabilities (..., C): the Potts conditional."""
    return _softmax(np.log(weight) + beta * neighbour_q)


@dataclass(frozen=True)
class PottsFit:
    """One fit: classes in increasing order of D. `q` (n, C) are the tracks' class probabilities given their own
    data and their neighbours; `evidence` (n, C) is e_ic relative to each track's likelihood maximum."""
    centre: np.ndarray  # (C,) class means in ln D
    sd: np.ndarray      # (C,) class sds in ln D
    weight: np.ndarray  # (C,)
    beta: float
    q: np.ndarray
    evidence: np.ndarray
    n_iter: int
    converged: bool

    @property
    def n_classes(self) -> int:
        return len(self.centre)

    def neighbour_q(self, neighbours: np.ndarray) -> np.ndarray:
        """(n, C) each track's class probabilities from its neighbours alone: its own data left out."""
        return conditional(self.q[neighbours].mean(axis=1), self.weight, self.beta)

    def score(self, neighbours: np.ndarray, offset: np.ndarray) -> float:
        """Neighbour-predictive log-likelihood: each track's data under the classes its neighbours predict.
        `offset` is each row's log-likelihood maximum, so scores of different C compare."""
        p = self.neighbour_q(neighbours)
        return float(np.sum(np.log((p * self.evidence).sum(axis=1) + TINY) + offset))

    def track_posteriors(self, lik: np.ndarray, u: np.ndarray) -> np.ndarray:
        """(n, len(u)) each track's posterior over ln D with the class mixture as its prior, rows summing to 1."""
        p = densities(u, self.centre, self.sd)
        post = lik * ((self.q / (self.evidence + TINY)) @ p.T)
        return post / post.sum(axis=1, keepdims=True)


def _weights(q: np.ndarray, m_nb: np.ndarray, beta: float, w: np.ndarray) -> np.ndarray:
    """Class weights maximizing the pseudo-likelihood at fixed beta (plain means at beta = 0)."""
    if beta == 0:
        return q.mean(axis=0)
    target = q.sum(axis=0)
    for _ in range(200):
        pr = conditional(m_nb, w, beta)
        step = target / pr.sum(axis=0)
        w = w * step
        w /= w.sum()
        if np.abs(step - 1).max() < 1e-6:
            break
    return w


def _beta(ev: np.ndarray, m_nb: np.ndarray, w: np.ndarray, near: float | None = None) -> float:
    """beta maximizing the neighbour-predictive likelihood sum_i log sum_c P(c | neighbours of i) e_ic: how well
    a track's neighbours predict its own data. The mean-field pseudo-likelihood instead rewards neighbours for
    agreeing with q, which the mean field has already pulled toward them, so it climbs with no pattern at all."""
    log_w = np.log(w)

    def negative(b):
        # conditional() without its max shift: w <= 1 and the neighbour fractions are in [0, 1], so the
        # exponent is at most BETA_MAX.
        e = np.exp(log_w + b * m_nb)
        return -(np.log((e * ev).sum(axis=1)) - np.log(e.sum(axis=1))).sum()
    if near is not None:  # a step from the last beta; the scan below only if it runs to the window's edge
        lo, hi = max(near - BETA_STEP, 0.), min(near + BETA_STEP, BETA_MAX)
        b = float(minimize_scalar(negative, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4}).x)
        if (b - lo > 1e-3 or lo == 0.) and (hi - b > 1e-3 or hi == BETA_MAX):
            return b
    # The objective can have two maxima (near 0 and at a large beta), so a coarse scan picks the basin first.
    grid = np.linspace(0., BETA_MAX, 61)
    i = int(np.argmin([negative(b) for b in grid]))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    return float(minimize_scalar(negative, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4}).x)


def initial(lik: np.ndarray, u: np.ndarray, n_class: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classes spread over the quantiles of the tracks' most likely ln D, each as wide as a third of that range."""
    peak = u[lik.argmax(axis=1)]
    centre = np.quantile(peak, np.linspace(.1, .9, n_class)) if n_class > 1 else np.array([np.median(peak)])
    width = max(float(np.subtract(*np.quantile(peak, [.9, .1]))) / 3, MIN_SD)
    return centre, np.full(n_class, width), np.full(n_class, 1 / n_class)


def fit(lik: np.ndarray, u: np.ndarray, neighbours: np.ndarray, n_class: int, beta: float | None = None,
        init: PottsFit | tuple | None = None, max_iter: int = 300, tol: float = 1e-3) -> PottsFit:
    """Fit `n_class` classes to `lik` (n, len(u)): each track's likelihood on the grid, maximum 1.

    `beta` None learns it; a number fixes it (0: the plain mixture). `init` is an earlier fit or a
    (centre, sd, weight) tuple; by default `initial`.
    """
    if init is None:
        centre, sd, w = initial(lik, u, n_class)
    elif isinstance(init, PottsFit):
        centre, sd, w = init.centre.copy(), init.sd.copy(), init.weight.copy()
    else:
        centre, sd, w = (np.asarray(a, float).copy() for a in init)
    if len(centre) != n_class:
        raise ValueError(f"init has {len(centre)} classes, asked for {n_class}")
    b = 0. if beta is None else float(beta)
    if beta is None and init is not None and isinstance(init, PottsFit):
        b = init.beta
    q = np.full((len(lik), n_class), 1 / n_class)
    avg = averaging(neighbours)
    converged = False
    for it in range(1, max_iter + 1):
        q_old, b_old = q, b
        p = densities(u, centre, sd)
        ev = lik @ p + TINY
        log_w_ev = np.log(w) + np.log(ev)
        for _ in range(SWEEPS):
            q = _softmax(log_w_ev + b * (avg @ q))
        m_nb = avg @ q
        # Each class's grid mass: sum_i q_ic L_i(u) p_c(u) / e_ic.
        mass = (lik.T @ (q / ev)) * p
        total = mass.sum(axis=0)
        centre = u @ mass / total
        var = (u ** 2) @ mass / total - centre ** 2
        sd = np.sqrt(np.maximum(var, 0.) + MIN_SD ** 2)
        if beta is None:
            b = _beta(ev, m_nb, w, None if it <= 3 or it % SCAN_EVERY == 0 else b)
        w = _weights(q, m_nb, b, w)
        if it > 10 and np.abs(q - q_old).max() < tol and abs(b - b_old) < tol:
            converged = True
            break
    p = densities(u, centre, sd)
    ev = lik @ p
    order = np.argsort(centre)
    return PottsFit(centre[order], sd[order], w[order], b, q[:, order], ev[:, order], it, converged)
