"""Classes of D on a track graph: a hidden Potts model fitted by EM, beta by its exact M-step.

Each track i has a log-likelihood row L_i(u) on the ln D grid u (`gridpost.GridLikelihoods`). Class c is a
normal density p_c(u) on the grid, so track i's evidence for it is e_ic = sum_u L_i(u) p_c(u). Labels z follow a
Potts prior on the symmetric k-nearest-neighbour graph of the track centroids (i ~ j if either lists the other):

    P(z | beta) = exp(beta S(z)) / Z(beta),   S(z) = number of edges whose two tracks share a class,

so P(z_i = c | neighbours) ∝ exp(beta * number of i's neighbours in c). There are no class weights.

- beta = 0 is the plain mixture with equal weights; beta > 0 means neighbouring tracks tend to share a class.
- EM: the E-step is mean field (each neighbour's label replaced by its class probabilities q_j), and the M-step
  refits each class's normal from the tracks' grid posteriors under it.
- beta's M-step is exact: the expected complete-data log-likelihood is beta E_q[S] - log Z(beta), maximized where
  E_beta[S] = E_q[S], the prior's expected agreement equal to the posterior's. E_beta[S] depends only on the graph
  and C, so it is tabulated once by Swendsen-Wang (`agreement_table`) and each M-step is a lookup (`beta_for`).
  log Z(beta) is what holds beta back: a surrogate without it (pseudo-likelihood and its kin) lets beta run past
  the Potts phase transition, where every track takes its neighbours' class and the data are outvoted.
- The fit with C classes starts from the plain mixture and from the C - 1 fit with each class split in two, and
  keeps the best neighbour-predictive score (`PottsFit.score`): the plain mixture's equal weights alone can split
  a large class and merge two small ones.
- Each of those starts beta at 0 and at `ordered_beta`, inside the prior's ordered phase. When each track's data
  are weak, EM has two stable solutions, one near beta = 0 and one ordered, and a start at 0 stays in the first
  even when the pattern is real. The ordered start reaches the second, and the score picks between them.
- Whether there is a pattern is tested by `workflow` (the neighbour gain against shuffled positions).
- The class densities only assign tracks. What a class's tracks look like is read from them as a group with the
  population models (`gridpost.deconvolve`, `fbmpost.deconvolve_tracks`), never from p_c.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MIN_SD = .1  # added in quadrature to a class's sd in ln D, so a class never collapses onto one grid cell
SWEEPS = 5  # mean-field sweeps per E-step
BETA_GRID = np.linspace(0., 1.5, 121)  # beta per edge; on kNN graphs the ordered phase starts well below 1
TABLE_BURN, TABLE_SWEEPS = 50, 100  # Swendsen-Wang sweeps per grid point: discarded, then averaged
ORDERED_AGREEMENT = .9  # the prior agreement at which `ordered_beta` starts beta: past the transition
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


def graph(xy: np.ndarray, k: int) -> csr_matrix:
    """(n, n) 0/1 adjacency of the symmetric k-nearest-neighbour graph: i ~ j if either lists the other."""
    nb = knn(xy, k)
    n = len(nb)
    A = csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), nb.ravel())), shape=(n, n))
    return csr_matrix(((A + A.T) > 0).astype(float))


def densities(u: np.ndarray, centre: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(len(u), C) normal densities on the grid, each column summing to 1."""
    z = (u[:, None] - centre[None, :]) / sd[None, :]
    p = np.exp(-.5 * z ** 2)
    return p / p.sum(axis=0, keepdims=True)


def _softmax(lp: np.ndarray) -> np.ndarray:
    p = np.exp(lp - lp.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def conditional(neighbour_counts: np.ndarray, beta: float) -> np.ndarray:
    """P(class | neighbours) from the (expected) number of neighbours in each class (..., C): the Potts
    conditional."""
    return _softmax(beta * neighbour_counts)


def agreement_table(adjacency: csr_matrix, n_class: int, seed: int = 0) -> np.ndarray:
    """E_beta[S] / (number of edges) on `BETA_GRID`: the Potts prior's expected fraction of agreeing edges, with no
    data, by Swendsen-Wang (which, unlike single-site updates, keeps mixing in the ordered phase). One chain is
    carried up the grid. Made non-decreasing, as the exact curve is."""
    rng = np.random.default_rng(seed)
    n = adjacency.shape[0]
    e = triu(adjacency, 1).tocoo()
    i, j = e.row, e.col
    z = rng.integers(n_class, size=n)
    out = np.empty(len(BETA_GRID))
    for g, b in enumerate(BETA_GRID):
        keep = 1 - np.exp(-b)
        total = 0.
        for s in range(TABLE_BURN + TABLE_SWEEPS):
            same = z[i] == z[j]
            if s >= TABLE_BURN:
                total += same.mean()
            bond = same & (rng.random(len(i)) < keep)
            bonds = csr_matrix((np.ones(int(bond.sum())), (i[bond], j[bond])), shape=(n, n))
            n_clusters, cluster = connected_components(bonds, directed=False)
            z = rng.integers(n_class, size=n_clusters)[cluster]
        out[g] = total / TABLE_SWEEPS
    return np.maximum.accumulate(out)


def beta_for(agreement: float, table: np.ndarray) -> float:
    """beta's M-step: the beta whose prior expected agreement is `agreement`, by interpolating `table`; 0 when the
    posterior agrees no more than unrelated labels do, the grid's top when it agrees more than the table reaches."""
    if agreement <= table[0]:
        return 0.
    return float(np.interp(agreement, table, BETA_GRID))


def ordered_beta(table: np.ndarray) -> float:
    """The beta at which the prior's expected agreement reaches `ORDERED_AGREEMENT`: a start for beta inside the
    ordered phase, just past the graph's transition."""
    return beta_for(ORDERED_AGREEMENT, table)


def edge_agreement(q: np.ndarray, adjacency: csr_matrix) -> float:
    """E_q[S] / (number of edges) under the mean field: the mean over edges of sum_c q_ic q_jc."""
    e = triu(adjacency, 1).tocoo()
    return float(np.mean(np.sum(q[e.row] * q[e.col], axis=1)))


@dataclass(frozen=True)
class PottsFit:
    """One fit: classes in increasing order of D. `q` (n, C) are the tracks' class probabilities given their own
    data and their neighbours; `evidence` (n, C) is e_ic relative to each track's likelihood maximum; `agreement`
    is the fraction of edges the fit expects to agree, which beta matches in the prior."""
    centre: np.ndarray  # (C,) class means in ln D
    sd: np.ndarray      # (C,) class sds in ln D
    beta: float         # per edge
    agreement: float
    q: np.ndarray
    evidence: np.ndarray
    n_iter: int
    converged: bool

    @property
    def n_classes(self) -> int:
        return len(self.centre)

    def neighbour_q(self, adjacency: csr_matrix) -> np.ndarray:
        """(n, C) each track's class probabilities from its neighbours alone: its own data left out."""
        return conditional(adjacency @ self.q, self.beta)

    def score(self, adjacency: csr_matrix, offset: np.ndarray) -> float:
        """Neighbour-predictive log-likelihood: each track's data under the classes its neighbours predict.
        `offset` is each row's log-likelihood maximum, so scores of different C compare."""
        p = self.neighbour_q(adjacency)
        return float(np.sum(np.log((p * self.evidence).sum(axis=1) + TINY) + offset))

    def track_posteriors(self, lik: np.ndarray, u: np.ndarray) -> np.ndarray:
        """(n, len(u)) each track's posterior over ln D with the class mixture as its prior, rows summing to 1."""
        p = densities(u, self.centre, self.sd)
        post = lik * ((self.q / (self.evidence + TINY)) @ p.T)
        return post / post.sum(axis=1, keepdims=True)


def initial(lik: np.ndarray, u: np.ndarray, n_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Classes spread over the quantiles of the tracks' most likely ln D, each as wide as a third of that range."""
    peak = u[lik.argmax(axis=1)]
    centre = np.quantile(peak, np.linspace(.1, .9, n_class)) if n_class > 1 else np.array([np.median(peak)])
    width = max(float(np.subtract(*np.quantile(peak, [.9, .1]))) / 3, MIN_SD)
    return centre, np.full(n_class, width)


def splits(f: PottsFit) -> list[tuple[np.ndarray, np.ndarray]]:
    """Starts for C + 1 classes from a C-class fit: each class in turn split into two half a width either side of
    its centre (the split move of split-and-merge EM)."""
    out = []
    for j in range(f.n_classes):
        centre = np.concatenate([np.delete(f.centre, j), f.centre[j] + np.array([-.5, .5]) * f.sd[j]])
        sd = np.concatenate([np.delete(f.sd, j), np.full(2, f.sd[j] / np.sqrt(2))])
        out.append((centre, sd))
    return out


def fit(lik: np.ndarray, u: np.ndarray, adjacency: csr_matrix, n_class: int, table: np.ndarray | None = None,
        beta: float | None = None, init: PottsFit | tuple | None = None, beta0: float = 0., max_iter: int = 300,
        tol: float = 1e-3) -> PottsFit:
    """Fit `n_class` classes to `lik` (n, len(u)): each track's likelihood on the grid, maximum 1.

    `beta` None learns it by the exact M-step, which needs `agreement_table(adjacency, n_class)` as `table`; a
    number fixes it (0: the plain mixture). `init` is an earlier fit or a (centre, sd) tuple; by default
    `initial`. A learned beta starts from `beta0` (0, or `ordered_beta(table)` for the ordered solution), never
    from `init`'s, so a fit and its shuffles follow the same path.
    """
    if init is None:
        centre, sd = initial(lik, u, n_class)
    elif isinstance(init, PottsFit):
        centre, sd = init.centre.copy(), init.sd.copy()
    else:
        centre, sd = (np.asarray(a, float).copy() for a in init)
    if len(centre) != n_class:
        raise ValueError(f"init has {len(centre)} classes, asked for {n_class}")
    learn = beta is None and n_class > 1
    if learn and table is None:
        raise ValueError("learning beta needs the graph's agreement table")
    b = (float(beta0) if learn else 0.) if beta is None else float(beta)
    q = np.full((len(lik), n_class), 1 / n_class)
    converged = False
    for it in range(1, max_iter + 1):
        q_old, b_old = q, b
        p = densities(u, centre, sd)
        ev = lik @ p + TINY
        log_ev = np.log(ev)
        for _ in range(SWEEPS):
            q = _softmax(log_ev + b * (adjacency @ q))
        # Each class's grid mass: sum_i q_ic L_i(u) p_c(u) / e_ic.
        mass = (lik.T @ (q / ev)) * p
        total = mass.sum(axis=0) + TINY
        centre = u @ mass / total
        var = (u ** 2) @ mass / total - centre ** 2
        sd = np.sqrt(np.maximum(var, 0.) + MIN_SD ** 2)
        if learn:
            b = beta_for(edge_agreement(q, adjacency), table)
        if it > 10 and np.abs(q - q_old).max() < tol and abs(b - b_old) < tol:
            converged = True
            break
    p = densities(u, centre, sd)
    ev = lik @ p
    order = np.argsort(centre)
    q = q[:, order]
    return PottsFit(centre[order], sd[order], b, edge_agreement(q, adjacency), q, ev[:, order], it, converged)
