"""Maximum-entropy distribution of D across tracks, alpha chosen by evidence.

Companion to `posterior_1d.deconvolve` (its unregularized-EM / Gaussian-smoothed
sibling). Nothing here imports diffusionkit; only numpy, scipy, and
`posterior_1d` for the grid and its helpers.

`deconvolve` maximizes `sum_i log(sum_D L_i(D) g(D))` over grid weights g -- the
nonparametric MLE of a mixing distribution. That objective is merely concave
(not strictly), so its unregularized maximum can collapse mass onto a handful
of grid points; `deconvolve` guards against this with an ad-hoc Gaussian blur
each EM step, whose width is a hand-tuned constant, not something the data
chooses.

This module regularizes properly instead: add `alpha * S(w)`, the cross-entropy
of w against a default measure m, to the objective. For any alpha > 0 this is
*strictly* concave (S's Hessian is `-diag(1/w)`, negative definite), so its
maximizer w_alpha is unique and cannot be a spike. `alpha` is then chosen the
way Gull & Skilling (1989, "Developments in Maximum Entropy Data Analysis")
choose it: by maximizing the Laplace-approximated evidence `P(data | alpha)`,
rather than tuning against a known truth. Because w lives on a ~500-point 1D
grid (not a multi-megapixel image), the evidence's eigenvalue problem is a
single dense ~500x500 generalized eigendecomposition -- no need for the
Skilling-Bryan control-subspace machinery built for much larger reconstructions.

Model, in brief. w is parametrized as softmax(theta), theta in R^K (K = the
number of grid cells where the prior is finite), which makes the constrained
simplex optimization an unconstrained smooth one; this does not distort the
entropy-regularization geometry the evidence formula depends on (the softmax
Jacobian's correction term to the Hessian vanishes exactly at the constrained
optimum, and generalized eigenvalues restricted to the sum-zero tangent
hyperplane don't depend on which basis of it is used). At the optimum w_alpha:

    A_kl = -d^2(log L)/dw_k dw_l    (data curvature, PSD, rank <= n_tracks)
    g_kl = -d^2 S/dw_k dw_l = delta_kl / w_alpha[k]    (entropy metric)

Both are projected onto the sum-zero tangent hyperplane and their generalized
eigenvalues lambda_k solved (`A_tan v = lambda g_tan v`). The evidence is then

    log P(data|alpha) ~= alpha*S(w_alpha) + log L(w_alpha)
                          - 1/2 sum_k log(1 + lambda_k/alpha)

v1 scans a geometric alpha grid (built from the data's own curvature scale) and
picks the grid argmax -- no continuous refinement yet.
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple
from warnings import warn

import numpy as np
from scipy.linalg import eigh, null_space
from scipy.optimize import minimize

import posterior_1d as P1D


class MaxEntFit(NamedTuple):
    w: np.ndarray              # (len(u),) grid weights, zero outside the prior's support, sum to 1
    alpha: float                 # evidence-chosen regularization weight
    log_evidence: float          # log P(data|alpha), up to an alpha-independent additive constant
    alpha_grid: np.ndarray       # (n_alpha,) the coarse scan grid actually used
    evidence_curve: np.ndarray   # (n_alpha,) log P(data|alpha) on alpha_grid, same constant as log_evidence
    eigenvalues: np.ndarray      # (K - 1,) generalized eigenvalues of (A, g) at alpha, ascending
    n_good: float                 # sum lambda_k / (alpha + lambda_k): effective number of resolved components


# --------------------------------------------------------------------------
# Objective: mixture log-likelihood + alpha * cross-entropy
# --------------------------------------------------------------------------


def default_measure(K: int) -> np.ndarray:
    """The flat default measure: 1/K on each of the K active grid cells."""
    return np.full(K, 1.0 / K)


def mixture_loglik(L: np.ndarray, w: np.ndarray) -> float:
    """sum_i log((L @ w)_i); L is (n_tracks, K) row-max-rescaled likelihood, w is (K,) on the simplex."""
    return float(np.sum(np.log(np.maximum(L @ w, 1e-300))))


def entropy(w: np.ndarray, m: np.ndarray) -> float:
    """-sum_k w_k log(w_k / m_k): the cross-entropy regularizer S(w) (plain Shannon entropy if m is flat).

    Floors w inside the log only, so an underflowed-to-exact-zero w_k contributes its
    correct limit (0 * log(0) -> 0), not a NaN from 0 * -inf.
    """
    w_safe = np.maximum(w, 1e-300)
    return float(-np.sum(w * np.log(w_safe / m)))


def _softmax(theta: np.ndarray) -> np.ndarray:
    t = theta - theta.max()
    e = np.exp(t)
    return e / e.sum()


def _neg_objective_and_grad(theta: np.ndarray, L: np.ndarray, m: np.ndarray, alpha: float):
    """-(log L(w) + alpha*S(w)) and its theta-gradient, w = softmax(theta)."""
    w = _softmax(theta)
    r = np.maximum(L @ w, 1e-300)
    logL = np.sum(np.log(r))
    S = entropy(w, m)
    F = logL + alpha * S

    w_safe = np.maximum(w, 1e-300)
    dlogL_dw = L.T @ (1.0 / r)
    dS_dw = -(np.log(w_safe / m) + 1.0)  # floored: an underflowed w contributes 0 via w_j * dF/dtheta_j below
    dF_dw = dlogL_dw + alpha * dS_dw
    dF_dtheta = w * (dF_dw - w @ dF_dw)  # softmax vector-Jacobian product
    return -F, -dF_dtheta


# --------------------------------------------------------------------------
# Inner solve at fixed alpha
# --------------------------------------------------------------------------


def fit_maxent(
    L: np.ndarray,
    m: np.ndarray,
    alpha: float,
    theta0: np.ndarray | None = None,
    gtol: float = 1e-8,
    maxiter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximize log L(w) + alpha*S(w) over the simplex via softmax(theta) + L-BFGS-B.

    Returns (w, theta) at the optimum; theta is a warm-start for a neighboring alpha.
    """
    if theta0 is None:
        theta0 = np.log(m)
    res = minimize(
        _neg_objective_and_grad,
        theta0,
        args=(L, m, alpha),
        jac=True,
        method="L-BFGS-B",
        options={"gtol": gtol, "maxiter": maxiter},
    )
    return _softmax(res.x), res.x


# --------------------------------------------------------------------------
# Evidence: Laplace approximation to P(data | alpha)
# --------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _tangent_basis(K: int) -> np.ndarray:
    """Orthonormal basis (K, K - 1) of the sum-zero hyperplane in R^K. Depends only on K, so cached."""
    return null_space(np.ones((1, K)))


def _generalized_eigs(L: np.ndarray, w: np.ndarray, w_floor: float = 1e-12) -> np.ndarray:
    """Generalized eigenvalues lambda_k of (A_tan, g_tan), ascending, clipped at 0.

    A = data curvature (-Hessian of log L) at w; g = diag(1/w), the entropy metric.
    Both restricted to the sum-zero tangent hyperplane before solving.
    """
    K = L.shape[1]
    r = np.maximum(L @ w, 1e-300)
    R = L / r[:, None]
    A = R.T @ R
    g_diag = 1.0 / np.maximum(w, w_floor)
    Pn = _tangent_basis(K)
    A_tan = Pn.T @ A @ Pn
    g_tan = Pn.T @ (g_diag[:, None] * Pn)
    eigvals = eigh(A_tan, g_tan, eigvals_only=True)
    return np.clip(eigvals, 0.0, None)


def evidence(L: np.ndarray, m: np.ndarray, alpha: float, w: np.ndarray) -> tuple[float, np.ndarray]:
    """Gull & Skilling Laplace evidence at a converged w = w_alpha.

    log P(data|alpha) ~= alpha*S(w) + log L(w) - 1/2 sum_k log(1 + lambda_k/alpha),
    lambda_k the generalized eigenvalues of the data curvature against the entropy
    metric, restricted to the sum-zero tangent hyperplane. See module docstring.
    Returns (log_evidence, eigenvalues).
    """
    eigvals = _generalized_eigs(L, w)
    log_ev = alpha * entropy(w, m) + mixture_loglik(L, w) - 0.5 * np.sum(np.log1p(eigvals / alpha))
    return float(log_ev), eigvals


def _n_good(alpha: float, eigvals: np.ndarray) -> float:
    """Gull's 'number of good measurements': sum lambda_k / (alpha + lambda_k)."""
    return float(np.sum(eigvals / (alpha + eigvals)))


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------


def maxent_deconvolve(
    lls: np.ndarray,
    log_prior: np.ndarray,
    m: np.ndarray | None = None,
    alpha_grid: np.ndarray | None = None,
    u: np.ndarray = P1D.U,
) -> MaxEntFit:
    """Distribution of D across tracks by maximum entropy, alpha chosen by evidence.

    Drop-in alternative to `posterior_1d.deconvolve`'s Gaussian-smoothed EM: maximizes
    log L(w) + alpha*S(w) over the grid simplex (S = cross-entropy against the default
    measure m, flat by default) instead of unregularized NPMLE, and picks alpha by
    maximizing the Gull & Skilling (1989) Laplace evidence over a log(alpha) scan.
    v1: alpha is the argmax over `alpha_grid` -- no continuous refinement yet.

    Only the finite entries of `log_prior` are used, as the grid's support; `m`
    defaults to flat over that support (`default_measure`). Warns if the evidence
    peak sits at a boundary of `alpha_grid` (widen it, or pass an explicit one).
    """
    support = np.isfinite(log_prior)
    K = int(support.sum())
    L = np.exp(lls[:, support] - lls[:, support].max(axis=1, keepdims=True))
    if m is None:
        m = default_measure(K)

    if alpha_grid is None:
        lambda_max0 = _generalized_eigs(L, m)[-1]
        alpha_grid = np.geomspace(lambda_max0 * 1e-4, lambda_max0 * 1e2, 33)
    alpha_grid = np.asarray(alpha_grid, dtype=float)

    n_alpha = len(alpha_grid)
    ws = np.empty((n_alpha, K))
    evidence_curve = np.empty(n_alpha)
    eigs = [None] * n_alpha

    theta = np.log(m)
    for i, alpha in enumerate(alpha_grid):  # ascending alpha: spiky/hard first, flat/easy last
        w, theta = fit_maxent(L, m, alpha, theta0=theta)
        ws[i] = w
        evidence_curve[i], eigs[i] = evidence(L, m, alpha, w)

    best = int(np.argmax(evidence_curve))
    if best in (0, n_alpha - 1):
        warn(
            f"evidence peak at alpha_grid boundary (index {best}/{n_alpha - 1}); "
            "widen alpha_grid to trust alpha_hat",
            stacklevel=2,
        )

    alpha_hat = float(alpha_grid[best])
    eigvals_hat = eigs[best]

    w_full = np.zeros_like(u)
    w_full[support] = ws[best]

    return MaxEntFit(
        w=w_full,
        alpha=alpha_hat,
        log_evidence=float(evidence_curve[best]),
        alpha_grid=alpha_grid,
        evidence_curve=evidence_curve,
        eigenvalues=eigvals_hat,
        n_good=_n_good(alpha_hat, eigvals_hat),
    )
