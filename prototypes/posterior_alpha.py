"""Per-track posterior P(alpha | displacements) for anomalous diffusion. Standalone prototype.

Nothing here imports diffusionkit. Only numpy and scipy; matplotlib for the demo.

This generalizes posterior_1d.py's grid posterior over D to a grid posterior
over the fBm exponent alpha, with the generalized diffusion coefficient K
as a nuisance parameter marginalized out -- the same honesty posterior_1d
gives D (a short, uninformative track reports a wide posterior, not a
falsely confident point), applied to the question "is this motion
Brownian?" instead of the calibrated-z-score hypothesis test
diffusionkit.classic.likelihood's non-Brownian score answers.

Model, per axis. A track of n frames gives m = n - 1 displacements d, and

    d ~ N(0, K A(alpha) + B)

- A(alpha) (per unit K): the fractional-Gaussian-noise covariance of fBm
  increments, dt^alpha * (|k+1|^alpha - 2|k|^alpha + |k-1|^alpha) at lag k.
  At alpha=1 this reduces exactly to 2 dt at lag 0 and 0 beyond -- ordinary
  Brownian motion with independent increments, matching posterior_1d.py's
  `motion_cov(..., on_time=0)`.
- B: the known per-frame localization variances (same as posterior_1d.py).

No exposure-blur model: posterior_1d.py's closed-form Berglund R average is
specific to alpha=1 Brownian motion (the double integral of a *linear*
motion variance has a simple closed form); no comparably simple closed form
exists for exposure-averaged fBm at a general alpha (it would need the same
kind of numerical quadrature diffusionkit.classic used to drop). This
prototype therefore assumes exposure_s=0, matching the existing NumPyro
anomalous model (diffusionkit.bayes.model), which makes the same
simplification for the same reason.

For any *fixed* alpha, the covariance is linear in K exactly as
posterior_1d's is linear in D, so the same generalized-eigendecomposition
whitening trick applies unchanged: one eigendecomposition of A(alpha) per
grid point, then O(m) per (alpha, K) grid point. The 2D log-likelihood
surface over (alpha, ln K) is then reduced to a 1D posterior over alpha by
marginalizing out ln K (the nuisance parameter) with its own prior, exactly
mirroring how classic/posterior.py's empirical-Bayes prior integrates over a
nuisance-like nuisance grid -- here the K prior is hand-set, not fit.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp

# alpha grid: avoid the exact 0/2 edges, where A(alpha) degenerates.
ALPHA = np.linspace(0.05, 1.95, 39)
# ln K grid, K in um^2/s^alpha: 1e-4 to 10, matching posterior_1d.py's D grid.
U = np.linspace(np.log(1e-4), np.log(10.0), 251)


# --------------------------------------------------------------------------
# Likelihood
# --------------------------------------------------------------------------


def motion_cov(m: int, dt: float, alpha: float) -> np.ndarray:
    """(m, m) fGn covariance of m consecutive displacements per unit K, no blur.

    gamma(k) = dt^alpha (|k+1|^alpha - 2|k|^alpha + |k-1|^alpha); alpha=1
    gives gamma(0) = 2 dt, gamma(k>=1) = 0 (ordinary Brownian motion).
    """
    k = np.arange(m, dtype=float)
    gamma = dt**alpha * (np.abs(k + 1) ** alpha - 2 * np.abs(k) ** alpha + np.abs(k - 1) ** alpha)
    i = np.arange(m)
    return gamma[np.abs(i[:, None] - i[None, :])]


def localization_cov(sd) -> np.ndarray:
    """(m, m) displacement covariance from n = m + 1 per-frame position SDs."""
    v = np.asarray(sd, float) ** 2
    B = np.diag(v[:-1] + v[1:])
    i = np.arange(len(v) - 2)
    B[i, i + 1] = B[i + 1, i] = -v[1:-1]
    return B


class Whitened(NamedTuple):
    lam: np.ndarray  # (m,) variance of component k is 1 + K lam_k
    y: np.ndarray  # (m,)
    const: float  # log-likelihood terms that do not depend on K


def whiten(disp, sd, dt: float, alpha: float) -> Whitened:
    """One axis at a fixed alpha: displacements (m,), per-frame position SDs (m + 1,)."""
    disp = np.asarray(disp, float)
    if len(sd) != len(disp) + 1:
        raise ValueError("need one localization SD per frame")
    if np.any(np.asarray(sd) <= 0):
        raise ValueError("localization SDs must be positive")
    B = localization_cov(sd)
    lam, V = eigh(motion_cov(len(disp), dt, alpha), B)
    return Whitened(
        np.maximum(lam, 0.0),
        V.T @ disp,
        -0.5 * (len(disp) * np.log(2 * np.pi) + 2 * np.log(np.diag(np.linalg.cholesky(B))).sum()),
    )


def loglik(axes: list[Whitened], u: np.ndarray = U) -> np.ndarray:
    """(len(u),) log-likelihood of the whitened axes at K = exp(u), fixed alpha."""
    total = 0.0
    for lam, y, const in axes:
        d = 1 + np.exp(u)[:, None] * lam
        total = total + const - 0.5 * np.sum(np.log(d) + y**2 / d, axis=1)
    return total


def track_loglik_given_alpha(positions, sd, dt: float, alpha: float, u: np.ndarray = U) -> np.ndarray:
    """positions (n,) or (n, axes); sd (n,) or (n, axes) -- (len(u),) log-likelihood over K."""
    x = np.asarray(positions, float)
    x = x[:, None] if x.ndim == 1 else x
    s = np.broadcast_to(np.reshape(np.asarray(sd, float), (len(x), -1)), x.shape)
    return loglik([whiten(np.diff(x[:, a]), s[:, a], dt, alpha) for a in range(x.shape[1])], u)


def joint_loglik(positions, sd, dt: float, alphas: np.ndarray = ALPHA, u: np.ndarray = U) -> np.ndarray:
    """(len(alphas), len(u)) log-likelihood surface over (alpha, ln K)."""
    return np.array([track_loglik_given_alpha(positions, sd, dt, a, u) for a in alphas])


# --------------------------------------------------------------------------
# Priors: log-density on the grid, up to a constant
# --------------------------------------------------------------------------


def log_uniform_K(K_lo: float, K_hi: float, u: np.ndarray = U) -> np.ndarray:
    """Flat in ln K between the limits (the nuisance prior to integrate out)."""
    return np.where((u >= np.log(K_lo)) & (u <= np.log(K_hi)), 0.0, -np.inf)


def flat_alpha(alphas: np.ndarray = ALPHA) -> np.ndarray:
    """Flat over the alpha grid -- the least-informative default."""
    return np.zeros_like(alphas)


# --------------------------------------------------------------------------
# Simulation (independent of the likelihood: exact fGn draw, no exposure blur)
# --------------------------------------------------------------------------


def simulate(K: float, alpha: float, sd, dt: float, rng: np.random.Generator, axes: int = 2) -> np.ndarray:
    """(n, axes) measured positions: exact fGn increments (Cholesky), plus
    Gaussian localization error with SD sd[i]."""
    sd = np.asarray(sd, float)
    n = len(sd)
    L = np.linalg.cholesky(K * motion_cov(n - 1, dt, alpha))
    disp = np.stack([L @ rng.standard_normal(n - 1) for _ in range(axes)], axis=1)
    true = np.vstack([np.zeros(axes), np.cumsum(disp, axis=0)])
    return true + sd[:, None] * rng.standard_normal((n, axes))


# --------------------------------------------------------------------------
# Posterior over alpha: marginalize the nuisance ln K
# --------------------------------------------------------------------------


def alpha_posterior(joint_ll: np.ndarray, log_K_prior: np.ndarray,
                    log_alpha_prior: np.ndarray | None = None, u: np.ndarray = U) -> np.ndarray:
    """(len(alphas),) posterior over alpha: sum joint_ll + priors, integrate ln K, normalize.

    `joint_ll` is (len(alphas), len(u)) from `joint_loglik`. Integrating a
    nuisance parameter out is exactly `logsumexp` over its axis (a Riemann
    sum in ln K, grid step folded into the additive constant that
    normalization below removes).
    """
    lp = joint_ll + log_K_prior[None, :]
    log_mass = logsumexp(lp, axis=1)
    if log_alpha_prior is not None:
        log_mass = log_mass + log_alpha_prior
    p = np.exp(log_mass - log_mass.max())
    return p / p.sum()


def quantile(p: np.ndarray, q: float, alphas: np.ndarray = ALPHA) -> float:
    """Posterior q-quantile of alpha, interpolating the CDF at cell midpoints."""
    return float(np.interp(q, np.cumsum(p) - p / 2, alphas))


def summary(p: np.ndarray, alphas: np.ndarray = ALPHA, level: float = 0.9) -> dict[str, float]:
    """Median and equal-tailed credible interval of alpha."""
    return {
        "median": quantile(p, 0.5, alphas),
        "lo": quantile(p, (1 - level) / 2, alphas),
        "hi": quantile(p, (1 + level) / 2, alphas),
    }


def track_alpha_posterior(positions, sd, dt: float, log_K_prior: np.ndarray,
                          alphas: np.ndarray = ALPHA, u: np.ndarray = U) -> np.ndarray:
    """(len(alphas),) posterior over alpha for one track, K marginalized under `log_K_prior`."""
    return alpha_posterior(joint_loglik(positions, sd, dt, alphas, u), log_K_prior, u=u)
