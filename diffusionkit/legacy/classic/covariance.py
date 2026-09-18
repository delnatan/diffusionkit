"""Analytic covariance of the TAMSD curve, for GLS fitting instead of
ad-hoc lag truncation (see `fitting.py`'s `*_gls` functions).

`step_covariance` is a plain-numpy port of `bayes/likelihood.py`'s
`fgn_gamma`/`noise_covariance` -- same physics (fGn autocovariance for the
true motion + tridiagonal iid-localization-noise term, isotropic, R=0), and
the same formula written out in README's `Sigma_motion`/`Sigma_noise`
block. Deliberately duplicated rather than imported: `bayes/likelihood.py`
is jax, and `classic/` stays off the jax import path (see the package's
staged import graph -- matplotlib/jax only load when something actually
needs them, and nothing in `classic/` should pull jax in as a side effect
of importing `fitting.py`).

`tamsd_covariance` turns that per-step covariance into the covariance of
the *TAMSD estimator* at a chosen set of lags, via Isserlis'/Wick's theorem
applied to the single-track step process (Qian, Sheetz & Elson 1991;
same correlation structure Michalet & Berglund 2012 warn is being ignored
by naive OLS-on-truncated-window fits).
"""

from __future__ import annotations

import numpy as np


def step_covariance(
    n_disp: int, K: float, dt_s: float, alpha: float, sigma2_um2: float
) -> np.ndarray:
    """Per-axis (n_disp x n_disp) covariance of observed step displacements.

    True motion (fBm) + iid static localization noise, independent so they
    add. At alpha=1 the motion term is exactly diagonal (2*K*dt on the
    diagonal, 0 elsewhere) -- ordinary Brownian motion has independent
    increments, so any off-diagonal correlation in the alpha=1 case comes
    purely from the noise term.
    """
    idx = np.arange(n_disp)
    lag = np.abs(idx[:, None] - idx[None, :]).astype(float)

    motion = K * dt_s**alpha * (
        np.abs(lag + 1) ** alpha - 2.0 * lag**alpha + np.abs(lag - 1) ** alpha
    )
    noise = np.where(
        lag == 0, 2.0 * sigma2_um2, np.where(lag == 1, -sigma2_um2, 0.0)
    )
    return motion + noise


def tamsd_covariance(step_cov: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Covariance matrix of the total-2D TAMSD estimator at `lags`.

    rho(n) = (1/M_n) * sum_i [ (x[i+n]-x[i])^2 + (y[i+n]-y[i])^2 ],
    M_n = n_disp - n + 1 pairs at lag n.

    For zero-mean jointly Gaussian A, B: Cov(A^2, B^2) = 2*Cov(A,B)^2
    (Isserlis'/Wick's theorem). Summing over two independent, identically
    distributed axes (isotropic, x/y share `step_cov`):

        Cov(rho(n), rho(m)) = (4 / (M_n*M_m)) * sum_{i,j} C_ij^2

    where C_ij is the covariance between the length-n window sum starting
    at step i and the length-m window sum starting at step j, i.e. the
    block-sum of `step_cov` over rows [i, i+n) and columns [j, j+m).

    A direct (M_n x M_m) block-sum per (n, m) pair is O(n_disp^2) per pair
    and too slow once `lags` spans a whole long track (no fixed cap here,
    unlike `n_fit_points` -- see `fitting.n_gls_fit_points`). Since
    `step_cov` is Toeplitz (gamma(i-j), stationary), C_ij depends only on
    the offset `delta = j-i`, not on i, j individually: standard telescoping
    (two more cumulative sums, of the 1D gamma sequence instead of the 2D
    matrix) collapses each pair's cost to O(M_n + M_m). Verified against
    the direct O(n_disp^2) block-sum computation and against Monte Carlo
    simulation (see `tests/` or the module's development notes) before
    replacing it.
    """
    n_disp = step_cov.shape[0]
    lags = np.asarray(lags, dtype=int)
    L = len(lags)
    M = n_disp - lags + 1

    # gamma(k) for k=0..n_disp-1 from step_cov's Toeplitz first row;
    # gamma(-k)=gamma(k). Padded with zeros out to +-2*n_disp: those slots
    # are algebraic scaffolding for the telescoping sums below (G, P2) and
    # never correspond to an actual pair of valid step indices -- every
    # P2 difference this function takes is provably confined to the true
    # +-(n_disp-1) support, so the padding value is arbitrary (zero is
    # simplest) and doesn't affect the result.
    base = -2 * n_disp - 2
    gk = np.zeros(2 * (2 * n_disp + 2) + 1)
    k0 = -base  # array index of k=0
    k = np.arange(n_disp)
    gamma_row = step_cov[0, :]
    gk[k0 + k] = gamma_row
    gk[k0 - k] = gamma_row

    G = np.concatenate(([0.0], np.cumsum(gk)))  # G(x) at index x-base
    P2 = np.concatenate(([0.0], np.cumsum(G)))  # P2(x) at index x-base

    def P2_at(x: np.ndarray) -> np.ndarray:
        return P2[x - base]

    cov = np.empty((L, L))
    for a in range(L):
        n = int(lags[a])
        M_n = int(M[a])
        for b in range(a, L):
            m = int(lags[b])
            M_m = int(M[b])
            delta = np.arange(-(M_n - 1), M_m)  # all overlapping offsets
            h = (
                P2_at(n - delta + 1)
                - P2_at(-delta + 1)
                - P2_at(n - delta - m + 1)
                + P2_at(-delta - m + 1)
            )
            count = np.minimum(M_n, M_m - delta) - np.maximum(0, -delta)
            val = 4.0 * np.sum(h**2 * count) / (M_n * M_m)
            cov[a, b] = val
            cov[b, a] = val
    return cov
