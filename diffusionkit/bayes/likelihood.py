"""The physics: displacement covariance under fBm + static localization noise.

This module builds the covariance matrix that `model.py` plugs into
`numpyro.distributions.MultivariateNormal` -- it is the only place the
actual diffusion model lives; everything else (priors, inference, sampling)
is generic machinery on top. See `model.py`'s docstring for the physics
references (Michalet & Berglund 2012, Vestergaard et al. 2014, Kepten et al.
2013) and the full derivation.

Model (per spatial dimension, isotropic 2D motion so x and y share the same
K, alpha, sigma):

  True motion:        fBm with MSD_1D(t) = 2*K*t^alpha
                       (alpha=1 reduces exactly to ordinary Brownian motion.
                       Total 2D MSD = 4*K*tau^alpha, matching the
                       convention in analysis/fitting.py.)
  Observed position:  x_obs[n] = mean of X_true over frame n's exposure + eps[n],
                       eps[n] ~ iid N(0, sigma^2) (static localization noise).
                       The exposure average is the box-shutter blur the grid
                       posteriors model (`gridpost.likelihood.fgn_motion_covariance`,
                       Berglund 2010 at alpha=1); exposure_s=0 is instantaneous.

Differentiable in K and alpha (NUTS needs the gradient); the time lags are
concrete integers, so the blur formula's branches are chosen with numpy
rather than traced. Pure jax.numpy otherwise: no numpyro import here, so this module (and its covariance
formula) is equally usable from `simulate.py`'s plain-numpy ground-truth
generator with a single `np.asarray(...)` conversion at the boundary --
generative model and inference model share one implementation, not two.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..gridpost.likelihood import _BLUR_SERIES_RATIO, _BLUR_SERIES_TERMS


def _safe_pow(y: np.ndarray, p):
    """y**p for concrete y >= 0 and a (possibly traced) exponent p, with 0**p = 0.

    jax's own `0.0 ** p` has a NaN gradient in p (it differentiates through
    log 0); the blur formula meets y = 0 at lag 0, and at lag 1 when the
    exposure fills the frame. Both `where` branches stay finite here.
    """
    y = np.asarray(y, dtype=float)
    positive = y > 0
    return jnp.where(positive, jnp.exp(p * np.log(np.where(positive, y, 1.0))), 0.0)


def _blurred_abs_power(x: np.ndarray, alpha, exposure_s: float):
    """E|x + u - v|^alpha for u, v ~ Uniform(0, exposure_s), elementwise in concrete x.

    The same closed form and series switch as
    `gridpost.likelihood._blurred_abs_power` (see there), in jax so that it
    differentiates in alpha; alpha may be an array that broadcasts against x.
    """
    x = np.abs(np.asarray(x, dtype=float))
    if exposure_s == 0:
        return _safe_pow(x, alpha)
    te = float(exposure_s)
    series = x * _BLUR_SERIES_RATIO > te
    x_series = np.where(series, x, 1.0)
    r2 = np.where(series, (te / x_series) ** 2, 0.0)
    total, binom, power = 1.0, 1.0, np.ones_like(r2)
    for k in range(1, _BLUR_SERIES_TERMS + 1):
        n = 2 * k
        binom = binom * (alpha - n + 2) * (alpha - n + 1) / ((n - 1) * n)  # C(alpha, n)
        power = power * r2
        total = total + binom * (2 / ((n + 1) * (n + 2))) * power
    exact = (_safe_pow(x + te, alpha + 2) - 2 * _safe_pow(x, alpha + 2) + _safe_pow(np.abs(x - te), alpha + 2)) / (
        (alpha + 1) * (alpha + 2) * te**2)
    return jnp.where(series, _safe_pow(x_series, alpha) * total, exact)


def fgn_gamma(lag, K: float, dt_s: float, alpha: float, exposure_s: float = 0.0) -> jnp.ndarray:
    """Autocovariance of (exposure-averaged) fBm increments at concrete integer lags.

    gamma(k) = K * (G((k+1) dt) - 2 G(k dt) + G((k-1) dt)),  G(x) = E|x + u - v|^alpha

    with u, v uniform over the exposure (`_blurred_abs_power`). exposure_s=0
    gives fGn, K * dt^alpha * (|k+1|^alpha - 2|k|^alpha + |k-1|^alpha): then
    gamma(0) = 2*K*dt^alpha = Var(single increment) = MSD_1D(dt), and at
    alpha=1, gamma(0) = 2*D*dt and gamma(k>=1) = 0 -- ordinary Brownian
    motion has independent increments. With blur, alpha=1 gives Berglund's
    2 D dt (1 - 2R) and lag-1 2 D dt R, R = exposure_s / (6 dt).
    """
    k = np.abs(np.asarray(lag, dtype=float))
    G = lambda lags: _blurred_abs_power(lags * dt_s, alpha, exposure_s)  # noqa: E731
    return K * (G(k + 1) - 2 * G(k) + G(k - 1))


def fgn_covariance(n_disp: int, K: float, dt_s: float, alpha: float, exposure_s: float = 0.0) -> jnp.ndarray:
    """Dense (n_disp x n_disp) Toeplitz covariance matrix of (exposure-averaged) fGn increments.

    n_disp is the number of displacements in a track (= track_length - 1).
    Built explicitly rather than via a fast Toeplitz algorithm: track lengths
    here are <=200, so a dense (n_disp)^2 matrix is trivial, and staying
    dense keeps this identical in shape to `noise_covariance` (summed in
    `displacement_covariance`).
    """
    idx = np.arange(n_disp)
    lag = np.abs(idx[:, None] - idx[None, :])
    return fgn_gamma(lag, K, dt_s, alpha, exposure_s)


def noise_covariance(n_disp: int, sigma2_um2: float) -> jnp.ndarray:
    """Tridiagonal covariance contributed by iid static localization noise.

    For x_obs[n] = X_true[n] + eps[n] with eps iid N(0, sigma^2), the
    displacement noise term eps[n]-eps[n-1] has Var = 2*sigma^2 and is
    correlated with its immediate neighbor (sharing one eps term) at
    Cov = -sigma^2; non-adjacent displacements share no eps term, so
    Cov = 0. Independent of D/alpha, additive with `fgn_covariance` since
    the true-motion and noise processes are independent.
    """
    idx = jnp.arange(n_disp)
    lag = jnp.abs(idx[:, None] - idx[None, :])
    return jnp.where(lag == 0, 2.0 * sigma2_um2, jnp.where(lag == 1, -sigma2_um2, 0.0))


def displacement_covariance(
    n_disp: int, K: float, dt_s: float, alpha: float, sigma2_um2: float, exposure_s: float = 0.0
) -> jnp.ndarray:
    """Full covariance of observed displacements: true motion (exposure-averaged fGn) + localization noise."""
    return fgn_covariance(n_disp, K, dt_s, alpha, exposure_s) + noise_covariance(n_disp, sigma2_um2)
