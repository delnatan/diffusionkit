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
  Observed position:  x_obs[n] = X_true[n] + eps[n],  eps[n] ~ iid N(0, sigma^2)
                       (static localization noise, no motion-blur/R term --
                       same R=0 simplification analysis/fitting.py makes.)

Pure jax.numpy: no numpyro import here, so this module (and its covariance
formula) is equally usable from `simulate.py`'s plain-numpy ground-truth
generator with a single `np.asarray(...)` conversion at the boundary --
generative model and inference model share one implementation, not two.
"""
from __future__ import annotations

import jax.numpy as jnp


def fgn_gamma(lag: jnp.ndarray, K: float, dt_s: float, alpha: float) -> jnp.ndarray:
    """Autocovariance of fractional Gaussian noise (fBm increments) at integer lag.

    gamma(k) = K * dt^alpha * (|k+1|^alpha - 2|k|^alpha + |k-1|^alpha)

    Sanity checks: gamma(0) = 2*K*dt^alpha = Var(single increment) =
    MSD_1D(dt); at alpha=1, gamma(0) = 2*D*dt and gamma(k>=1) = 0, i.e.
    ordinary Brownian motion has independent increments.
    """
    k = jnp.abs(lag).astype(jnp.float64)
    return K * dt_s**alpha * (jnp.abs(k + 1) ** alpha - 2 * k**alpha + jnp.abs(k - 1) ** alpha)


def fgn_covariance(n_disp: int, K: float, dt_s: float, alpha: float) -> jnp.ndarray:
    """Dense (n_disp x n_disp) Toeplitz covariance matrix of fGn increments.

    n_disp is the number of displacements in a track (= track_length - 1).
    Built explicitly rather than via a fast Toeplitz algorithm: track lengths
    here are <=200, so a dense (n_disp)^2 matrix is trivial, and staying
    dense keeps this identical in shape to `noise_covariance` (summed in
    `displacement_covariance`).
    """
    idx = jnp.arange(n_disp)
    lag = jnp.abs(idx[:, None] - idx[None, :])
    return fgn_gamma(lag, K, dt_s, alpha)


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
    n_disp: int, K: float, dt_s: float, alpha: float, sigma2_um2: float
) -> jnp.ndarray:
    """Full covariance of observed displacements: true motion (fGn) + localization noise."""
    return fgn_covariance(n_disp, K, dt_s, alpha) + noise_covariance(n_disp, sigma2_um2)
