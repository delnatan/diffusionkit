"""The physics: displacement covariance under fBm + static localization noise.

This module builds the covariance matrix that `model.py` plugs into
`numpyro.distributions.MultivariateNormal` -- it is the only place the
actual diffusion model lives; everything else (priors, inference, sampling)
is generic machinery on top. See `model.py`'s docstring for the physics
references (Michalet & Berglund 2012, Vestergaard et al. 2014, Kepten et al.
2013) and the full derivation.

Model (per spatial dimension, isotropic 2D motion so x and y share the same
D_alpha, alpha, sigma):

  True motion:        fBm with MSD_1D(t) = 2*D_alpha*t^alpha
                       (alpha=1 reduces exactly to ordinary Brownian motion.
                       Total 2D MSD = 4*D_alpha*tau^alpha, matching the
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


def fgn_gamma(lag: jnp.ndarray, D_alpha: float, dt_s: float, alpha: float) -> jnp.ndarray:
    """Autocovariance of fractional Gaussian noise (fBm increments) at integer lag.

    gamma(k) = D_alpha * dt^alpha * (|k+1|^alpha - 2|k|^alpha + |k-1|^alpha)

    Sanity checks: gamma(0) = 2*D_alpha*dt^alpha = Var(single increment) =
    MSD_1D(dt); at alpha=1, gamma(0) = 2*D*dt and gamma(k>=1) = 0, i.e.
    ordinary Brownian motion has independent increments.
    """
    k = jnp.abs(lag).astype(jnp.float64)
    return D_alpha * dt_s**alpha * (jnp.abs(k + 1) ** alpha - 2 * k**alpha + jnp.abs(k - 1) ** alpha)


def fgn_covariance(n_disp: int, D_alpha: float, dt_s: float, alpha: float) -> jnp.ndarray:
    """Dense (n_disp x n_disp) Toeplitz covariance matrix of fGn increments.

    n_disp is the number of displacements in a track (= track_length - 1).
    Built explicitly rather than via a fast Toeplitz algorithm: track lengths
    here are <=200, so a dense (n_disp)^2 matrix is trivial, and staying
    dense keeps this identical in shape to `noise_covariance` (summed in
    `displacement_covariance`).
    """
    idx = jnp.arange(n_disp)
    lag = jnp.abs(idx[:, None] - idx[None, :])
    return fgn_gamma(lag, D_alpha, dt_s, alpha)


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
    n_disp: int, D_alpha: float, dt_s: float, alpha: float, sigma2_um2: float
) -> jnp.ndarray:
    """Full covariance of observed displacements: true motion (fGn) + localization noise."""
    return fgn_covariance(n_disp, D_alpha, dt_s, alpha) + noise_covariance(n_disp, sigma2_um2)


def anisotropic_step_covariance(
    D_mean: float, eps: float, psi: float, dt_s: float
) -> jnp.ndarray:
    """2x2 true-motion covariance of one Brownian step under a rotated,
    anisotropic diffusion tensor.

    D_par = D_mean*(1+eps), D_perp = D_mean*(1-eps) are the diffusivities
    along/across an axis at angle `psi` (radians, mod pi -- a diffusion
    tensor's axis has 180 degree symmetry, not 360); eps in [0,1) is the
    anisotropy fraction (D_par-D_perp)/(D_par+D_perp), 0 = isotropic. At
    eps=0 this reduces exactly to `2*D_mean*dt_s*I2`, independent of psi --
    i.e. the isotropic normal-diffusion model
    (`displacement_covariance(..., alpha=1.0, ...)`'s per-axis variance) is
    the eps=0 slice of this model, not a separately-derived special case.

    Written as elementwise trig arithmetic (not an explicit R @ diag @ R.T
    matmul) purely so this is broadcastable over an arbitrary leading batch
    shape in D_mean/eps/psi for free -- `model.py`'s batched anisotropic
    model calls this directly (unlike `anisotropic_displacement_covariance`
    below, which needs `jax.vmap` for batching).
    """
    D_par = D_mean * (1.0 + eps)
    D_perp = D_mean * (1.0 - eps)
    c, s = jnp.cos(psi), jnp.sin(psi)
    var_x = 2.0 * dt_s * (D_par * c**2 + D_perp * s**2)
    var_y = 2.0 * dt_s * (D_par * s**2 + D_perp * c**2)
    cov_xy = 2.0 * dt_s * (D_par - D_perp) * c * s
    row0 = jnp.stack([var_x, cov_xy], axis=-1)
    row1 = jnp.stack([cov_xy, var_y], axis=-1)
    return jnp.stack([row0, row1], axis=-2)


def anisotropic_displacement_covariance(
    n_disp: int, D_mean: float, eps: float, psi: float, dt_s: float, sigma2_um2: float
) -> jnp.ndarray:
    """Full (2*n_disp, 2*n_disp) covariance of observed 2D displacements,
    interleaved as (dx_1, dy_1, dx_2, dy_2, ..., dx_n, dy_n), under
    anisotropic Brownian motion (`anisotropic_step_covariance`, independent
    across steps since alpha=1) + iid static localization noise.

    Localization noise is still isotropic (eps_i ~ N(0, sigma^2 I2), same
    assumption as `noise_covariance`), so its contribution is a 2x2 identity
    block on the diagonal (2*sigma^2*I2) and off-diagonal (-sigma^2*I2)
    positions -- the direct block generalization of `noise_covariance`'s
    scalar lag-based structure, assembled the same way (`jnp.kron` of a
    (n_disp, n_disp) 0/1 lag pattern with the constant 2x2 block content,
    since every step shares the same D_mean/eps/psi -- no analog of
    `fgn_covariance`'s lag-dependent gamma(k) is needed at alpha=1).

    Scalar-parameter only (no leading batch axis): `jnp.kron` doesn't extend
    over a batch dimension the way `noise_covariance`'s plain `jnp.where`
    does, so `model.py`'s batched anisotropic model instead gets a
    (n_tracks, 2*n_disp, 2*n_disp) covariance via `jax.vmap` over this
    function.
    """
    motion_block = anisotropic_step_covariance(D_mean, eps, psi, dt_s)
    diag_block = motion_block + 2.0 * sigma2_um2 * jnp.eye(2)
    offdiag_block = -sigma2_um2 * jnp.eye(2)

    idx = jnp.arange(n_disp)
    lag = jnp.abs(idx[:, None] - idx[None, :])
    pattern_diag = (lag == 0).astype(diag_block.dtype)
    pattern_offdiag = (lag == 1).astype(diag_block.dtype)

    return jnp.kron(pattern_diag, diag_block) + jnp.kron(pattern_offdiag, offdiag_block)
