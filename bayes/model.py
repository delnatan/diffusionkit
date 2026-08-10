"""NumPyro probabilistic models -- the only place `numpyro.sample` appears.

Michalet & Berglund (Phys. Rev. E 85, 061916, 2012) show that MSD-curve
fitting is a lossy summary statistic: displacements at different lags come
from overlapping, correlated pairs of the same trajectory, and a curve fit
only ever sees the *mean* of that correlated mess at each lag. The efficient
(Cramer-Rao-achieving) estimator instead maximizes the exact likelihood of
the raw per-frame displacement sequence, built from the covariance
Vestergaard, Blainey & Flyvbjerg (Phys. Rev. E 89, 022726, 2014) derive for
Brownian motion + static localization noise. `likelihood.py` generalizes
that covariance to anomalous diffusion via fractional Gaussian noise, the
same construction Kepten, Bronshtein & Garini (Phys. Rev. E 87, 052713,
2013) use for anomalous-exponent estimation. **No MSD curve is computed
anywhere in this package** -- D_alpha, alpha, and the localization precision
sigma are fit directly from `dx = diff(x)`, `dy = diff(y)`.

Three motion models:
  `normal_diffusion_model`     -- 2 params (D, sigma), alpha pinned to 1,
                                   shares `likelihood.displacement_covariance`.
  `anomalous_diffusion_model`  -- 3 params (D_alpha, sigma, alpha), shares
                                   `likelihood.displacement_covariance`.
  `anisotropic_diffusion_model`-- 4 params (D_mean, eps, psi, sigma), alpha
                                   pinned to 1, shares
                                   `likelihood.anisotropic_displacement_covariance`.
`fgn_gamma`'s alpha=1 reduction is exact (see likelihood.py), so the normal
model isn't a separately-derived likelihood, just the alpha=1 restriction of
the anomalous one -- kept as its own model because a genuinely
Brownian-constrained D is often wanted on its own (e.g. for the classic-MSD
D comparison), not just an anomalous fit that happens to land near alpha=1.
Likewise `anisotropic_step_covariance`'s eps=0 reduction is exact (see
likelihood.py), so `anisotropic_diffusion_model` is a strict generalization
of `normal_diffusion_model`, not a separate model family -- see the project's
anisotropy-detection plan for why this matters for a Bayes-factor-style
comparison (H0 nested in H1, rather than a separately-constructed
alternative that the null and alternative likelihoods might silently
disagree about).

Each also has a `batched_*` counterpart that wraps the same per-track sample
statements in a `numpyro.plate("track", n_tracks)`: `likelihood.py`'s
covariance functions are plain broadcasting arithmetic (no reshaping of
their own), so passing (n_tracks, 1, 1)-shaped D/alpha/sigma inside the
plate is all it takes to get a batched (n_tracks, n_disp, n_disp) covariance
and a `MultivariateNormal` with batch_shape=(n_tracks,) -- same physics, one
extra `with` block. `inference.fit_all_tracks` groups tracks by shared
track_length (many tracking datasets have several tracks sharing a length
exactly, e.g. everything that survived to a fixed acquisition cutoff) and
fits each group with one `batched_*` SVI run instead of one optimizer call
per track: fitting independent tracks one Python-level call at a time pays
a fresh JAX trace per call regardless of shape reuse, which can dominate
runtime well beyond the sub-second cost the underlying linear algebra alone
would suggest -- one plated call per length-group amortizes that trace cost
across every track in the group (see FINDINGS.md for a measured example).

`inference.py` runs these two ways: fast batched MAP-like fit
(`fit_all_tracks`, mean-field SVI on the `batched_*` models) for the full
per-track table, and full NUTS posteriors (`sample_posterior`, single-track
models) for a handful of illustrative tracks. MLE, in the pre-numpyro
version of this package a separate implementation with its own optimizer, is
now just either of these called with `priors.WEAK_*_PRIOR` -- same model,
same code, an (almost) flat prior.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .likelihood import anisotropic_displacement_covariance, displacement_covariance
from .priors import AnisotropicModelPrior, AnomalousModelPrior, NormalModelPrior


def normal_diffusion_model(
    dx_um: jnp.ndarray, dy_um: jnp.ndarray, dt_s: float, n_disp: int, prior: NormalModelPrior
) -> None:
    D = numpyro.sample("D", dist.LogNormal(prior.log_D_mean, prior.log_D_sd))
    sigma = numpyro.sample("sigma", dist.LogNormal(prior.log_sigma_mean, prior.log_sigma_sd))
    cov = displacement_covariance(n_disp, D, dt_s, 1.0, sigma**2)
    mvn = dist.MultivariateNormal(jnp.zeros(n_disp), covariance_matrix=cov)
    numpyro.sample("dx_obs", mvn, obs=dx_um)
    numpyro.sample("dy_obs", mvn, obs=dy_um)


def anomalous_diffusion_model(
    dx_um: jnp.ndarray, dy_um: jnp.ndarray, dt_s: float, n_disp: int, prior: AnomalousModelPrior
) -> None:
    D_alpha = numpyro.sample("D_alpha", dist.LogNormal(prior.log_D_mean, prior.log_D_sd))
    sigma = numpyro.sample("sigma", dist.LogNormal(prior.log_sigma_mean, prior.log_sigma_sd))
    # Sampled on (0,1), where Beta's support is correctly reported and
    # numpyro's automatic unconstraining picks a proper SigmoidTransform;
    # `dist.TransformedDistribution(Beta, AffineTransform(0,2))` looks
    # equivalent but its `.support` incorrectly reports the untransformed
    # Real() line, so NUTS/MAP would optimize/sample alpha_unit as if
    # unbounded and wander into the affine map's -inf/NaN region outside
    # (0,2). Rescaling inline (and exposing the physical value as a
    # `deterministic` site, still present in `postprocess_fn`'s output)
    # avoids that bug entirely.
    alpha_unit = numpyro.sample("alpha_unit", dist.Beta(prior.alpha_conc, prior.alpha_conc))
    alpha = numpyro.deterministic("alpha", 2.0 * alpha_unit)
    cov = displacement_covariance(n_disp, D_alpha, dt_s, alpha, sigma**2)
    mvn = dist.MultivariateNormal(jnp.zeros(n_disp), covariance_matrix=cov)
    numpyro.sample("dx_obs", mvn, obs=dx_um)
    numpyro.sample("dy_obs", mvn, obs=dy_um)


def batched_normal_diffusion_model(
    dx_um: jnp.ndarray, dy_um: jnp.ndarray, dt_s: float, n_disp: int,
    prior: NormalModelPrior, n_tracks: int,
) -> None:
    """`normal_diffusion_model` for `n_tracks` tracks of the same `n_disp` at
    once: dx_um/dy_um shape (n_tracks, n_disp); prior fields may be plain
    floats (shared across the plate) or (n_tracks,) arrays (one prior per
    track, e.g. a per-track sigma prior from `sigma_prior_from_localization`)
    -- numpyro/jax broadcasting handles either transparently."""
    with numpyro.plate("track", n_tracks):
        D = numpyro.sample("D", dist.LogNormal(prior.log_D_mean, prior.log_D_sd))
        sigma = numpyro.sample("sigma", dist.LogNormal(prior.log_sigma_mean, prior.log_sigma_sd))
        cov = displacement_covariance(n_disp, D[:, None, None], dt_s, 1.0, (sigma**2)[:, None, None])
        mvn = dist.MultivariateNormal(jnp.zeros(n_disp), covariance_matrix=cov)
        numpyro.sample("dx_obs", mvn, obs=dx_um)
        numpyro.sample("dy_obs", mvn, obs=dy_um)


def batched_anomalous_diffusion_model(
    dx_um: jnp.ndarray, dy_um: jnp.ndarray, dt_s: float, n_disp: int,
    prior: AnomalousModelPrior, n_tracks: int,
) -> None:
    """`anomalous_diffusion_model` for `n_tracks` tracks of the same
    `n_disp` at once -- see `batched_normal_diffusion_model`."""
    with numpyro.plate("track", n_tracks):
        D_alpha = numpyro.sample("D_alpha", dist.LogNormal(prior.log_D_mean, prior.log_D_sd))
        sigma = numpyro.sample("sigma", dist.LogNormal(prior.log_sigma_mean, prior.log_sigma_sd))
        alpha_unit = numpyro.sample("alpha_unit", dist.Beta(prior.alpha_conc, prior.alpha_conc))
        alpha = numpyro.deterministic("alpha", 2.0 * alpha_unit)
        cov = displacement_covariance(
            n_disp, D_alpha[:, None, None], dt_s, alpha[:, None, None], (sigma**2)[:, None, None]
        )
        mvn = dist.MultivariateNormal(jnp.zeros(n_disp), covariance_matrix=cov)
        numpyro.sample("dx_obs", mvn, obs=dx_um)
        numpyro.sample("dy_obs", mvn, obs=dy_um)


def anisotropic_diffusion_model(
    dx_um: jnp.ndarray, dy_um: jnp.ndarray, dt_s: float, n_disp: int, prior: AnisotropicModelPrior
) -> None:
    """H1 of the anisotropy plan: `normal_diffusion_model` generalized to a
    rotated, anisotropic diffusion tensor (D_par/D_perp at unknown
    orientation psi) instead of a shared D. `eps=0` is exactly
    `normal_diffusion_model` (see `likelihood.anisotropic_step_covariance`'s
    docstring) -- H0 is nested in H1 here rather than being a separately
    derived model, so a Savage-Dickey-style comparison at eps=0 is valid and
    the reported `eps` posterior is directly interpretable as an effect size
    (see `priors.AnisotropicModelPrior` for why its prior shrinks toward 0).

    Unlike `normal_diffusion_model`/`anomalous_diffusion_model`, x and y are
    NOT sampled as two independent `n_disp`-dim MVNs sharing one covariance:
    an anisotropic tensor at an orientation not aligned with the x/y axes
    induces real x-y cross-covariance, so this needs one joint
    `2*n_disp`-dim `MultivariateNormal` over the interleaved (dx,dy)
    displacement vector (see `likelihood.anisotropic_displacement_covariance`).
    """
    D_mean = numpyro.sample("D_mean", dist.LogNormal(prior.log_D_mean, prior.log_D_sd))
    eps = numpyro.sample("eps", dist.Beta(prior.eps_a, prior.eps_b))
    psi = numpyro.sample("psi", dist.Uniform(0.0, jnp.pi))
    sigma = numpyro.sample("sigma", dist.LogNormal(prior.log_sigma_mean, prior.log_sigma_sd))
    numpyro.deterministic("D_par", D_mean * (1.0 + eps))
    numpyro.deterministic("D_perp", D_mean * (1.0 - eps))
    cov = anisotropic_displacement_covariance(n_disp, D_mean, eps, psi, dt_s, sigma**2)
    d_obs = jnp.stack([dx_um, dy_um], axis=-1).reshape(-1)
    mvn = dist.MultivariateNormal(jnp.zeros(2 * n_disp), covariance_matrix=cov)
    numpyro.sample("d_obs", mvn, obs=d_obs)


def batched_anisotropic_diffusion_model(
    dx_um: jnp.ndarray, dy_um: jnp.ndarray, dt_s: float, n_disp: int,
    prior: AnisotropicModelPrior, n_tracks: int,
) -> None:
    """`anisotropic_diffusion_model` for `n_tracks` tracks of the same
    `n_disp` at once. Batches the joint covariance via `jax.vmap` over
    `likelihood.anisotropic_displacement_covariance` rather than a
    hand-broadcast version (that function's `jnp.kron` block assembly
    doesn't extend over a leading batch axis the way the isotropic models'
    plain elementwise `displacement_covariance` does) -- no `[:, None,
    None]` reshaping needed as a result, `jax.vmap` handles it."""
    with numpyro.plate("track", n_tracks):
        D_mean = numpyro.sample("D_mean", dist.LogNormal(prior.log_D_mean, prior.log_D_sd))
        eps = numpyro.sample("eps", dist.Beta(prior.eps_a, prior.eps_b))
        psi = numpyro.sample("psi", dist.Uniform(0.0, jnp.pi))
        sigma = numpyro.sample("sigma", dist.LogNormal(prior.log_sigma_mean, prior.log_sigma_sd))
        numpyro.deterministic("D_par", D_mean * (1.0 + eps))
        numpyro.deterministic("D_perp", D_mean * (1.0 - eps))
        cov = jax.vmap(anisotropic_displacement_covariance, in_axes=(None, 0, 0, 0, None, 0))(
            n_disp, D_mean, eps, psi, dt_s, sigma**2
        )
        d_obs = jnp.stack([dx_um, dy_um], axis=-1).reshape(n_tracks, -1)
        mvn = dist.MultivariateNormal(jnp.zeros(2 * n_disp), covariance_matrix=cov)
        numpyro.sample("d_obs", mvn, obs=d_obs)
