"""Prior-predictive Monte Carlo Bayes factor: is a track more anisotropic
than free (isotropic) diffusion would produce at its own track length?

`priors.AnisotropicModelPrior`'s docstring and FINDINGS.md ("Anisotropy
detection") found that a point/interval estimate of `eps` is a poorly-posed
question at N=5-10: 4-9 displacement vectors carry a huge sampling-noise
floor on any *continuous* eccentricity estimate (median apparent eigenvalue
ratio ~5.8x for a genuinely isotropic n_disp=4 track), so no fixed prior
strength gives both a controlled false-positive rate and real sensitivity
from one track's own data. Model *comparison* is different: "is this data
more consistent with some anisotropy than with none" is well-posed even
when there isn't enough information to say *how much*, and a proper Bayes
factor answers it while automatically weighting in exactly how much
apparent elongation is expected from sampling noise alone at this n_disp --
that's precisely what integrating the isotropic model's own likelihood over
its prior encodes, so unlike the original Gemini-drafted note this needs no
separately-simulated "null reference distribution at this N" step.

Why Monte Carlo integration of the raw prior rather than a Savage-Dickey
ratio off the existing NUTS posterior (`model.anisotropic_diffusion_model`):
eps=0 sits at the boundary of a Beta-supported parameter, and no continuous
NUTS draw ever lands exactly there, so estimating the *posterior* density
at that exact point from a finite sample is numerically fragile (would need
a boundary-corrected density fit). Directly estimating both marginal
likelihoods needs no density estimation at all -- just averaging the
(closed-form Gaussian) data likelihood over prior draws. This is only a
good estimator because the N=5-10 regime this module targets has a *weak*
likelihood relative to the prior (see the sampling-noise-floor finding
above): the posterior isn't much more concentrated than the prior, so plain
prior-predictive Monte Carlo (as opposed to posterior importance/bridge
sampling, which would be needed for longer, much more informative tracks --
out of scope here) has acceptable variance.

D_mean and sigma are nuisance parameters shared between H0 (eps pinned to
0) and H1 (eps ~ prior): the same Monte Carlo draws of (D_mean, sigma) are
reused for both integrals (common random numbers), canceling most of their
sampling variance out of the *ratio* even though each marginal likelihood
individually still carries Monte Carlo error on its own.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import polars as pl
from jax.scipy.special import logsumexp
from tqdm import tqdm

from .inference import _stack_tracks
from .likelihood import anisotropic_displacement_covariance
from .priors import AnisotropicModelPrior


def _prior_draws(prior: AnisotropicModelPrior, n_mc: int, key: jax.Array):
    kD, ks, ke, kp = jax.random.split(key, 4)
    D_mean = dist.LogNormal(prior.log_D_mean, prior.log_D_sd).sample(kD, (n_mc,))
    sigma = dist.LogNormal(prior.log_sigma_mean, prior.log_sigma_sd).sample(ks, (n_mc,))
    eps = dist.Beta(prior.eps_a, prior.eps_b).sample(ke, (n_mc,))
    psi = dist.Uniform(0.0, jnp.pi).sample(kp, (n_mc,))
    return D_mean, sigma, eps, psi


def _log_marginal_likelihoods_batch(
    dx_um: jnp.ndarray,
    dy_um: jnp.ndarray,
    dt_s: float,
    n_disp: int,
    prior: AnisotropicModelPrior,
    n_mc: int,
    seed: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """log p(data|H1), log p(data|H0) for a batch of tracks at once
    (dx_um/dy_um shape (n_tracks, n_disp)), sharing one set of `n_mc` prior
    draws across every track: for each Monte Carlo draw, its (unbatched)
    2n-dim covariance is evaluated against every track's d_obs at once
    (`MultivariateNormal.log_prob` broadcasting a single distribution
    against a (n_tracks, 2*n_disp) batch of values, the same well-supported
    broadcast the single-track numpyro models already rely on), and
    `jax.vmap` stacks that over the `n_mc` draws.
    """
    key = jax.random.PRNGKey(seed)
    D_mean, sigma, eps, psi = _prior_draws(prior, n_mc, key)
    d_obs = jnp.stack([dx_um, dy_um], axis=-1).reshape(dx_um.shape[0], -1)  # (n_tracks, 2*n_disp)

    def _logp_all_tracks(D_mean_i, sigma_i, eps_i, psi_i):
        cov = anisotropic_displacement_covariance(n_disp, D_mean_i, eps_i, psi_i, dt_s, sigma_i**2)
        return dist.MultivariateNormal(jnp.zeros(2 * n_disp), covariance_matrix=cov).log_prob(d_obs)

    logp_h1 = jax.vmap(_logp_all_tracks, in_axes=(0, 0, 0, 0))(D_mean, sigma, eps, psi)  # (n_mc, n_tracks)
    log_marg_h1 = logsumexp(logp_h1, axis=0) - jnp.log(n_mc)

    zeros = jnp.zeros(n_mc)
    logp_h0 = jax.vmap(_logp_all_tracks, in_axes=(0, 0, 0, 0))(D_mean, sigma, zeros, zeros)
    log_marg_h0 = logsumexp(logp_h0, axis=0) - jnp.log(n_mc)

    return log_marg_h1, log_marg_h0


def log_bayes_factor_anisotropy(
    dx_um: jnp.ndarray,
    dy_um: jnp.ndarray,
    dt_s: float,
    n_disp: int,
    prior: AnisotropicModelPrior,
    n_mc: int = 20000,
    seed: int = 0,
) -> float:
    """log BF10 for one track (dx_um/dy_um shape (n_disp,)): positive favors
    real anisotropy, negative favors isotropic diffusion, ~0 is
    inconclusive -- see this module's docstring for why this, not a
    point/interval `eps` estimate, is the right tool for "is this track
    more anisotropic than free diffusion at this track length"."""
    log_marg_h1, log_marg_h0 = _log_marginal_likelihoods_batch(
        dx_um[None, :], dy_um[None, :], dt_s, n_disp, prior, n_mc, seed
    )
    return float(log_marg_h1[0] - log_marg_h0[0])


def batched_log_bayes_factor_anisotropy(
    dx_um: jnp.ndarray,
    dy_um: jnp.ndarray,
    dt_s: float,
    n_disp: int,
    prior: AnisotropicModelPrior,
    n_mc: int = 20000,
    seed: int = 0,
) -> jnp.ndarray:
    """log BF10 for `n_tracks` tracks sharing `n_disp` at once (dx_um/dy_um
    shape (n_tracks, n_disp)) -- see `log_bayes_factor_anisotropy`."""
    log_marg_h1, log_marg_h0 = _log_marginal_likelihoods_batch(
        dx_um, dy_um, dt_s, n_disp, prior, n_mc, seed
    )
    return log_marg_h1 - log_marg_h0


def per_track_log_bayes_factor(
    tracks: pl.DataFrame,
    dt_s: float,
    prior: AnisotropicModelPrior,
    min_track_length: int = 5,
    n_mc: int = 20000,
    seed: int = 0,
    show_progress: bool = True,
) -> pl.DataFrame:
    """log BF10 for every eligible track in `tracks` (same input schema
    `inference.fit_all_tracks` expects: particle, frame, x_um, y_um,
    track_length), one row per track (`particle`, `track_length`, `n_disp`,
    `log_bf10`).

    Tracks are grouped by shared track_length purely for efficient batched
    Monte Carlo evaluation (`batched_log_bayes_factor_anisotropy` needs
    every track in one call to share a covariance shape, same reason
    `fit_all_tracks`/`fit_batch_map` group that way -- reuses
    `inference._stack_tracks` for the grouping/stacking itself). This is an
    implementation detail only: `log_bf10` is a per-track additive quantity
    (log evidence) by the time this returns, so it can be aggregated
    (`aggregate_log_bayes_factor`) across *any* grouping the caller wants
    afterward, including across tracks of different lengths.

    `prior` is shared across every track (not a per-track `prior_fn` the way
    `fit_all_tracks` takes one, e.g. from `sigma_prior_from_localization`):
    `_log_marginal_likelihoods_batch` draws one shared set of Monte Carlo
    prior samples for a whole batch specifically so every track in it can
    reuse the same `n_mc` covariance matrices -- a genuinely per-track prior
    would need its own Monte Carlo draws per track, losing that reuse. Not
    needed for the N=5-10 regime this targets (see this module's docstring).
    """
    eligible = tracks.filter(pl.col("track_length") >= min_track_length)
    lengths = eligible["track_length"].unique().sort().to_list()

    chunks = []
    progress = tqdm(total=eligible["particle"].n_unique(), desc="per_track_log_bayes_factor",
                     unit="track", disable=not show_progress)
    for track_length in lengths:
        group = eligible.filter(pl.col("track_length") == track_length)
        particles, dx, dy, _, _ = _stack_tracks(group)
        n_disp = track_length - 1

        progress.set_postfix(track_length=track_length, n_tracks=len(particles))
        logbf = np.asarray(batched_log_bayes_factor_anisotropy(
            jnp.asarray(dx), jnp.asarray(dy), dt_s, n_disp, prior, n_mc=n_mc, seed=seed
        ))
        chunks.append(pl.DataFrame({
            "particle": particles,
            "track_length": [track_length] * len(particles),
            "n_disp": [n_disp] * len(particles),
            "log_bf10": logbf.tolist(),
        }))
        progress.update(len(particles))
    progress.close()

    return pl.concat(chunks).sort("particle")


def aggregate_log_bayes_factor(per_track: pl.DataFrame, label_col: str) -> pl.DataFrame:
    """Population-level log BF10 (`sum_log_bf10`), grouped by `label_col` --
    "is this group of tracks, taken together, more anisotropic than free
    diffusion" (see FINDINGS.md's ensemble-aggregation check: individual
    tracks are almost always inconclusive at N=5-10, but summed evidence
    over a group that genuinely shares anisotropic behavior accumulates
    correctly, without inflating for a group that doesn't).

    `label_col` is any column already on `per_track` (from
    `per_track_log_bayes_factor`) or joined onto it beforehand -- there is
    nothing anisotropy-specific about what defines a group here. It's
    written this way (a plain column name, not a hardcoded notion of
    "condition" or "particle type") specifically so a future per-particle
    classification -- e.g. a spatial region or structure ID, once that
    information exists -- can be joined onto the per-track table and used
    directly as `label_col` without changing this function. Summing
    per-track log-evidence is valid for any grouping of independent tracks,
    not just ones defined ahead of time by the experiment design.
    """
    return (
        per_track.group_by(label_col)
        .agg(n_tracks=pl.len(), sum_log_bf10=pl.col("log_bf10").sum())
        .sort(label_col)
    )
