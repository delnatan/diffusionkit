"""Per-track anisotropy evidence by nested sampling (jaxns).

This replaces `bayes_factor.py` (removed), which estimated log BF10 by
prior-predictive Monte Carlo -- a good estimator only while the likelihood
stays weak relative to the prior, true at track_length 5-10 and false as
soon as a track is long enough for the question to actually be answerable.
That module's own docstring named nested sampling as the fix; this is it.

Nested sampling computes each hypothesis' evidence directly, so unlike
`fit_map`'s Laplace approximation it does not assume the posterior is
Gaussian, and unlike a Savage-Dickey ratio it never evaluates a density at a
point. Measured against prior-predictive Monte Carlo where MC is still
trustworthy, and against Laplace where it is not (D=0.05 um^2/s,
sigma_loc=0.025 um, dt=0.033 s, eps=0.8):

    n_disp    brute MC     jaxns       Laplace
       4       +0.13       +0.12        +0.14
      19       +2.82       +2.85        +2.99
      99         --       +18.57       +15.79     <- Laplace 2.8 nats low

The cost is a stochastic evidence: jaxns reports +/-0.18 to 0.41 nats per
track here. That is negligible against a long track's log BF10 (+8 to +52,
see below) and larger than the whole signal on a 5-frame track, which is the
honest shape of the problem rather than a defect -- a 4-displacement track
does not contain the information (FINDINGS.md, "The real obstacle is a
sampling-noise floor"), and the evidence correctly says so. Detectability
measured per track, no pooling anywhere (fraction of single tracks reaching
log BF10 > 3 on their own):

    track_length     eps=0     eps=0.5    eps=0.8
         5             0%         0%         0%
        20             0%         3%        33%
        50             0%        35%        97%
       200             2%        98%       100%

So this is deliberately not gated on track length: short tracks return ~0
and that is the answer, not a reason to exclude them.

Parameterization differs from `model.anisotropic_diffusion_model`. The
diffusion tensor is carried in log-Euclidean coordinates,
`Sigma = 2*dt*D_g*expm(h1*sigma_z + h2*sigma_x)`, so (u=log D_g, h1, h2,
v=log sigma) is unconstrained in R^4 with positive-definiteness automatic:

  * isotropy is the *interior* point h=(0,0), not a boundary with an
    unidentified psi sitting on it, so nothing here is a non-regular
    comparison;
  * a lab-frame rotation by theta rotates (h1,h2) by 2*theta, so an
    isotropic prior on h is exactly invariant to the mounting angle -- the
    arbitrary-orientation concern in `anisotropy.py` becomes structural
    rather than something Uniform(0,pi) has to supply;
  * the prior scale `tau_log_ratio` is the prior sd of
    0.5*log(D_par/D_perp), an elicitable log-fold, unlike `eps_b` of a Beta
    (whose pushforward onto the h-plane is singular at isotropy, which is
    what makes the eps_b sweep in FINDINGS.md behave like a step function).

eps/psi/D_par/D_perp are recovered from the posterior.

Note that `exp(u)` is the *geometric* mean sqrt(D_par*D_perp), which is the
natural scale in these coordinates but is **not** the diffusion coefficient
an isotropic MSD fit recovers. Since MSD_2D(tau) = 2*(D_par+D_perp)*tau, the
MSD-equivalent D is the *arithmetic* mean tr(D)/2 = D_g*cosh(|h|). Both are
reported (`D_arith_mean_um2_s`, `D_geom_mean_um2_s`) and neither is called
plain "D_mean": at D_par/D_perp = 9 they differ by 40%, and an earlier
version of this package used the name `D_mean` for the arithmetic one.

jaxns is an optional dependency (`pip install -e ".[nested]"`); it is
imported lazily so this module can be imported without it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from .inference import _Batch, _per_track_table

__all__ = [
    "LogEuclideanAnisotropicPrior",
    "NestedFit",
    "dst_displacements",
    "log_likelihood",
    "fit_track_nested",
    "per_track_nested",
]

_SZ = jnp.array([[1.0, 0.0], [0.0, -1.0]])
_SX = jnp.array([[0.0, 1.0], [1.0, 0.0]])


@dataclass(frozen=True)
class LogEuclideanAnisotropicPrior:
    """Prior for (u=log D_g, h1, h2, v=log sigma). All four are Gaussian, so
    the nested sampler's prior space needs no transform.

    `tau_log_ratio` is the prior sd of each h component, i.e. of
    0.5*log(D_par/D_perp): 0.30 puts a 2-fold axis ratio at ~1.2 sd and a
    5-fold one at ~2.7 sd. It replaces `priors.AnisotropicModelPrior`'s
    `eps_a`/`eps_b` and must be reported alongside any Bayes factor -- the
    evidence for H1 depends on it (Lindley/Bartlett), which is how H1 pays
    for its extra freedom, not a defect to be defaulted away silently.
    """

    log_D_mean: float = float(np.log(0.05))
    log_D_sd: float = float(np.log(10) * 2)
    tau_log_ratio: float = 0.30
    log_sigma_mean: float = float(np.log(0.025))
    log_sigma_sd: float = 0.5


def dst_displacements(dx_um: np.ndarray, dy_um: np.ndarray) -> jnp.ndarray:
    """Type-I discrete sine transform of a track's 2D displacements, shape
    (n_disp, 2).

    Static localization noise contributes `T (x) sigma^2 I2` to the
    displacement covariance with T = tridiag(-1, 2, -1) (see
    `likelihood.anisotropic_displacement_covariance`), and T's eigenbasis is
    the DST-I. Because that noise term is *isotropic*, the transform block-
    diagonalizes the full (2n, 2n) covariance into n independent 2x2 blocks
    `Sigma_m + lambda_j*sigma^2*I2` -- exactly, not approximately (verified
    to ~1e-17 against the dense matrix). The motion tensor is shared across
    every block and the noise only inflates each one isotropically, so the
    anisotropy signal is the eccentricity common to all n modes.

    This turns one (2n)^3 Cholesky into n closed-form 2x2 solves, which is
    what makes a five-figure likelihood-evaluation count per track cheap
    enough for nested sampling. The transform is orthogonal, so no Jacobian
    term enters the likelihood.
    """
    d = np.stack([np.asarray(dx_um), np.asarray(dy_um)], axis=-1)
    n = d.shape[-2]
    k = np.arange(1, n + 1)
    U = np.sqrt(2.0 / (n + 1)) * np.sin(np.outer(k, k) * np.pi / (n + 1))
    return jnp.asarray(U @ d)


def log_likelihood(u: jax.Array, h1: jax.Array, h2: jax.Array, v: jax.Array,
                   w: jnp.ndarray, dt_s: float) -> jax.Array:
    """log p(w | u, h1, h2, v) for one track, `w` from `dst_displacements`."""
    n = w.shape[0]
    r = jnp.sqrt(h1**2 + h2**2 + 1e-300)
    expm = jnp.cosh(r) * jnp.eye(2) + (jnp.sinh(r) / r) * (h1 * _SZ + h2 * _SX)
    sigma_m = 2.0 * dt_s * jnp.exp(u) * expm
    lam = 2.0 - 2.0 * jnp.cos(jnp.arange(1, n + 1) * jnp.pi / (n + 1))
    blocks = sigma_m[None] + lam[:, None, None] * jnp.exp(2.0 * v) * jnp.eye(2)[None]
    a, b, c = blocks[:, 0, 0], blocks[:, 0, 1], blocks[:, 1, 1]
    det = a * c - b * b
    quad = (c * w[:, 0] ** 2 - 2 * b * w[:, 0] * w[:, 1] + a * w[:, 1] ** 2) / det
    return -0.5 * jnp.sum(jnp.log(det) + quad) - n * jnp.log(2 * jnp.pi)


@dataclass(frozen=True)
class NestedFit:
    """One track's anisotropy evidence and posterior, from a single pair of
    nested-sampling runs (H1 with h free, H0 with h pinned to zero).

    `log_bf10_stderr` combines both runs' evidence uncertainties in
    quadrature. They are independent runs, so unlike the removed Monte Carlo
    estimator's shared draws there is no common-random-number cancellation;
    the sampler's own error is the price of not assuming a posterior shape.
    """

    log_bf10: float
    log_bf10_stderr: float
    log_Z1: float
    log_Z0: float
    # Two different means, both reported, because they answer different
    # questions and silently substituting one for the other is a ~40% error
    # at strong anisotropy. See TABLES.md.
    D_arith_mean_um2_s: float    # (D_par+D_perp)/2 = tr(D)/2 -- reproduces the 2D MSD
    D_geom_mean_um2_s: float     # sqrt(D_par*D_perp) -- the log-Euclidean scale, exp(u)
    eps_median: float
    eps_lo: float
    eps_hi: float
    psi_rad: float               # circular mean; psi is defined mod pi
    psi_circular_sd: float
    D_par_um2_s: float
    D_perp_um2_s: float
    n_likelihood_evals: int


def _evidence(w, dt_s, prior, full, seed, max_samples, num_live_points):
    try:
        from jaxns import Model, NestedSampler, Prior, TerminationCondition
        import tensorflow_probability.substrates.jax as tfp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "nested sampling needs the optional 'nested' extra: "
            'pip install -e ".[nested]"'
        ) from exc
    tfpd = tfp.distributions

    def prior_model():
        u = yield Prior(tfpd.Normal(prior.log_D_mean, prior.log_D_sd), name="u")
        v = yield Prior(tfpd.Normal(prior.log_sigma_mean, prior.log_sigma_sd), name="v")
        if not full:
            return u, v
        h1 = yield Prior(tfpd.Normal(0.0, prior.tau_log_ratio), name="h1")
        h2 = yield Prior(tfpd.Normal(0.0, prior.tau_log_ratio), name="h2")
        return u, v, h1, h2

    if full:
        ll = lambda u, v, h1, h2: log_likelihood(u, h1, h2, v, w, dt_s)
    else:
        ll = lambda u, v: log_likelihood(u, 0.0, 0.0, v, w, dt_s)

    ns = NestedSampler(
        model=Model(prior_model=prior_model, log_likelihood=ll),
        max_samples=max_samples, num_live_points=num_live_points,
    )
    reason, state = jax.jit(lambda k: ns(k, TerminationCondition(dlogZ=1e-4)))(
        jax.random.PRNGKey(seed)
    )
    return ns.to_results(termination_reason=reason, state=state)


def fit_track_nested(
    dx_um: np.ndarray,
    dy_um: np.ndarray,
    dt_s: float,
    prior: LogEuclideanAnisotropicPrior | None = None,
    seed: int = 0,
    max_samples: int = 200_000,
    num_live_points: int | None = None,
    n_posterior: int = 4000,
) -> NestedFit:
    """Anisotropy evidence + posterior for one track, by nested sampling.

    Runs the sampler twice -- H1 with the anisotropy free, H0 with it pinned
    to isotropy -- and differences the log evidences. The H1 run's posterior
    samples give eps/psi/D_par/D_perp in the same pass, so this replaces both
    halves of `anisotropy.analyze` (a Monte Carlo Bayes factor *and* a
    separate NUTS fit for the descriptive interval) with one engine.
    """
    p = prior if prior is not None else LogEuclideanAnisotropicPrior()
    w = dst_displacements(dx_um, dy_um)

    r1 = _evidence(w, dt_s, p, True, seed, max_samples, num_live_points)
    r0 = _evidence(w, dt_s, p, False, seed, max_samples, num_live_points)

    from jaxns import resample
    from numpyro.diagnostics import hpdi

    post = resample(jax.random.PRNGKey(seed + 1), r1.samples,
                    r1.log_dp_mean, S=n_posterior, replace=True)
    h1 = np.asarray(post["h1"]); h2 = np.asarray(post["h2"])
    D_g = np.exp(np.asarray(post["u"]))
    r = np.hypot(h1, h2)
    eps = np.tanh(r)
    lo, hi = hpdi(eps, prob=0.9)
    # psi is an axis direction (period pi), so it is summarised circularly on
    # the doubled angle, where the period is 2*pi and a mean is well defined.
    # That doubled angle is atan2(h2, h1) itself -- psi is *half* of it -- so
    # the unit vectors to average are just (h1, h2)/|h|; doubling again here
    # would report 2*psi.
    C, S = np.mean(h1 / np.maximum(r, 1e-300)), np.mean(h2 / np.maximum(r, 1e-300))
    R = np.hypot(C, S)
    return NestedFit(
        log_bf10=float(r1.log_Z_mean - r0.log_Z_mean),
        log_bf10_stderr=float(np.hypot(r1.log_Z_uncert, r0.log_Z_uncert)),
        log_Z1=float(r1.log_Z_mean), log_Z0=float(r0.log_Z_mean),
        D_arith_mean_um2_s=float(np.median(D_g * np.cosh(r))),
        D_geom_mean_um2_s=float(np.median(D_g)),
        eps_median=float(np.median(eps)), eps_lo=float(lo), eps_hi=float(hi),
        psi_rad=float(0.5 * np.arctan2(S, C) % np.pi),
        psi_circular_sd=float(0.5 * np.sqrt(-2.0 * np.log(max(R, 1e-12)))),
        D_par_um2_s=float(np.median(D_g * np.exp(r))),
        D_perp_um2_s=float(np.median(D_g * np.exp(-r))),
        n_likelihood_evals=int(r1.total_num_likelihood_evaluations
                               + r0.total_num_likelihood_evaluations),
    )


def per_track_nested(
    tracks: pl.DataFrame,
    dt_s: float,
    prior: LogEuclideanAnisotropicPrior | None = None,
    min_track_length: int = 5,
    seed: int = 0,
    show_progress: bool = True,
    progress: Callable[[int, int], None] | None = None,
    **fit_kwargs,
) -> pl.DataFrame:
    """`fit_track_nested` for every eligible track, one row each.

    Unlike `anisotropy.analyze` there is no `max_track_length`: nested
    sampling is exactly the engine that stays valid as a track gets long and
    informative, which is where per-track anisotropy is answerable at all.
    Tracks are still grouped by `track_length` (via
    `inference._per_track_table`) but each is sampled on its own -- nested
    sampling computes one evidence per run and cannot be plated across
    tracks the way the MAP/SVI engines are.
    """
    p = prior if prior is not None else LogEuclideanAnisotropicPrior()

    def fit_batch(batch: _Batch) -> pl.DataFrame:
        rows = [
            fit_track_nested(batch.dx[i], batch.dy[i], dt_s, p, seed=seed, **fit_kwargs)
            for i in range(batch.n_tracks)
        ]
        return pl.DataFrame({
            **batch.index(),
            "log_bf10": [r.log_bf10 for r in rows],
            "log_bf10_stderr": [r.log_bf10_stderr for r in rows],
            "eps_median": [r.eps_median for r in rows],
            "eps_lo": [r.eps_lo for r in rows],
            "eps_hi": [r.eps_hi for r in rows],
            "psi_median_rad": [r.psi_rad for r in rows],
            "psi_circular_sd_rad": [r.psi_circular_sd for r in rows],
            "D_arith_mean_median_um2_s": [r.D_arith_mean_um2_s for r in rows],
            "D_geom_mean_median_um2_s": [r.D_geom_mean_um2_s for r in rows],
            "D_par_median_um2_s": [r.D_par_um2_s for r in rows],
            "D_perp_median_um2_s": [r.D_perp_um2_s for r in rows],
        })

    return _per_track_table(
        tracks, fit_batch, "per_track_nested", min_track_length,
        show_progress=show_progress, progress=progress,
    )
