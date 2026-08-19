"""Fit `model.py`'s numpyro models: fast batched MAP-like fit for the full
per-track table, and full NUTS posteriors for a handful of illustrative
tracks.

One inference engine, not a separate MLE/Bayes split. `fit_all_tracks`/
`fit_map`/`sample_posterior` are generic over *any* numpyro model and prior
-- `priors.WEAK_*_PRIOR` makes them behave like a flat-prior MLE,
`priors.NormalModelPrior()`/`AnomalousModelPrior()` (or a per-track prior
from `sigma_prior_from_localization`) makes them a proper Bayesian fit.
There is exactly one code path either way.

`fit_map` (single track or batched model alike) works directly with
numpyro's own unconstrained `potential_fn` (the same one `NUTS` uses
internally) plus exact JAX gradients/Hessian -- including
`unconstrained_params`/`unconstrained_cov` on the returned `MAPFit`, the
theta-space (e.g. log(D) for a LogNormal-supported site) MAP and covariance
that the Laplace approximation is actually Gaussian in, before any
delta-method transform to physical units. `fit_all_tracks` (the current
full-table path) instead groups tracks by shared track_length and fits each
group in one mean-field (`AutoNormal`) SVI run on a `batched_*` model from
model.py -- see that module's docstring for why grouping matters (fitting
many tracks one Python-level optimizer call at a time pays a fresh JAX trace
per call regardless of shape reuse). FINDINGS.md documents follow-up checks
finding `fit_map`'s exact L-BFGS-B fit both more accurate and better-
calibrated than `fit_all_tracks`'s SVI/Adam path, and the decision to
report D (and D_alpha) via their log-space Laplace fit with an asymmetric
back-transformed interval rather than a symmetric one in linear units --
`fit_all_tracks` has not yet been updated to a batched exact-MAP path (the
dense per-group Hessian `fit_map` uses does not scale past a few dozen
tracks; see FINDINGS.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import polars as pl
from jax.flatten_util import ravel_pytree
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import initialize_model
from scipy.optimize import minimize
from tqdm import tqdm


@dataclass(frozen=True)
class MAPFit:
    params: dict[str, float | np.ndarray]  # physical (constrained) units
    stderr: dict[str, float | np.ndarray]  # marginal Laplace stderr, physical units
    cov: dict[str, dict[str, float]] | None  # full Laplace covariance (physical units);
    # single-track fits only (see fit_map) -- None for batched (array-valued) fits,
    # where a single full covariance matrix would mix different tracks' parameters.
    unconstrained_params: dict[str, float | np.ndarray]  # theta-space MAP, e.g. log(D) for a
    # LogNormal-supported site (numpyro's default unconstraining bijector for positive-real
    # support is exp/log) -- the Laplace approximation is a Gaussian in *this* space by
    # construction (it's literally what the Hessian is taken with respect to); `cov` above is
    # that Gaussian pushed forward through the constraining transform via the delta method,
    # which is only accurate where the transform is locally ~linear over the posterior's width.
    # For a scale parameter like D with a wide/short-track posterior, working directly in
    # `unconstrained_params`/`unconstrained_cov` (i.e. log(D)) avoids that approximation error
    # entirely instead of introducing and then correcting for it.
    unconstrained_cov: dict[str, dict[str, float]] | None  # full covariance in theta-space;
    # same single-track-only scope as `cov`.
    unconstrained_stderr: dict[str, float | np.ndarray]  # marginal theta-space stderr (sqrt of
    # cov_unconstrained's diagonal) -- unlike `unconstrained_cov`, works for batched (array-valued)
    # fits too, since a marginal (per-site, per-track) stderr doesn't need a shared name ordering
    # the way a full cross-parameter covariance matrix does.
    log_posterior: float
    converged: bool


def _to_python(x) -> float | np.ndarray:
    arr = np.asarray(x)
    return float(arr) if arr.ndim == 0 else arr


def fit_map(model_fn: Callable, model_args: tuple, seed: int = 0) -> MAPFit:
    """Exact MAP + Laplace (delta-method) stderr, via numpyro's own
    unconstrained `potential_fn` + exact JAX gradient/Hessian -- no
    hand-written parameter transforms (replacing this package's old
    `params.py`) and no finite-difference derivatives (replacing the old
    `posterior.numerical_hessian`). The delta-method covariance from
    unconstrained to physical units is likewise exact autodiff (`jax.jacfwd`
    of numpyro's `postprocess_fn`), correct for whatever prior/support each
    parameter has without a hand-derived formula per parameter.

    Works on both a single-track model and a `batched_*` model (`model_fn`
    is generic; `ravel_pytree` flattens whatever pytree of sites
    `initialize_model` produces, scalar or plated (n_tracks,) arrays alike)
    -- see `fit_batch_map` for the batched entry point and why exact L-BFGS-B
    is worth it over `fit_batch_svi`'s Adam-optimized ELBO for a flat-prior
    (MLE) fit specifically.
    """
    model_info = initialize_model(
        jax.random.PRNGKey(seed), model_fn, model_args=model_args, dynamic_args=False
    )
    potential_fn = model_info.potential_fn
    flat_init, unravel = ravel_pytree(model_info.param_info.z)

    def neg_log_post(flat_params):
        return potential_fn(unravel(flat_params))

    val_grad = jax.jit(jax.value_and_grad(neg_log_post))

    def scipy_obj(flat_params):
        v, g = val_grad(jnp.asarray(flat_params))
        return float(v), np.asarray(g, dtype=np.float64)

    res = minimize(scipy_obj, np.asarray(flat_init), jac=True, method="L-BFGS-B")
    flat_map = jnp.asarray(res.x)
    z_map = unravel(flat_map)
    constrained_map = model_info.postprocess_fn(z_map)
    flat_map_c, unravel_c = ravel_pytree(constrained_map)

    def constrained_flat(flat_params):
        c = model_info.postprocess_fn(unravel(flat_params))
        flat_c, _ = ravel_pytree(c)
        return flat_c

    stderr = {k: _to_python(np.full(np.shape(v), np.nan)) for k, v in constrained_map.items()}
    unconstrained_stderr = {k: _to_python(np.full(np.shape(v), np.nan)) for k, v in z_map.items()}
    cov: dict[str, dict[str, float]] | None = None
    unconstrained_cov: dict[str, dict[str, float]] | None = None
    try:
        hessian = jax.hessian(neg_log_post)(flat_map)
        cov_unconstrained = jnp.linalg.inv(hessian)
        jac = jax.jacfwd(constrained_flat)(flat_map)
        cov_constrained = jac @ cov_unconstrained @ jac.T
        stderr_flat = jnp.sqrt(jnp.clip(jnp.diag(cov_constrained), 0.0, None))
        if jnp.all(jnp.isfinite(stderr_flat)):
            stderr = {k: _to_python(v) for k, v in unravel_c(stderr_flat).items()}
        u_stderr_flat = jnp.sqrt(jnp.clip(jnp.diag(cov_unconstrained), 0.0, None))
        if jnp.all(jnp.isfinite(u_stderr_flat)):
            unconstrained_stderr = {k: _to_python(v) for k, v in unravel(u_stderr_flat).items()}
        # Only a single, unambiguous parameter-name ordering exists when every
        # site is scalar (single-track fit); ravel_pytree flattens dicts by
        # sorted key, so that order is what cov_constrained's/cov_unconstrained's
        # rows/cols mean.
        if all(np.ndim(v) == 0 for v in constrained_map.values()) and np.all(np.isfinite(cov_constrained)):
            names = sorted(constrained_map.keys())
            cov_np = np.asarray(cov_constrained)
            cov = {a: {b: float(cov_np[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)}
        if all(np.ndim(v) == 0 for v in z_map.values()) and np.all(np.isfinite(cov_unconstrained)):
            u_names = sorted(z_map.keys())
            ucov_np = np.asarray(cov_unconstrained)
            unconstrained_cov = {
                a: {b: float(ucov_np[i, j]) for j, b in enumerate(u_names)} for i, a in enumerate(u_names)
            }
    except Exception:
        pass  # Hessian not usable at this optimum -- leave stderr as NaN, params/MAP still valid

    return MAPFit(
        params={k: _to_python(v) for k, v in constrained_map.items()},
        stderr=stderr,
        cov=cov,
        unconstrained_params={k: _to_python(v) for k, v in z_map.items()},
        unconstrained_cov=unconstrained_cov,
        unconstrained_stderr=unconstrained_stderr,
        log_posterior=float(-res.fun),
        converged=bool(res.success),
    )


def sample_posterior(
    model_fn: Callable,
    model_args: tuple,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], MCMC]:
    """Full NUTS posterior for a single track. Returns (samples, mcmc);
    samples[name] has shape (num_chains, num_samples). Genuine parallel
    chains require `numpyro.set_host_device_count` called before jax
    initializes any device (see scripts/run_bayes_analysis.py) -- without it
    numpyro still runs the requested chains correctly, just sequentially.
    """
    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), *model_args)
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    return samples, mcmc


def sample_posterior_table(
    tracks: pl.DataFrame,
    model_fn: Callable,
    dt_s: float,
    prior_fn: Callable[[np.ndarray, np.ndarray], object],
    param_names: list[str],
    min_track_length: int = 5,
    hpdi_prob: float = 0.9,
    num_warmup: int = 400,
    num_samples: int = 800,
    num_chains: int = 4,
    seed: int = 0,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Full-NUTS-posterior per-track table: median + `hpdi_prob` HPDI for
    each of `param_names`, one row per track (`track_id`, `track_length`,
    `n_disp`, then `{name}_median`/`{name}_lo`/`{name}_hi`).

    `fit_all_tracks`/`fit_batch_map`'s counterpart for when the *posterior
    shape itself*, not just its median, is what's needed -- a Gaussian
    (Laplace/mean-field) approximation is unreliable exactly where this
    matters, e.g. `anisotropic_diffusion_model`'s eps=0 boundary ridge (see
    FINDINGS.md). Generic over `model_fn`/`param_names` the same way those
    two are, including any `numpyro.deterministic` site (e.g. `D_par`,
    `D_perp`), not just `numpyro.sample` sites -- both appear in
    `mcmc.get_samples()` identically.

    Tracks are grouped by shared track_length for the same reason
    `fit_all_tracks`/`fit_batch_map` are (`batched_*` models need one
    covariance shape per call); each group gets one `sample_posterior` call
    on a `batched_*` model.
    """
    eligible = tracks.filter(pl.col("track_length") >= min_track_length)
    lengths = eligible["track_length"].unique().sort().to_list()

    chunks = []
    progress = tqdm(total=eligible["track_id"].n_unique(), desc="sample_posterior_table",
                     unit="track", disable=not show_progress)
    for track_length in lengths:
        group = eligible.filter(pl.col("track_length") == track_length)
        particles, dx, dy, xstd, ystd = _stack_tracks(group)
        n_tracks, n_disp = len(particles), track_length - 1
        prior = prior_fn(xstd, ystd)

        progress.set_postfix(track_length=track_length, n_tracks=n_tracks)
        samples, _ = sample_posterior(
            model_fn, (jnp.asarray(dx), jnp.asarray(dy), dt_s, n_disp, prior, n_tracks),
            num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, seed=seed,
        )
        row = {"track_id": particles, "track_length": [track_length] * n_tracks,
               "n_disp": [n_disp] * n_tracks}
        for name in param_names:
            arr = samples[name].reshape(-1, n_tracks)  # (draws, n_tracks)
            lo, hi = hpdi(arr, hpdi_prob, axis=0)
            row[f"{name}_median"] = np.median(arr, axis=0).tolist()
            row[f"{name}_lo"] = np.asarray(lo).tolist()
            row[f"{name}_hi"] = np.asarray(hi).tolist()
        chunks.append(pl.DataFrame(row))
        progress.update(n_tracks)
    progress.close()

    return pl.concat(chunks).sort("track_id")


def fit_batch_svi(
    model_fn: Callable,
    dx_um: jnp.ndarray,
    dy_um: jnp.ndarray,
    dt_s: float,
    n_disp: int,
    prior: object,
    n_tracks: int,
    num_steps: int = 2000,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """MAP-like medians + marginal stderr for `n_tracks` tracks sharing the
    same `n_disp`, via one SVI run on a `batched_*` model's plated
    (n_tracks,) dimension (see model.py). A mean-field (`AutoNormal`) guide
    is enough here: only marginal stderr per parameter is ever reported
    downstream (never the cross-parameter covariance a full Laplace fit
    would add), so the guide's independence assumption costs nothing.
    Stderr is (84th - 16th percentile)/2 of the guide's posterior in
    physical units, not a symmetric delta-method estimate -- correctly
    asymmetric near a bounded parameter's edge (e.g. alpha close to 0 or 2).
    """
    guide = AutoNormal(model_fn)
    svi = SVI(model_fn, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    result = svi.run(
        jax.random.PRNGKey(seed), num_steps, dx_um, dy_um, dt_s, n_disp, prior, n_tracks,
        progress_bar=False,
    )
    medians = {k: np.asarray(v) for k, v in guide.median(result.params).items()}
    quantiles = guide.quantiles(result.params, [0.159, 0.841])
    stderrs = {k: np.asarray((v[1] - v[0]) / 2.0) for k, v in quantiles.items()}
    return medians, stderrs


def _stack_tracks(
    group: pl.DataFrame,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """group: rows for many tracks all at one shared track_length.
    Returns (track_ids, dx, dy, sigma_x_um, sigma_y_um), each of the latter
    four stacked to shape (n_tracks, track_length[-1]) or (n_tracks, track_length)."""
    particles = group["track_id"].unique().sort().to_list()
    dx_list, dy_list, xstd_list, ystd_list = [], [], [], []
    for pid in particles:
        g = group.filter(pl.col("track_id") == pid).sort("frame")
        dx_list.append(np.diff(g["x_um"].to_numpy()))
        dy_list.append(np.diff(g["y_um"].to_numpy()))
        xstd_list.append(g["sigma_x_um"].to_numpy())
        ystd_list.append(g["sigma_y_um"].to_numpy())
    return particles, np.stack(dx_list), np.stack(dy_list), np.stack(xstd_list), np.stack(ystd_list)


def fit_all_tracks(
    tracks: pl.DataFrame,
    model_fn: Callable,
    dt_s: float,
    prior_fn: Callable[[np.ndarray, np.ndarray], object],
    param_names: list[str],
    min_track_length: int = 10,
    num_steps: int = 2000,
    seed: int = 0,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Fast batched fit for every eligible track.

    Tracks are grouped by shared track_length (tracking data commonly has
    many tracks sharing a length exactly -- e.g. every track that survived
    to a fixed acquisition cutoff) and each group fit in one
    `fit_batch_svi` call on `model_fn` (a `batched_*` model from model.py);
    see model.py's and this module's docstrings for why that matters for
    performance.

    `prior_fn(sigma_x_um, sigma_y_um) -> prior` receives the whole group's
    localization-precision arrays (shape (n_tracks_in_group, track_length))
    and returns a prior whose fields may be per-track (n_tracks,) arrays
    (e.g. from `sigma_prior_from_localization`, called on these same
    arrays) or plain floats shared across the group (e.g.
    `priors.WEAK_ANOMALOUS_PRIOR`, for a flat-prior / MLE-like fit) --
    numpyro/jax broadcasting handles either.

    No row is dropped or flagged for producing an implausible estimate the
    way the classic MSD fit's `D_negative`/`intercept_negative` flags were
    needed: D/D_alpha and alpha are structurally positive/bounded in (0,2)
    by each parameter's own distributional support, not by a post-hoc check.
    """
    eligible = tracks.filter(pl.col("track_length") >= min_track_length)
    lengths = eligible["track_length"].unique().sort().to_list()

    chunks = []
    progress = tqdm(total=eligible["track_id"].n_unique(), desc="fit_all_tracks", unit="track",
                     disable=not show_progress)
    for track_length in lengths:
        group = eligible.filter(pl.col("track_length") == track_length)
        particles, dx, dy, xstd, ystd = _stack_tracks(group)
        n_tracks = len(particles)
        n_disp = track_length - 1
        prior = prior_fn(xstd, ystd)

        progress.set_postfix(track_length=track_length, n_tracks=n_tracks)
        medians, stderrs = fit_batch_svi(
            model_fn, jnp.asarray(dx), jnp.asarray(dy), dt_s, n_disp, prior, n_tracks,
            num_steps=num_steps, seed=seed,
        )

        chunk = {
            "track_id": particles,
            "track_length": [track_length] * n_tracks,
            "n_disp": [n_disp] * n_tracks,
        }
        for name in param_names:
            chunk[name] = medians[name].tolist()
            chunk[f"{name}_stderr"] = stderrs[name].tolist()
        chunks.append(pl.DataFrame(chunk))
        progress.update(n_tracks)
    progress.close()

    return pl.concat(chunks).sort("track_id")


def _map_fit_chunk(
    fit: MAPFit, particles: list[int], track_length: int, n_disp: int, param_names: list[str]
) -> pl.DataFrame:
    """One `fit_map` result (batched, `sub_n` tracks) -> a table chunk.

    For any `name` that is itself a numpyro sample site with positive-real
    (LogNormal) support -- true of D, D_alpha, sigma, identifiable because
    `fit.unconstrained_params` has a matching key -- reports the Laplace fit
    in log-space with an asymmetric back-transformed interval
    (`{name}_median`/`_lo`/`_hi` in physical units, `log10_{name}`/`_stderr`
    in log10 units), per FINDINGS.md ("D should be reported in log-space,
    with an asymmetric interval"): pushing a wide/short-track log-space
    Gaussian through `exp()` and reporting mean +/- stderr in linear units
    understates the true skew and can report an interval touching or
    crossing zero for a strictly-positive quantity.

    Other names (e.g. `alpha`, a numpyro.deterministic derived from a
    Beta-distributed site with no `log`-shaped unconstrained form of its
    own) get the physical-space MAP +/- symmetric Laplace stderr, which
    FINDINGS.md's checks found adequate for alpha specifically.
    """
    sub_n = len(particles)
    row = {
        "track_id": particles,
        "track_length": [track_length] * sub_n,
        "n_disp": [n_disp] * sub_n,
        "converged": [fit.converged] * sub_n,
    }
    log10 = np.log(10.0)
    for name in param_names:
        if name in fit.unconstrained_params:
            log_mean = np.atleast_1d(np.asarray(fit.unconstrained_params[name]))
            log_stderr = np.atleast_1d(np.asarray(fit.unconstrained_stderr[name]))
            row[f"{name}_median"] = np.exp(log_mean).tolist()
            row[f"{name}_lo"] = np.exp(log_mean - log_stderr).tolist()
            row[f"{name}_hi"] = np.exp(log_mean + log_stderr).tolist()
            row[f"log10_{name}"] = (log_mean / log10).tolist()
            row[f"log10_{name}_stderr"] = (log_stderr / log10).tolist()
        else:
            row[name] = np.atleast_1d(np.asarray(fit.params[name])).tolist()
            row[f"{name}_stderr"] = np.atleast_1d(np.asarray(fit.stderr[name])).tolist()
    return pl.DataFrame(row)


def fit_batch_map(
    tracks: pl.DataFrame,
    model_fn: Callable,
    dt_s: float,
    prior_fn: Callable[[np.ndarray, np.ndarray], object],
    param_names: list[str],
    min_track_length: int = 10,
    max_batch_size: int = 20,
    seed: int = 0,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Batched exact MAP (L-BFGS-B on the joint likelihood) for every
    eligible track -- the production alternative to `fit_all_tracks`
    (SVI/Adam) that FINDINGS.md's "Inference-engine choice for production"
    section documents as more accurate and better-calibrated.

    Tracks are grouped by shared track_length as in `fit_all_tracks`, but
    each group is additionally split into sub-batches of at most
    `max_batch_size` tracks before calling `fit_map`: `fit_map`'s Hessian is
    dense over *all* free parameters in the call at once, so its cost is
    superlinear in track count and it OOM-crashes on a large single group
    (measured: fine up to a few dozen tracks, crashed at 140 -- see
    FINDINGS.md). Sub-batch results are appended into one table, so this
    trades a bit of amortization (more, smaller L-BFGS-B/Hessian calls than
    `fit_all_tracks`'s one-call-per-length-group) for staying within memory
    regardless of how many tracks share a length.

    `converged` reflects the *whole sub-batch's* joint L-BFGS-B optimization
    (there is one `scipy.optimize.minimize` call per sub-batch, not per
    track), so it is necessarily coarser than a genuine per-track
    diagnostic; keep `max_batch_size` modest if per-track convergence
    granularity matters.

    See `_map_fit_chunk` for the log-space-with-asymmetric-interval vs.
    symmetric-physical-space reporting split.

    `show_progress` prints a `tqdm` bar over tracks fit so far (each
    sub-batch can take anywhere from under a second to tens of seconds
    depending on track length and `max_batch_size`, and a full real dataset
    is on the order of a hundred-plus sub-batches -- worth seeing progress
    on rather than waiting on a fully-buffered print at the end, especially
    for a backgrounded/redirected run).
    """
    eligible = tracks.filter(pl.col("track_length") >= min_track_length)
    lengths = eligible["track_length"].unique().sort().to_list()

    chunks = []
    total_tracks = eligible["track_id"].n_unique()
    progress = tqdm(total=total_tracks, desc="fit_batch_map", unit="track", disable=not show_progress)
    for track_length in lengths:
        group = eligible.filter(pl.col("track_length") == track_length)
        particles, dx, dy, xstd, ystd = _stack_tracks(group)
        n_tracks = len(particles)
        n_disp = track_length - 1

        for start in range(0, n_tracks, max_batch_size):
            end = min(start + max_batch_size, n_tracks)
            sub_particles = particles[start:end]
            sub_n = end - start
            prior = prior_fn(xstd[start:end], ystd[start:end])

            progress.set_postfix(track_length=track_length, batch=f"{start}-{end}/{n_tracks}")
            fit = fit_map(
                model_fn,
                (jnp.asarray(dx[start:end]), jnp.asarray(dy[start:end]), dt_s, n_disp, prior, sub_n),
                seed=seed,
            )
            chunks.append(_map_fit_chunk(fit, sub_particles, track_length, n_disp, param_names))
            progress.update(sub_n)
    progress.close()

    return pl.concat(chunks).sort("track_id")
