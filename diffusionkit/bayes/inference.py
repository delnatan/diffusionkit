"""Fit `model.py`'s numpyro models.

One inference engine, not a separate MLE/Bayes split: every function here is
generic over *any* numpyro model and prior, so `priors.WEAK_*_PRIOR` makes
them behave like a flat-prior MLE and an informative prior (e.g. per-track,
from `sigma_prior_from_localization`) makes them a proper Bayesian fit.
Exactly one code path either way.

Three single-call engines, each fitting one batch:

  `fit_map`          exact MAP + Laplace, via L-BFGS-B on numpyro's own
                     unconstrained `potential_fn` with exact JAX
                     gradient/Hessian. Single-track or batched model alike.
  `sample_posterior` full NUTS.
  `fit_batch_svi`    mean-field (`AutoNormal`) SVI.

and three per-track table builders wrapping them -- `fit_table_map`
(production), `fit_table_svi` (validation/comparison), `fit_table_nuts`
(when posterior shape matters). All three share `_per_track_table`, which
groups tracks by shared track_length and loops: fitting tracks one
Python-level call at a time pays a fresh JAX trace per call regardless of
shape reuse, so batching a whole length-group into one plated call is what
makes a full dataset tractable (see model.py, and FINDINGS.md for measured
numbers).

FINDINGS.md documents why `fit_table_map` is the production choice over
`fit_table_svi` (more accurate, far better calibrated) and why D is reported
from its log-space Laplace fit with an asymmetric interval.
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
    """MAP estimate + Laplace uncertainty, in both parameter spaces.

    The Laplace approximation is a Gaussian in numpyro's *unconstrained*
    space by construction -- that is what the Hessian is taken with respect
    to. `params`/`stderr`/`cov` are that Gaussian pushed forward to physical
    units by the delta method, which is only accurate where the transform is
    locally ~linear over the posterior's width; for a scale parameter like D
    with a wide short-track posterior, reading `unconstrained_*` (i.e.
    log(D)) directly avoids that error rather than correcting for it. See
    `_map_chunk`.
    """

    params: dict[str, float | np.ndarray]  # physical (constrained) units
    stderr: dict[str, float | np.ndarray]  # marginal Laplace stderr, physical units
    cov: dict[str, dict[str, float]] | None  # full Laplace covariance, physical units.
    # Single-track fits only -- None for batched (array-valued) fits, where one covariance
    # matrix would mix different tracks' parameters.
    unconstrained_params: dict[str, float | np.ndarray]  # theta-space MAP, e.g. log(D)
    unconstrained_stderr: dict[str, float | np.ndarray]  # marginal theta-space stderr; unlike
    # `cov`, this works for batched fits too, since a per-site marginal needs no shared
    # name ordering across tracks.
    converged: bool


def _to_python(x) -> float | np.ndarray:
    arr = np.asarray(x)
    return float(arr) if arr.ndim == 0 else arr


def fit_map(model_fn: Callable, model_args: tuple, seed: int = 0) -> MAPFit:
    """Exact MAP + Laplace (delta-method) stderr, via numpyro's own
    unconstrained `potential_fn` + exact JAX gradient/Hessian -- no
    hand-written parameter transforms and no finite-difference derivatives.
    The delta-method covariance from
    unconstrained to physical units is likewise exact autodiff (`jax.jacfwd`
    of numpyro's `postprocess_fn`), correct for whatever prior/support each
    parameter has without a hand-derived formula per parameter.

    Works on both a single-track model and a `batched_*` model (`model_fn`
    is generic; `ravel_pytree` flattens whatever pytree of sites
    `initialize_model` produces, scalar or plated (n_tracks,) arrays alike)
    -- see `fit_table_map` for the batched entry point.
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
    except Exception:
        pass  # Hessian not usable at this optimum -- leave stderr as NaN, params/MAP still valid

    return MAPFit(
        params={k: _to_python(v) for k, v in constrained_map.items()},
        stderr=stderr,
        cov=cov,
        unconstrained_params={k: _to_python(v) for k, v in z_map.items()},
        unconstrained_stderr=unconstrained_stderr,
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




@dataclass(frozen=True)
class _Batch:
    """One call's worth of tracks: all the same `track_length`, stacked.

    `batched_*` models build a single covariance of one shape per call, so a
    batch is always length-homogeneous. `dx`/`dy` have shape
    (n_tracks, n_disp); `sigma_x_um`/`sigma_y_um` have shape
    (n_tracks, track_length) and are what `prior_fn` sees.
    """

    track_ids: list[int]
    track_length: int
    dx: np.ndarray
    dy: np.ndarray
    sigma_x_um: np.ndarray
    sigma_y_um: np.ndarray

    @property
    def n_tracks(self) -> int:
        return len(self.track_ids)

    @property
    def n_disp(self) -> int:
        return self.track_length - 1

    def index(self) -> dict[str, list]:
        """The three identity columns every per-track table starts with."""
        return {
            "track_id": self.track_ids,
            "track_length": [self.track_length] * self.n_tracks,
            "n_disp": [self.n_disp] * self.n_tracks,
        }


def _stack_tracks(group: pl.DataFrame) -> _Batch:
    """Rows for many tracks of one shared track_length -> a stacked `_Batch`."""
    parts = group.sort(["track_id", "frame"]).partition_by("track_id", maintain_order=True)
    return _Batch(
        track_ids=[int(part["track_id"][0]) for part in parts],
        track_length=int(group["track_length"][0]),
        dx=np.stack([np.diff(part["x_um"].to_numpy()) for part in parts]),
        dy=np.stack([np.diff(part["y_um"].to_numpy()) for part in parts]),
        sigma_x_um=np.stack([part["sigma_x_um"].to_numpy() for part in parts]),
        sigma_y_um=np.stack([part["sigma_y_um"].to_numpy() for part in parts]),
    )


def _batches(tracks: pl.DataFrame, max_batch_size: int | None) -> list[_Batch]:
    """Split `tracks` into length-homogeneous batches, largest unit first.

    Grouping by shared track_length is what makes batched inference possible
    at all (one covariance shape per call), and tracking datasets commonly
    have many tracks sharing a length exactly -- everything that survived to
    a fixed acquisition cutoff. `max_batch_size` additionally caps how many
    tracks go into one call, for engines whose cost is superlinear in batch
    size (see `fit_table_map`); `None` means "the whole length-group".
    """
    out = []
    for track_length in tracks["track_length"].unique().sort().to_list():
        batch = _stack_tracks(tracks.filter(pl.col("track_length") == track_length))
        size = max_batch_size or batch.n_tracks
        for start in range(0, batch.n_tracks, size):
            stop = min(start + size, batch.n_tracks)
            out.append(
                _Batch(
                    track_ids=batch.track_ids[start:stop],
                    track_length=batch.track_length,
                    dx=batch.dx[start:stop],
                    dy=batch.dy[start:stop],
                    sigma_x_um=batch.sigma_x_um[start:stop],
                    sigma_y_um=batch.sigma_y_um[start:stop],
                )
            )
    return out


def _per_track_table(
    tracks: pl.DataFrame,
    fit_batch: Callable[[_Batch], pl.DataFrame],
    desc: str,
    min_track_length: int,
    max_batch_size: int | None = None,
    show_progress: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """Run `fit_batch` over every eligible batch and stack the results.

    The one loop shared by every per-track table in this package
    (`fit_table_map`/`_svi`/`_nuts` and `nested.per_track_nested`):
    they differ only in which engine `fit_batch` calls and which columns it
    returns, never in how tracks are selected, grouped, or concatenated.

    `progress`, if given, is called as `progress(done, total)` in tracks --
    once before the first batch and after every batch -- for a caller that
    cannot read tqdm's terminal output (a GUI progress bar). It is called
    from whatever thread runs this loop.
    """
    eligible = tracks.filter(pl.col("track_length") >= min_track_length)
    batches = _batches(eligible, max_batch_size)

    chunks = []
    total = eligible["track_id"].n_unique()
    bar = tqdm(total=total, desc=desc, unit="track", disable=not show_progress)
    done = 0
    if progress is not None:
        progress(done, total)
    for batch in batches:
        bar.set_postfix(track_length=batch.track_length, n_tracks=batch.n_tracks)
        chunks.append(fit_batch(batch))
        bar.update(batch.n_tracks)
        done += batch.n_tracks
        if progress is not None:
            progress(done, total)
    bar.close()

    return pl.concat(chunks).sort("track_id")


def _map_chunk(fit: MAPFit, batch: _Batch, param_names: list[str]) -> pl.DataFrame:
    """One `fit_map` result -> a table chunk.

    Any `name` that is itself a sample site with positive-real (LogNormal)
    support -- true of D, K, sigma, identifiable because
    `fit.unconstrained_params` has a matching key -- is reported from its
    log-space Laplace fit as an asymmetric back-transformed interval
    (`{name}_median`/`_lo`/`_hi` physical, `log10_{name}`/`_stderr`), per
    FINDINGS.md ("D should be reported in log-space"): pushing a wide
    log-space Gaussian through `exp()` and quoting mean +/- stderr in linear
    units understates the skew and can produce an interval touching zero for
    a strictly positive quantity.

    Everything else (e.g. `alpha`, a deterministic site derived from a
    Beta-distributed one, with no log-shaped unconstrained form) keeps the
    physical-space MAP +/- symmetric Laplace stderr, adequate for alpha per
    FINDINGS.md.
    """
    row = {**batch.index(), "converged": [fit.converged] * batch.n_tracks}
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


def fit_table_map(
    tracks: pl.DataFrame,
    model_fn: Callable,
    dt_s: float,
    prior_fn: Callable[[np.ndarray, np.ndarray], object],
    param_names: list[str],
    min_track_length: int = 10,
    max_batch_size: int = 20,
    seed: int = 0,
    show_progress: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """Batched exact MAP (L-BFGS-B) for every eligible track -- production.

    FINDINGS.md ("Inference-engine choice for production") documents this as
    more accurate and better-calibrated than `fit_table_svi`.

    `max_batch_size` exists because `fit_map`'s Hessian is dense over *all*
    free parameters in a call at once, so its cost is superlinear in track
    count and it OOM-crashes on a large group (measured: fine to a few dozen
    tracks, crashed at 140). Sub-batching trades a little amortization for a
    memory ceiling independent of how many tracks share a length.

    `converged` therefore reflects a whole sub-batch's joint optimization
    (one `scipy.optimize.minimize` call per sub-batch, not per track) -- keep
    `max_batch_size` modest if per-track granularity matters.
    """

    def fit_batch(batch: _Batch) -> pl.DataFrame:
        prior = prior_fn(batch.sigma_x_um, batch.sigma_y_um)
        fit = fit_map(
            model_fn,
            (jnp.asarray(batch.dx), jnp.asarray(batch.dy), dt_s, batch.n_disp, prior, batch.n_tracks),
            seed=seed,
        )
        return _map_chunk(fit, batch, param_names)

    return _per_track_table(
        tracks, fit_batch, "fit_table_map", min_track_length,
        max_batch_size=max_batch_size, show_progress=show_progress, progress=progress,
    )


def fit_table_svi(
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
    """Batched mean-field SVI for every eligible track: `{name}`/`{name}_stderr`.

    Faster in aggregate than `fit_table_map` but materially worse calibrated
    (FINDINGS.md: reported uncertainty 3-10x too narrow, because the
    mean-field guide discards real posterior correlation between K,
    alpha and sigma). Kept as the validation/comparison engine the recovery
    scripts run against, not as a production path.

    `prior_fn(sigma_x_um, sigma_y_um)` receives the batch's whole
    localization-precision arrays and may return per-track (n_tracks,) prior
    fields (e.g. `sigma_prior_from_localization`) or plain floats shared
    across the batch (e.g. `WEAK_ANOMALOUS_PRIOR`) -- broadcasting handles
    either.

    No row is dropped or flagged for an implausible estimate the way the
    classic MSD fit's `D_negative`/`intercept_negative` flags are needed: D
    and alpha are structurally positive/bounded by each parameter's own
    support, not by a post-hoc check.
    """

    def fit_batch(batch: _Batch) -> pl.DataFrame:
        prior = prior_fn(batch.sigma_x_um, batch.sigma_y_um)
        medians, stderrs = fit_batch_svi(
            model_fn, jnp.asarray(batch.dx), jnp.asarray(batch.dy), dt_s, batch.n_disp,
            prior, batch.n_tracks, num_steps=num_steps, seed=seed,
        )
        row = batch.index()
        for name in param_names:
            row[name] = medians[name].tolist()
            row[f"{name}_stderr"] = stderrs[name].tolist()
        return pl.DataFrame(row)

    return _per_track_table(
        tracks, fit_batch, "fit_table_svi", min_track_length, show_progress=show_progress
    )


def fit_table_nuts(
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
    """Full-NUTS per-track table: median + `hpdi_prob` HPDI per parameter.

    For when the posterior's *shape*, not just its median, is what's needed
    -- a Gaussian (Laplace/mean-field) approximation is unreliable exactly
    where this matters, e.g. the anisotropic model's eps=0 boundary ridge
    (FINDINGS.md). Generic over any site name including `numpyro.deterministic`
    ones (`D_par`, `D_perp`), which appear in `mcmc.get_samples()` identically.
    """

    def fit_batch(batch: _Batch) -> pl.DataFrame:
        prior = prior_fn(batch.sigma_x_um, batch.sigma_y_um)
        samples, _ = sample_posterior(
            model_fn,
            (jnp.asarray(batch.dx), jnp.asarray(batch.dy), dt_s, batch.n_disp, prior, batch.n_tracks),
            num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, seed=seed,
        )
        row = batch.index()
        for name in param_names:
            draws = samples[name].reshape(-1, batch.n_tracks)
            lo, hi = hpdi(draws, hpdi_prob, axis=0)
            row[f"{name}_median"] = np.median(draws, axis=0).tolist()
            row[f"{name}_lo"] = np.asarray(lo).tolist()
            row[f"{name}_hi"] = np.asarray(hi).tolist()
        return pl.DataFrame(row)

    return _per_track_table(
        tracks, fit_batch, "fit_table_nuts", min_track_length, show_progress=show_progress
    )
