"""Fit `model.py`'s numpyro models.

One inference engine, not a separate MLE/Bayes split: every function here is
generic over *any* numpyro model and prior, so `priors.WEAK_*_PRIOR` makes
them behave like a flat-prior MLE and an informative prior (e.g. per-track,
from `sigma_prior_from_localization`) makes them a proper Bayesian fit.
Exactly one code path either way.

Two single-call engines, each fitting one batch:

  `sample_posterior` full NUTS -- the per-track diagnostic tool, when the
                     posterior's shape (not just its center) matters.
  `fit_batch_svi`    mean-field (`AutoNormal`) SVI -- validation/comparison
                     only (see `fit_table_svi`).

and two per-track table builders wrapping them -- `fit_table_svi`
(validation/comparison) and `fit_table_nuts` (when posterior shape matters).
Both share `_per_track_table`, which groups tracks by shared track_length
and loops: fitting tracks one Python-level call at a time pays a fresh JAX
trace per call regardless of shape reuse, so batching a whole length-group
into one plated call is what makes a full dataset tractable (see model.py,
and FINDINGS.md for measured numbers).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import polars as pl
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from tqdm import tqdm


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
    tracks go into one call, for an engine whose cost is superlinear in batch
    size; `None` means "the whole length-group".
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
    (`fit_table_svi`/`fit_table_nuts`): they differ only in which engine
    `fit_batch` calls and which columns it returns, never in how tracks are
    selected, grouped, or concatenated.

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

    Materially worse calibrated than full NUTS (FINDINGS.md: reported
    uncertainty 3-10x too narrow, because the mean-field guide discards real
    posterior correlation between K, alpha and sigma). Kept as the
    validation/comparison engine the recovery scripts run against, not as a
    production path.

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
    -- a mean-field (SVI) approximation is unreliable exactly where this
    matters, e.g. a short track's weakly-identified alpha. Generic over any
    site name including `numpyro.deterministic` ones, which appear in
    `mcmc.get_samples()` identically to sampled sites.
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
