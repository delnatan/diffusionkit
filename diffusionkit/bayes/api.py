"""One-call entry points for the exact-likelihood pipeline, split by regime.

`fit_track` -- low-data / interactive: one track in, one `TrackFit` out.
`fit_population` -- bulk (hundreds-to-thousands of tracks): one table out.

Both hide the same three things every call site otherwise has to assemble
and keep in sync by hand (see `scripts/run_bayes_analysis.py` before this
module existed): which `model_fn` goes with which prior dataclass, which
`param_names` that model exposes, and the project's unit-suffixed column
renaming for the bulk table. No inference math lives here -- this only
wires together `inference.py`'s `fit_map`/`sample_posterior`/`fit_table_map`
and `model.py`'s normal/anomalous models.

`prior=None` on either function means "build an informative prior from this
track's (or this track-length-group's) own measured localization precision"
(`priors.sigma_prior_from_localization`) -- the honest default for the
low-data regime the project's guidance singles out for a Bayesian-first
approach. Passing a concrete `NormalModelPrior`/`AnomalousModelPrior`
instance (e.g. `WEAK_ANOMALOUS_PRIOR`) opts out and fits every track against
that same fixed prior instead.

Anisotropy is deliberately not a third `model=` option here -- see
`bayes/anisotropy.py`'s module docstring for why it gets its own entry point.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
import polars as pl
from numpyro.diagnostics import hpdi

from .inference import fit_map, fit_table_map, sample_posterior
from .model import (
    anomalous_diffusion_model,
    batched_anomalous_diffusion_model,
    batched_normal_diffusion_model,
    normal_diffusion_model,
)
from .priors import AnomalousModelPrior, NormalModelPrior, sigma_prior_from_localization

# Column names match the project's unit-suffixed convention (see README's
# "Results tables" reference) -- moved here verbatim from what used to be
# NORMAL_RENAME/ANOM_RENAME in scripts/run_bayes_analysis.py, so
# fit_population's output is byte-for-byte the same schema that script
# already produced and README already documents.
_NORMAL_RENAME = {
    "converged": "normal_converged",
    "D_median": "D_median_um2_s",
    "D_lo": "D_lo_um2_s",
    "D_hi": "D_hi_um2_s",
    "sigma_median": "sigma_normal_median_um",
    "sigma_lo": "sigma_normal_lo_um",
    "sigma_hi": "sigma_normal_hi_um",
    "log10_sigma": "log10_sigma_normal",
    "log10_sigma_stderr": "log10_sigma_normal_stderr",
}
_ANOM_RENAME = {
    "converged": "anomalous_converged",
    "D_alpha_median": "D_alpha_median_um2_s_alpha",
    "D_alpha_lo": "D_alpha_lo_um2_s_alpha",
    "D_alpha_hi": "D_alpha_hi_um2_s_alpha",
    "sigma_median": "sigma_anom_median_um",
    "sigma_lo": "sigma_anom_lo_um",
    "sigma_hi": "sigma_anom_hi_um",
    "log10_sigma": "log10_sigma_anom",
    "log10_sigma_stderr": "log10_sigma_anom_stderr",
}

# name -> (single-track model, batched model, prior class, param_names, bulk-table rename)
MODEL_REGISTRY: dict[str, tuple] = {
    "normal": (
        normal_diffusion_model,
        batched_normal_diffusion_model,
        NormalModelPrior,
        ["D", "sigma"],
        _NORMAL_RENAME,
    ),
    "anomalous": (
        anomalous_diffusion_model,
        batched_anomalous_diffusion_model,
        AnomalousModelPrior,
        ["D_alpha", "sigma", "alpha"],
        _ANOM_RENAME,
    ),
}


def _default_prior(prior_cls: type, sigma_x_um: np.ndarray, sigma_y_um: np.ndarray):
    mean, sd = sigma_prior_from_localization(sigma_x_um, sigma_y_um)
    return prior_cls(log_sigma_mean=mean, log_sigma_sd=sd)


def _track_arrays(track: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = track.sort("frame")
    dx = np.diff(g["x_um"].to_numpy())
    dy = np.diff(g["y_um"].to_numpy())
    return dx, dy, g["sigma_x_um"].to_numpy(), g["sigma_y_um"].to_numpy()


@dataclass(frozen=True)
class TrackFit:
    """One track's fit, any method -- same field shapes regardless of
    whether `method="map"` (Laplace) or `"nuts"` (full posterior) produced
    them, so downstream code doesn't need to branch on how the fit was done.
    """

    track_id: int
    track_length: int
    n_disp: int
    model: str  # "normal" | "anomalous"
    method: str  # "map" | "nuts"
    params: dict[str, float]  # median (nuts) / MAP (map), physical units, plain param names
    lo: dict[str, float]  # map: log-space asymmetric interval back-transformed (LogNormal
    hi: dict[str, float]  # sites) or symmetric Laplace (alpha); nuts: hpdi_prob HPDI
    raw: object  # MAPFit (method="map") or (samples, MCMC) (method="nuts") -- escape hatch


def fit_track(
    track: pl.DataFrame,
    dt_s: float,
    model: Literal["normal", "anomalous"] = "anomalous",
    prior: NormalModelPrior | AnomalousModelPrior | None = None,
    method: Literal["map", "nuts"] = "map",
    hpdi_prob: float = 0.9,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
) -> TrackFit:
    """Fit one track (rows for a single track_id, schema from `io.load_tracks`).

    `method="map"` (default) is `inference.fit_map` -- fast, and FINDINGS.md
    ("D's posterior is much better-behaved than D_alpha on short tracks")
    found it adequately calibrated for D specifically even at N=5. Reach for
    `method="nuts"` when the posterior's *shape* matters, not just its
    center -- e.g. a very short or weakly-identified track where a Gaussian
    (Laplace) approximation is suspect; `.raw` then holds the full
    `(samples, mcmc)` for direct inspection (`bayes.viz.plot_posterior_corner`
    etc.).
    """
    model_fn, _, prior_cls, param_names, _ = MODEL_REGISTRY[model]
    dx, dy, xstd, ystd = _track_arrays(track)
    n_disp = dx.shape[0]
    p = prior if prior is not None else _default_prior(prior_cls, xstd, ystd)
    track_id = int(track["track_id"][0])
    track_length = int(track["track_length"][0])

    if method == "map":
        fit = fit_map(model_fn, (jnp.asarray(dx), jnp.asarray(dy), dt_s, n_disp, p), seed=seed)
        params, lo, hi = {}, {}, {}
        for name in param_names:
            if name in fit.unconstrained_params:
                log_mean = float(np.asarray(fit.unconstrained_params[name]))
                log_stderr = float(np.asarray(fit.unconstrained_stderr[name]))
                params[name] = float(np.exp(log_mean))
                lo[name] = float(np.exp(log_mean - log_stderr))
                hi[name] = float(np.exp(log_mean + log_stderr))
            else:
                mean = float(np.asarray(fit.params[name]))
                stderr = float(np.asarray(fit.stderr[name]))
                params[name] = mean
                lo[name] = mean - stderr
                hi[name] = mean + stderr
        raw = fit
    elif method == "nuts":
        samples, mcmc = sample_posterior(
            model_fn, (jnp.asarray(dx), jnp.asarray(dy), dt_s, n_disp, p),
            num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, seed=seed,
        )
        params, lo, hi = {}, {}, {}
        for name in param_names:
            arr = samples[name].reshape(-1)
            params[name] = float(np.median(arr))
            lo_v, hi_v = hpdi(arr, hpdi_prob)
            lo[name], hi[name] = float(lo_v), float(hi_v)
        raw = (samples, mcmc)
    else:
        raise ValueError(f"method must be 'map' or 'nuts', got {method!r}")

    return TrackFit(
        track_id=track_id, track_length=track_length, n_disp=n_disp,
        model=model, method=method, params=params, lo=lo, hi=hi, raw=raw,
    )


def fit_population(
    tracks: pl.DataFrame,
    dt_s: float,
    model: Literal["normal", "anomalous", "both"] = "anomalous",
    prior: NormalModelPrior | AnomalousModelPrior | None = None,
    min_track_length: int = 10,
    max_batch_size: int = 20,
    seed: int = 0,
    show_progress: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """Fit every eligible track in `tracks`, population regime.

    Runs `inference.fit_table_map` (batched exact MAP), FINDINGS.md's
    production choice for accuracy and calibration. `inference.fit_table_svi`
    is faster in aggregate but reports uncertainty 3-10x too narrow, so it is
    not offered here -- call it directly if you want the comparison the
    validation scripts make.

    `model="both"` fits the normal and anomalous models and inner-joins them
    on `(track_id, track_length, n_disp)`: primary D from the normal model,
    primary alpha from the anomalous one (see TABLES.md).

    Columns follow the project's unit-suffixed median/asymmetric-interval
    convention (`D_median_um2_s`/`D_lo_um2_s`/`D_hi_um2_s`/`log10_D`/...).

    `progress(done, total)`, if given, reports tracks fitted so far (see
    `inference._per_track_table`); with `model="both"` it counts through
    each model's pass in turn.
    """
    if model == "both":
        if prior is not None:
            raise ValueError(
                "prior= cannot be used with model='both': the normal and anomalous models "
                "take different prior types. Fit each model separately, or pass prior=None "
                "to build an informative per-track prior for both."
            )
        normal = fit_population(
            tracks, dt_s, model="normal", min_track_length=min_track_length,
            max_batch_size=max_batch_size, seed=seed, show_progress=show_progress,
            progress=progress,
        )
        anomalous = fit_population(
            tracks, dt_s, model="anomalous", min_track_length=min_track_length,
            max_batch_size=max_batch_size, seed=seed, show_progress=show_progress,
            progress=progress,
        )
        return normal.join(anomalous, on=["track_id", "track_length", "n_disp"], how="inner")

    _, batched_model_fn, prior_cls, param_names, rename = MODEL_REGISTRY[model]

    if prior is not None:
        def prior_fn(xstd_um: np.ndarray, ystd_um: np.ndarray, _p=prior):
            return _p
    else:
        def prior_fn(xstd_um: np.ndarray, ystd_um: np.ndarray, _cls=prior_cls):
            return _default_prior(_cls, xstd_um, ystd_um)

    table = fit_table_map(
        tracks, batched_model_fn, dt_s, prior_fn, param_names,
        min_track_length=min_track_length, max_batch_size=max_batch_size,
        seed=seed, show_progress=show_progress, progress=progress,
    )
    return table.rename(rename)
