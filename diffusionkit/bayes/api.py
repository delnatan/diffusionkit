"""One-call entry point for the exact-likelihood pipeline: `fit_track`.

Low-data / interactive use: one track in, one `TrackFit` out (full NUTS
posterior). This hides the two things every call site otherwise has to
assemble and keep in sync by hand: which `model_fn` goes with which prior
dataclass, and which `param_names` that model exposes. No inference math
lives here -- this only wires together `inference.py`'s `sample_posterior`
and `model.py`'s normal/anomalous models.

`prior=None` means "build an informative prior from this track's own
measured localization precision" (`priors.sigma_prior_from_localization`)
-- the honest default for the low-data regime. Passing a concrete
`NormalModelPrior`/`AnomalousModelPrior` instance (e.g. `WEAK_ANOMALOUS_PRIOR`)
opts out and fits against that fixed prior instead.

Bulk per-track diffusivity estimation is `diffusionkit.classic.posterior`'s
job (the grid posterior over D), not this module's -- NUTS here is a
per-track diagnostic tool, not a production population pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
import polars as pl
from numpyro.diagnostics import hpdi

from .inference import sample_posterior
from .model import (
    anomalous_diffusion_model,
    batched_anomalous_diffusion_model,
    batched_normal_diffusion_model,
    normal_diffusion_model,
)
from .priors import AnomalousModelPrior, NormalModelPrior, sigma_prior_from_localization

# name -> (single-track model, batched model, prior class, param_names)
MODEL_REGISTRY: dict[str, tuple] = {
    "normal": (normal_diffusion_model, batched_normal_diffusion_model, NormalModelPrior, ["D", "sigma"]),
    "anomalous": (anomalous_diffusion_model, batched_anomalous_diffusion_model,
                  AnomalousModelPrior, ["K", "sigma", "alpha"]),
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
    """One track's full-NUTS fit."""

    track_id: int
    track_length: int
    n_disp: int
    model: str  # "normal" | "anomalous"
    params: dict[str, float]  # posterior median, physical units, plain param names
    lo: dict[str, float]  # hpdi_prob HPDI
    hi: dict[str, float]
    raw: object  # (samples, MCMC) -- escape hatch for bayes.viz


def fit_track(
    track: pl.DataFrame,
    dt_s: float,
    model: Literal["normal", "anomalous"] = "anomalous",
    prior: NormalModelPrior | AnomalousModelPrior | None = None,
    hpdi_prob: float = 0.9,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
) -> TrackFit:
    """Full NUTS posterior for one track (rows for a single track_id, schema
    from `io.load_tracks`) -- the per-track diagnostic tool for when a
    posterior's shape matters, not just its center. `.raw` holds the full
    `(samples, mcmc)` for direct inspection (`bayes.viz.plot_posterior_corner`
    etc.).
    """
    model_fn, _, prior_cls, param_names = MODEL_REGISTRY[model]
    dx, dy, xstd, ystd = _track_arrays(track)
    n_disp = dx.shape[0]
    p = prior if prior is not None else _default_prior(prior_cls, xstd, ystd)
    track_id = int(track["track_id"][0])
    track_length = int(track["track_length"][0])

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

    return TrackFit(
        track_id=track_id, track_length=track_length, n_disp=n_disp,
        model=model, params=params, lo=lo, hi=hi, raw=(samples, mcmc),
    )
