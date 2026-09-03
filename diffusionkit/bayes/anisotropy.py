"""Anisotropy-detection workflow: everything specific to
`model.anisotropic_diffusion_model` / `bayes_factor.py`, kept out of the
bulk `bayes` namespace (`api.fit_track`/`fit_population`) on purpose.

This targets short (N=5-10) tracks specifically, is a model-*comparison*
question ("is this more anisotropic than free diffusion at this track
length") rather than a per-track point estimate (see `priors.
AnisotropicModelPrior` and FINDINGS.md, "Anisotropy detection", for why),
and each dataset's orientation is arbitrary (2D acquisition at an unknown
mounting angle) -- a genuinely different usage pattern from the per-track
table `fit_population` produces for the bulk regime, not just a third
`model=` string on the same call. Hence its own module: `import
bayes.anisotropy as anisotropy`, not a name buried in `from bayes import *`.

`analyze` wraps the per-track log Bayes factor
(`bayes_factor.per_track_log_bayes_factor`) together with the descriptive
`eps`/`psi` posterior (`inference.fit_table_nuts` on the batched
anisotropic model) into one call -- see FINDINGS.md for why both are
reported and why only the Bayes factor (not `eps` alone) is the detector.
`null_calibration` is the matched-composition Monte Carlo check for turning
an ensemble `sum_log_bf10` into a p-value; kept as an explicit, separate
call (not part of `analyze`'s default path) since it costs real time
(O(100) simulated replicate datasets) and is inherently about validating a
specific `analyze()` result, not something to run unconditionally.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import polars as pl

from .bayes_factor import (
    aggregate_log_bayes_factor,
    batched_log_bayes_factor_anisotropy,
    log_bayes_factor_anisotropy,
    per_track_log_bayes_factor,
)
from .inference import fit_table_nuts
from .model import anisotropic_diffusion_model, batched_anisotropic_diffusion_model
from .priors import WEAK_ANISOTROPIC_PRIOR, AnisotropicModelPrior
from .simulate import simulate_anisotropic_tracks
from .viz import (
    plot_eps_forest,
    plot_eps_vs_log_bf,
    plot_log_bf_distribution,
    plot_spatial_map,
    plot_trajectory_gallery,
)

__all__ = [
    "AnisotropicModelPrior",
    "WEAK_ANISOTROPIC_PRIOR",
    "anisotropic_diffusion_model",
    "batched_anisotropic_diffusion_model",
    "simulate_anisotropic_tracks",
    "log_bayes_factor_anisotropy",
    "batched_log_bayes_factor_anisotropy",
    "per_track_log_bayes_factor",
    "aggregate_log_bayes_factor",
    "plot_log_bf_distribution",
    "plot_trajectory_gallery",
    "plot_spatial_map",
    "plot_eps_vs_log_bf",
    "plot_eps_forest",
    "AnisotropyResult",
    "analyze",
    "null_calibration",
]

# eps/psi posterior fields fit alongside log_bf10 -- D_par/D_perp are
# numpyro.deterministic sites derived from D_mean/eps, included since they're
# often the more directly interpretable pair (FINDINGS.md, README's
# per_track_master.csv reference).
_EPS_PARAM_NAMES = ["D_mean", "eps", "psi", "D_par", "D_perp"]
_EPS_RENAME = {
    "D_mean_median": "D_mean_median_um2_s", "D_mean_lo": "D_mean_lo_um2_s", "D_mean_hi": "D_mean_hi_um2_s",
    "D_par_median": "D_par_median_um2_s", "D_par_lo": "D_par_lo_um2_s", "D_par_hi": "D_par_hi_um2_s",
    "D_perp_median": "D_perp_median_um2_s", "D_perp_lo": "D_perp_lo_um2_s", "D_perp_hi": "D_perp_hi_um2_s",
    "psi_median": "psi_median_rad", "psi_lo": "psi_lo_rad", "psi_hi": "psi_hi_rad",
}


@dataclass(frozen=True)
class AnisotropyResult:
    per_track: pl.DataFrame  # per track: log_bf10 (the detector) + eps/psi posterior (descriptive) + geometry
    ensemble: pl.DataFrame  # aggregate_log_bayes_factor output, grouped by label_col (or one "all" group)


def analyze(
    tracks: pl.DataFrame,
    dt_s: float,
    prior: AnisotropicModelPrior | None = None,
    min_track_length: int = 5,
    max_track_length: int = 10,
    label_col: str | None = None,
    fit_eps_posterior: bool = True,
    n_mc: int = 20000,
    hpdi_prob: float = 0.9,
    seed: int = 0,
    show_progress: bool = True,
) -> AnisotropyResult:
    """Anisotropy Bayes-factor workflow for every track with
    `min_track_length <= track_length <= max_track_length` in `tracks`.

    `prior=None` uses `AnisotropicModelPrior()`'s shrunk-toward-isotropy
    default (see its docstring for why that shrinkage is deliberate at this
    N). `label_col`, if given, must already be a column on `tracks` (e.g. an
    experimental condition or a per-track spatial/structural label) --
    the ensemble sum is grouped by it; `None` sums everything into one
    "all_tracks" group. `fit_eps_posterior=False` skips the (slower, NUTS)
    per-track `eps`/`psi`/`D_mean` posterior and returns only `log_bf10` --
    useful when only the detector, not the descriptive interval, is needed.
    """
    p = prior if prior is not None else AnisotropicModelPrior()
    short = tracks.filter(
        (pl.col("track_length") >= min_track_length) & (pl.col("track_length") <= max_track_length)
    )

    per_track = per_track_log_bayes_factor(
        short, dt_s, p, min_track_length=min_track_length, n_mc=n_mc, seed=seed,
        show_progress=show_progress,
    )

    if fit_eps_posterior:
        eps_posterior = fit_table_nuts(
            short, batched_anisotropic_diffusion_model, dt_s, lambda xstd, ystd, _p=p: _p,
            param_names=_EPS_PARAM_NAMES, min_track_length=min_track_length, hpdi_prob=hpdi_prob,
            seed=seed, show_progress=show_progress,
        ).rename(_EPS_RENAME)
        geometry = short.group_by("track_id").agg(
            x_mean_um=pl.col("x_um").mean(), y_mean_um=pl.col("y_um").mean()
        )
        per_track = (
            per_track.join(eps_posterior, on=["track_id", "track_length", "n_disp"])
            .join(geometry, on="track_id")
            .sort("track_id")
        )
    else:
        per_track = per_track.sort("track_id")

    if label_col is not None:
        ensemble = aggregate_log_bayes_factor(per_track, label_col)
    else:
        ensemble = aggregate_log_bayes_factor(
            per_track.with_columns(all_tracks=pl.lit("all_tracks")), "all_tracks"
        )

    return AnisotropyResult(per_track=per_track, ensemble=ensemble)


def null_calibration(
    composition: dict[int, int],
    dt_s: float,
    prior: AnisotropicModelPrior,
    n_null: int = 200,
    n_mc: int = 20000,
    seed: int = 0,
) -> np.ndarray:
    """Empirical null distribution of the ensemble `sum_log_bf10`, simulated
    at the *exact* track-length composition given (not just a fixed track
    count) -- lets an observed ensemble sum be turned into a p-value against
    "what a matched-composition, genuinely isotropic population would give
    by chance," rather than read off the generic Jeffreys-scale buckets
    (which say nothing about a *specific* sample's sampling variance -- see
    FINDINGS.md's "Real-data anisotropy check").

    `composition`: `{track_length: n_tracks}`, e.g. from
    `tracks.group_by("track_length").agg(...)`. D_mean/sigma_loc for the
    simulation are taken from `prior`'s own central values (not fit from
    real data), so this stays a generic, reusable calibration. Returns one
    ensemble-sum value per of `n_null` replicate isotropic datasets.
    """
    D_mean = float(np.exp(prior.log_D_mean))
    sigma_loc = float(np.exp(prior.log_sigma_mean))
    rng = np.random.default_rng(seed)

    null_sums = np.empty(n_null)
    for rep in range(n_null):
        total = 0.0
        for track_length, n_tracks in composition.items():
            n_disp = track_length - 1
            sim = simulate_anisotropic_tracks(
                params=[(D_mean, 0.0, 0.0)], n_replicates=n_tracks, track_length=track_length,
                dt_s=dt_s, sigma_loc_um=sigma_loc, seed=int(rng.integers(0, 1_000_000)),
            )
            particles = sim["track_id"].unique().sort().to_list()
            dx = np.stack([np.diff(sim.filter(pl.col("track_id") == pid).sort("frame")["x_um"].to_numpy())
                            for pid in particles])
            dy = np.stack([np.diff(sim.filter(pl.col("track_id") == pid).sort("frame")["y_um"].to_numpy())
                            for pid in particles])
            logbf = np.asarray(batched_log_bayes_factor_anisotropy(
                jnp.asarray(dx), jnp.asarray(dy), dt_s, n_disp, prior, n_mc=n_mc, seed=0
            ))
            total += float(logbf.sum())
        null_sums[rep] = total
    return null_sums
