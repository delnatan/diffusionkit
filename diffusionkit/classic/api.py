"""One-call bulk entry point for the classic MSD pipeline.

`fit_population` is a pure repackaging of the call sequence every classic-
pipeline script already runs by hand (`compute_all_tamsd` ->
`ensemble_average_msd` -> `n_fit_points` -> `fit_normal_diffusion`/
`fit_anomalous_diffusion` on the ensemble curve -> `localization_offset_by_track`
-> `weighted_expected_offset` -> `fit_all_tracks` for the per-track table)
into one call and one result object -- no new math, no changed defaults.

Named to mirror `bayes.fit_population` for the same "bulk regime" concept
(hundreds-to-thousands of tracks). Deliberately not named `fit_all_tracks`:
that name already means two different-signature things across
`analysis.fit_all_tracks` (this module) and `bayes.fit_all_tracks` -- both
stay as they are, this is a new top-level name, not a third meaning for an
existing one.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .fitting import (
    AnomalousDiffusionFit,
    NormalDiffusionFit,
    fit_all_tracks,
    fit_anomalous_diffusion,
    fit_normal_diffusion,
    localization_offset_by_track,
    n_fit_points,
    weighted_expected_offset,
)
from .msd import compute_all_tamsd, ensemble_average_msd


@dataclass(frozen=True)
class PopulationFit:
    """Everything `fit_population` computes, plain fields (dot access)."""

    tamsd: pl.DataFrame
    ensemble: pl.DataFrame
    ensemble_n_points_used: int  # lags actually used for both ensemble-curve fits, see n_fit_points
    ensemble_normal_fit: NormalDiffusionFit
    ensemble_anomalous_fit: AnomalousDiffusionFit
    per_track: pl.DataFrame
    mean_localization_offset_um2: float


def fit_population(
    tracks: pl.DataFrame,
    dt_s: float,
    min_track_length: int = 10,
    frac_points: float = 0.25,
    min_points: int = 3,
    max_points: int = 10,
    min_tracks_for_ensemble: int = 10,
) -> PopulationFit:
    """Classic MSD pipeline, bulk regime: TAMSD -> ensemble fit -> per-track
    fits, for every track in `tracks` (schema from `io.load_tracks`).

    `frac_points`/`min_points`/`max_points` control both the ensemble-curve
    fit range and each per-track fit range (`fitting.n_fit_points`'s rule --
    see its docstring for why both the fraction and the cap matter).
    `min_track_length` gates which tracks enter the per-track table;
    `min_tracks_for_ensemble` gates which lags survive into the ensemble
    curve. Defaults match what `scripts/run_msd_analysis.py` has always used.
    """
    tamsd = compute_all_tamsd(tracks, dt_s=dt_s)
    ensemble = ensemble_average_msd(tamsd, min_tracks=min_tracks_for_ensemble)
    loc_offset = localization_offset_by_track(tracks)

    tau = ensemble["tau_s"].to_numpy()
    msd = ensemble["msd_um2"].to_numpy()
    n_pairs = ensemble["n_pairs_total"].to_numpy()
    npts = n_fit_points(
        len(tau), frac=frac_points, min_points=min_points, max_points=max_points
    )

    ensemble_normal_fit = fit_normal_diffusion(tau, msd, npts, weights=n_pairs)
    ensemble_anomalous_fit = fit_anomalous_diffusion(tau, msd, npts)
    mean_offset = weighted_expected_offset(tamsd, loc_offset, n_points=npts)

    per_track = fit_all_tracks(
        tamsd,
        min_track_length=min_track_length,
        frac_points=frac_points,
        min_points=min_points,
        max_points=max_points,
        localization_offset=loc_offset,
    ).join(loc_offset, on="track_id", how="left").sort("track_id")

    return PopulationFit(
        tamsd=tamsd,
        ensemble=ensemble,
        ensemble_n_points_used=npts,
        ensemble_normal_fit=ensemble_normal_fit,
        ensemble_anomalous_fit=ensemble_anomalous_fit,
        per_track=per_track,
        mean_localization_offset_um2=mean_offset,
    )
