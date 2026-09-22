"""Functional single-track and table MSD workflows; no ensemble or plotting side effects."""
from collections.abc import Callable

import polars as pl

from ..data import Acquisition
from ..io import validate_table_schema, validated_track_frame
from ..validation import validate_acquisition
from .analysis import compute_msd, validate_options
from .data import ClassicAnalysis, MSDOptions, TrackAnalysis
from .estimators import empty_fit, fit_anomalous_msd, fit_brownian_msd


FIT_SCHEMA = {
    "track_id": pl.Int64, "n_frames": pl.Int64, "model": pl.String,
    "method": pl.String, "status": pl.String, "message": pl.String,
    "D_um2_s": pl.Float64, "K_um2_s_alpha": pl.Float64, "alpha": pl.Float64,
    "n_lags": pl.Int64, "residual_sum_squares_um4": pl.Float64,
    "optimizer_status": pl.Int64, "nfev": pl.Int64,
    "localization": pl.String, "uncertainty_method": pl.String,
}
MSD_SCHEMA = {
    "track_id": pl.Int64, "lag": pl.Int64, "tau_s": pl.Float64,
    "n_pairs": pl.Int64, "msd_um2": pl.Float64,
    "localization_offset_um2": pl.Float64, "corrected_msd_um2": pl.Float64,
}


def analyze_track(track: pl.DataFrame, acquisition: Acquisition,
                  options: MSDOptions = MSDOptions()) -> TrackAnalysis:
    """Fit MSD D and alpha for one track's rows.

    Invalid input raises; short tracks are explicit results. `status="excluded"`
    when `exposure_s > 0`: the MSD estimators assume instantaneous positions,
    no motion-blur model.
    """
    validate_acquisition(acquisition, allow_exposure=True)
    validate_options(options)
    track = validated_track_frame(track, acquisition, require_localization=options.localization == "provided")
    track_id = int(track["track_id"][0])
    n = track.height
    if n < options.min_frames:
        message = f"{n} frames; min_frames={options.min_frames}"
        return TrackAnalysis(track_id, n, None,
                             empty_fit("brownian", "excluded", message),
                             empty_fit("power_law", "excluded", message), acquisition, options)
    if acquisition.exposure_s > 0:
        message = "MSD estimators assume exposure_s=0 (no motion-blur model)"
        return TrackAnalysis(track_id, n, None, empty_fit("brownian", "excluded", message),
                             empty_fit("power_law", "excluded", message), acquisition, options)
    msd = compute_msd(track, acquisition, options)
    return TrackAnalysis(track_id, n, msd, fit_brownian_msd(msd),
                         fit_anomalous_msd(msd, max_nfev=options.max_nfev), acquisition, options)


def analyze_tracks(table: pl.DataFrame, acquisition: Acquisition,
                   options: MSDOptions = MSDOptions(), *,
                   progress: Callable[[int, int], None] | None = None) -> ClassicAnalysis:
    """Two fit rows per input track, including exclusions and invalid tracks.

    Schema/configuration errors raise before work starts. Invalid individual
    tracks get status='invalid_input' and do not prevent other tracks fitting.
    Progress runs in the caller's thread; no global state or worker is created.
    """
    validate_acquisition(acquisition, allow_exposure=True)
    validate_options(options)
    validate_table_schema(table)
    groups = table.sort("track_id", "frame").partition_by("track_id", maintain_order=True)
    fit_rows, msd_rows = [], []
    if progress is not None:
        progress(0, len(groups))
    for done, group in enumerate(groups, 1):
        track_id = int(group["track_id"][0])
        try:
            result = analyze_track(group, acquisition, options)
        except ValueError as exc:
            result = TrackAnalysis(track_id, group.height, None,
                                   empty_fit("brownian", "invalid_input", str(exc)),
                                   empty_fit("power_law", "invalid_input", str(exc)), acquisition, options)
        for fit in (result.brownian, result.anomalous):
            fit_rows.append({"track_id": track_id, "n_frames": result.n_frames,
                             "model": fit.model, "method": fit.method, "status": fit.status,
                             "message": fit.message, "n_lags": fit.n_lags,
                             "residual_sum_squares_um4": fit.residual_sum_squares_um4,
                             "optimizer_status": fit.optimizer_status, "nfev": fit.nfev,
                             "localization": options.localization,
                             "uncertainty_method": fit.uncertainty_method, **fit.parameters})
        curve = result.msd
        if curve is not None:
            for i, lag in enumerate(curve.lag):
                msd_rows.append({"track_id": track_id, "lag": int(lag),
                                 "tau_s": float(curve.tau_s[i]), "n_pairs": int(curve.n_pairs[i]),
                                 "msd_um2": float(curve.msd_um2[i]),
                                 "localization_offset_um2": float(curve.localization_offset_um2[i]),
                                 "corrected_msd_um2": float(curve.msd_um2[i] - curve.localization_offset_um2[i])})
        if progress is not None:
            progress(done, len(groups))
    return ClassicAnalysis(pl.DataFrame(fit_rows, schema=FIT_SCHEMA),
                           pl.DataFrame(msd_rows, schema=MSD_SCHEMA), acquisition, options)
