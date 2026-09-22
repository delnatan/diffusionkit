"""Functional single-track and table workflows for the D and alpha grid posteriors."""
from collections.abc import Callable

import polars as pl

from ..data import Acquisition
from ..io import validate_table_schema, validated_track_frame
from ..validation import validate_acquisition
from . import posterior as posterior_mod
from . import posterior_alpha as posterior_alpha_mod
from .data import GridPosteriorAnalysis, GridPostOptions, PosteriorAlpha, PosteriorD, TrackPosterior

FIT_SCHEMA = {
    "track_id": pl.Int64, "n_frames": pl.Int64, "model": pl.String,
    "method": pl.String, "status": pl.String, "message": pl.String,
    "uncertainty_method": pl.String,
    **{name: pl.Float64 for name in PosteriorD.PARAMETERS},
    **{name: pl.Float64 for name in PosteriorAlpha.PARAMETERS},
}


def _posterior_D_or_invalid(track: pl.DataFrame, acquisition: Acquisition, options: GridPostOptions) -> PosteriorD:
    try:
        s = posterior_mod.track_posterior(track, acquisition, level=options.level)
        return PosteriorD({"D_post_median_um2_s": s["median"], "D_post_lo_um2_s": s["lo"],
                           "D_post_hi_um2_s": s["hi"]}, "ok", "")
    except ValueError as exc:  # e.g. a zero localization SD
        return PosteriorD(dict.fromkeys(PosteriorD.PARAMETERS), "invalid_input", str(exc))


def _posterior_alpha_or_invalid(track: pl.DataFrame, acquisition: Acquisition, options: GridPostOptions) -> PosteriorAlpha:
    try:
        s = posterior_alpha_mod.track_alpha_posterior(track, acquisition, level=options.level)
        return PosteriorAlpha({"alpha_post_median": s["median"], "alpha_post_lo": s["lo"],
                               "alpha_post_hi": s["hi"]}, "ok", "")
    except ValueError as exc:  # e.g. a zero localization SD
        return PosteriorAlpha(dict.fromkeys(PosteriorAlpha.PARAMETERS), "invalid_input", str(exc))


def analyze_track(track: pl.DataFrame, acquisition: Acquisition,
                  options: GridPostOptions = GridPostOptions()) -> TrackPosterior:
    """Grid posteriors over D and alpha for one track's rows.

    Invalid input raises; short tracks are explicit results. With
    exposure_s > 0 the D posterior (which models blur) is still computed;
    the alpha posterior (no blur model) is excluded.
    """
    validate_acquisition(acquisition, allow_exposure=True)
    track = validated_track_frame(track, acquisition, require_localization=True)
    track_id = int(track["track_id"][0])
    n = track.height
    if n < options.min_frames:
        message = f"{n} frames; min_frames={options.min_frames}"
        return TrackPosterior(track_id, n,
                              PosteriorD(dict.fromkeys(PosteriorD.PARAMETERS), "excluded", message),
                              PosteriorAlpha(dict.fromkeys(PosteriorAlpha.PARAMETERS), "excluded", message),
                              acquisition, options)
    posterior_D = _posterior_D_or_invalid(track, acquisition, options)
    if acquisition.exposure_s > 0:
        posterior_alpha = PosteriorAlpha(dict.fromkeys(PosteriorAlpha.PARAMETERS), "excluded",
                                         "The alpha posterior has no exposure-blur model")
    else:
        posterior_alpha = _posterior_alpha_or_invalid(track, acquisition, options)
    return TrackPosterior(track_id, n, posterior_D, posterior_alpha, acquisition, options)


def analyze_tracks(table: pl.DataFrame, acquisition: Acquisition,
                   options: GridPostOptions = GridPostOptions(), *,
                   progress: Callable[[int, int], None] | None = None) -> GridPosteriorAnalysis:
    """Two fit rows per input track (posterior_D, posterior_alpha), including
    exclusions and invalid tracks.

    Schema/configuration errors raise before work starts. Invalid individual
    tracks get status='invalid_input' and do not prevent other tracks fitting.
    Progress runs in the caller's thread; no global state or worker is created.
    """
    validate_acquisition(acquisition, allow_exposure=True)
    validate_table_schema(table)
    groups = table.sort("track_id", "frame").partition_by("track_id", maintain_order=True)
    fit_rows = []
    if progress is not None:
        progress(0, len(groups))
    for done, group in enumerate(groups, 1):
        track_id = int(group["track_id"][0])
        try:
            result = analyze_track(group, acquisition, options)
        except ValueError as exc:
            message = str(exc)
            result = TrackPosterior(track_id, group.height,
                                    PosteriorD(dict.fromkeys(PosteriorD.PARAMETERS), "invalid_input", message),
                                    PosteriorAlpha(dict.fromkeys(PosteriorAlpha.PARAMETERS), "invalid_input", message),
                                    acquisition, options)
        for post in (result.posterior_D, result.posterior_alpha):
            fit_rows.append({"track_id": track_id, "n_frames": result.n_frames, "model": post.model,
                             "method": post.method, "status": post.status, "message": post.message,
                             "uncertainty_method": post.uncertainty_method, **post.parameters})
        if progress is not None:
            progress(done, len(groups))
    return GridPosteriorAnalysis(pl.DataFrame(fit_rows, schema=FIT_SCHEMA), acquisition, options)
