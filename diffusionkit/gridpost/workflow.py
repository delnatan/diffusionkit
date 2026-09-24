"""Functional single-track and table workflows for the D and alpha grid posteriors."""
from collections.abc import Callable

import numpy as np
import polars as pl

from ..data import Acquisition
from ..io import validate_table_schema, validated_track_frame
from ..validation import validate_acquisition
from . import posterior as posterior_mod
from . import posterior_alpha as posterior_alpha_mod
from .comparison import _motion_lrt
from .likelihood import _loglik, _prepared, _whiten
from .data import GridPosteriorAnalysis, GridPosteriors, GridPostOptions, PosteriorAlpha, PosteriorD, TrackPosterior

FIT_SCHEMA = {
    "track_id": pl.Int64, "n_frames": pl.Int64, "model": pl.String,
    "method": pl.String, "status": pl.String, "message": pl.String,
    "uncertainty_method": pl.String,
    **{name: pl.Float64 for name in PosteriorD.PARAMETERS},
    **{name: pl.Float64 for name in PosteriorAlpha.PARAMETERS},
}


def _edge_message(p: np.ndarray, options: GridPostOptions) -> str:
    """Why this D posterior's summary depends on the grid, or "" when it does not."""
    ratios = posterior_mod.edge_ratios(p)
    cut = [f"{name}={edge:g} um^2/s (edge/peak {r:.2g})"
           for name, edge, r in (("D_min_um2_s", options.D_min_um2_s, ratios[0]),
                                 ("D_max_um2_s", options.D_max_um2_s, ratios[1]))
           if r > posterior_mod.EDGE_RATIO_WARN]
    return "posterior cut by the grid edge at " + ", ".join(cut) if cut else ""


def _posterior_D_or_invalid(track: pl.DataFrame, acquisition: Acquisition,
                            options: GridPostOptions) -> tuple[PosteriorD, np.ndarray | None]:
    try:
        u = options.u_D()
        prior = posterior_mod.flat(u)
        w = _whiten(_prepared(track, acquisition), acquisition)
        lp = posterior_mod.log_posterior(_loglik(np.exp(u), w["lam"], w["y"][None], w["const"]), prior)
        motion_lrt = _motion_lrt(w["lam"], w["y"])
    except ValueError as exc:  # e.g. a zero localization SD
        return PosteriorD(dict.fromkeys(PosteriorD.PARAMETERS), "invalid_input", str(exc)), None
    p = np.exp(lp)
    s = posterior_mod.summary(p, u, options.level)
    message = _edge_message(p, options)
    if motion_lrt is None:
        message = "; ".join(filter(None, [message, "D_motion_lrt undefined for zero displacements"]))
    return PosteriorD({"D_post_median_um2_s": s["median"], "D_post_lo_um2_s": s["lo"],
                       "D_post_hi_um2_s": s["hi"],
                       "D_post_info_bits": posterior_mod.information_bits(lp, prior),
                       "D_motion_lrt": motion_lrt},
                      "ok", message), lp


def _posterior_alpha_or_invalid(track: pl.DataFrame, acquisition: Acquisition,
                                options: GridPostOptions) -> tuple[PosteriorAlpha, np.ndarray | None]:
    try:
        alphas, u = options.alphas(), options.u_K()
        lp = posterior_alpha_mod.log_alpha_posterior(
            posterior_alpha_mod.joint_loglik(track, acquisition, alphas, u), posterior_alpha_mod.flat_K(u))
    except ValueError as exc:  # e.g. a zero localization SD
        return PosteriorAlpha(dict.fromkeys(PosteriorAlpha.PARAMETERS), "invalid_input", str(exc)), None
    s = posterior_alpha_mod.summary(np.exp(lp), alphas, options.level)
    return PosteriorAlpha({"alpha_post_median": s["median"], "alpha_post_lo": s["lo"],
                           "alpha_post_hi": s["hi"]}, "ok", ""), lp


def analyze_track(track: pl.DataFrame, acquisition: Acquisition,
                  options: GridPostOptions = GridPostOptions()) -> TrackPosterior:
    """Grid posteriors over D and alpha for one track's rows, on `options`' grids.

    Invalid input raises; short tracks are explicit results. With
    exposure_s > 0 the D posterior (which models blur) is still computed;
    the alpha posterior (no blur model) is excluded, as it is with
    `options.compute_alpha=False`. An "ok" D posterior cut by a grid edge
    (`posterior.edge_ratios`) says so in its message. The result carries each
    "ok" posterior's normalized log weights (`log_post_D`, `log_post_alpha`).
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
    posterior_D, log_post_D = _posterior_D_or_invalid(track, acquisition, options)
    log_post_alpha = None
    if not options.compute_alpha:
        posterior_alpha = PosteriorAlpha(dict.fromkeys(PosteriorAlpha.PARAMETERS), "excluded",
                                         "The alpha posterior was not requested (compute_alpha=False)")
    elif acquisition.exposure_s > 0:
        posterior_alpha = PosteriorAlpha(dict.fromkeys(PosteriorAlpha.PARAMETERS), "excluded",
                                         "The alpha posterior has no exposure-blur model")
    else:
        posterior_alpha, log_post_alpha = _posterior_alpha_or_invalid(track, acquisition, options)
    return TrackPosterior(track_id, n, posterior_D, posterior_alpha, acquisition, options,
                          log_post_D, log_post_alpha)


def analyze_tracks(table: pl.DataFrame, acquisition: Acquisition,
                   options: GridPostOptions = GridPostOptions(), *,
                   progress: Callable[[int, int], None] | None = None,
                   keep_posteriors: bool = False) -> GridPosteriorAnalysis:
    """Two fit rows per input track (posterior_D, posterior_alpha), including
    exclusions and invalid tracks.

    With `keep_posteriors`, `result.posteriors` also holds every "ok" track's
    normalized log posterior on `options`' grids (`GridPosteriors`).

    Schema/configuration errors raise before work starts. Invalid individual
    tracks get status='invalid_input' and do not prevent other tracks fitting.
    Progress runs in the caller's thread; no global state or worker is created.
    """
    validate_acquisition(acquisition, allow_exposure=True)
    validate_table_schema(table)
    groups = table.sort("track_id", "frame").partition_by("track_id", maintain_order=True)
    fit_rows = []
    D_ids, log_post_D, alpha_ids, log_post_alpha = [], [], [], []
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
        if keep_posteriors and result.log_post_D is not None:
            D_ids.append(track_id)
            log_post_D.append(result.log_post_D)
        if keep_posteriors and result.log_post_alpha is not None:
            alpha_ids.append(track_id)
            log_post_alpha.append(result.log_post_alpha)
        if progress is not None:
            progress(done, len(groups))
    posteriors = None
    if keep_posteriors:
        posteriors = GridPosteriors(
            np.array(D_ids, dtype=np.int64), np.array(log_post_D).reshape(len(D_ids), options.n_D),
            np.array(alpha_ids, dtype=np.int64), np.array(log_post_alpha).reshape(len(alpha_ids), options.n_alpha))
    return GridPosteriorAnalysis(pl.DataFrame(fit_rows, schema=FIT_SCHEMA), acquisition, options, posteriors)
