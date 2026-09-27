"""Functional single-track and table workflows for the D and alpha grid posteriors."""
import functools
from collections.abc import Callable, Iterable

import numpy as np
import polars as pl

from ..data import Acquisition
from ..io import validate_table_schema, validated_track_frame
from ..validation import validate_acquisition
from . import posterior as posterior_mod
from . import posterior_alpha as posterior_alpha_mod
from . import timescale as timescale_mod
from .comparison import _motion_lrt
from .likelihood import _loglik, _prepared, _whiten
from .data import (GridPosteriorAnalysis, GridPosteriors, GridPostOptions, PosteriorAlpha, PosteriorD,
                   PosteriorDTimescale, TrackPosterior)

FIT_SCHEMA = {
    "track_id": pl.Int64, "n_frames": pl.Int64, "model": pl.String,
    "method": pl.String, "status": pl.String, "message": pl.String,
    "uncertainty_method": pl.String,
    **{name: pl.Float64 for name in PosteriorD.PARAMETERS},
    **{name: pl.Float64 for name in PosteriorAlpha.PARAMETERS},
    **{name: pl.Float64 for name in PosteriorDTimescale.PARAMETERS},
}


# `fits.method` for an alpha row, by the likelihood that produced it.
_ALPHA_METHOD_NAMES = {"exact": PosteriorAlpha.method, "whittle": PosteriorAlpha.method + "_whittle"}


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
        track = _prepared(track, acquisition)
        likelihood = options.alpha_likelihood(track.height)
        ll = posterior_alpha_mod._JOINT_LOGLIK[likelihood](track, acquisition, alphas, u)
        lp = posterior_alpha_mod.log_alpha_posterior(ll, posterior_alpha_mod.flat_K(u))
    except ValueError as exc:  # e.g. a zero localization SD
        return PosteriorAlpha(dict.fromkeys(PosteriorAlpha.PARAMETERS), "invalid_input", str(exc)), None
    s = posterior_alpha_mod.summary(np.exp(lp), alphas, options.level)
    return PosteriorAlpha({"alpha_post_median": s["median"], "alpha_post_lo": s["lo"],
                           "alpha_post_hi": s["hi"]}, "ok", "",
                          method=_ALPHA_METHOD_NAMES[likelihood]), lp


def _posterior_D_long(track: pl.DataFrame, acquisition: Acquisition, options: GridPostOptions,
                      posterior_D: PosteriorD, log_post_D: np.ndarray | None
                      ) -> tuple[PosteriorDTimescale, np.ndarray | None]:
    """D at tau = options.D_long_stride * dt and its ratio to D (`gridpost.timescale`)."""
    stride = options.D_long_stride
    empty = dict.fromkeys(PosteriorDTimescale.PARAMETERS)
    if log_post_D is None:
        return PosteriorDTimescale(empty, posterior_D.status, posterior_D.message), None
    u = options.u_D()
    ll, n_phases = timescale_mod.thinned_loglik(_prepared(track, acquisition), acquisition, u, stride,
                                                options.min_frames)
    if ll is None:
        needed = (options.min_frames - 1) * stride + 1
        return PosteriorDTimescale(empty, "excluded",
                                   f"{track.height} frames; D_long_stride={stride} needs >= {needed}"), None
    prior = posterior_mod.flat(u)
    lp = posterior_mod.log_posterior(ll, prior)
    p = np.exp(lp)
    s = posterior_mod.summary(p, u, options.level)
    r = timescale_mod.ratio_summary(*timescale_mod.ratio_posterior(lp, log_post_D, u), options.level)
    notes = [_edge_message(p, options)]
    if n_phases < stride:
        notes.append(f"{n_phases} of {stride} phases long enough")
    return PosteriorDTimescale({
        "tau_long_s": stride * float(acquisition.dt_s),
        "D_long_post_median_um2_s": s["median"], "D_long_post_lo_um2_s": s["lo"], "D_long_post_hi_um2_s": s["hi"],
        "D_ratio_post_median": r["median"], "D_ratio_post_lo": r["lo"], "D_ratio_post_hi": r["hi"],
        "P_D_decrease": r["p_decrease"]}, "ok", "; ".join(filter(None, notes))), lp


def analyze_track(track: pl.DataFrame, acquisition: Acquisition,
                  options: GridPostOptions = GridPostOptions()) -> TrackPosterior:
    """Grid posteriors over D and alpha for one track's rows, on `options`' grids.

    Invalid input raises; short tracks are explicit results. Both posteriors
    model `acquisition.exposure_s` as box-shutter blur; the alpha posterior is
    excluded only with `options.compute_alpha=False`. An "ok" D posterior cut by a grid edge
    (`posterior.edge_ratios`) says so in its message. The result carries each
    "ok" posterior's normalized log weights (`log_post_D`, `log_post_alpha`).
    """
    validate_acquisition(acquisition, allow_exposure=True)
    track = validated_track_frame(track, acquisition, require_localization=True)
    track_id = int(track["track_id"][0])
    n = track.height
    if n < options.min_frames:
        message = f"{n} frames; min_frames={options.min_frames}"
        D_long = (PosteriorDTimescale(dict.fromkeys(PosteriorDTimescale.PARAMETERS), "excluded", message)
                  if options.D_long_stride else None)
        return TrackPosterior(track_id, n,
                              PosteriorD(dict.fromkeys(PosteriorD.PARAMETERS), "excluded", message),
                              PosteriorAlpha(dict.fromkeys(PosteriorAlpha.PARAMETERS), "excluded", message),
                              acquisition, options, posterior_D_long=D_long)
    posterior_D, log_post_D = _posterior_D_or_invalid(track, acquisition, options)
    log_post_alpha = None
    if not options.compute_alpha:
        posterior_alpha = PosteriorAlpha(dict.fromkeys(PosteriorAlpha.PARAMETERS), "excluded",
                                         "The alpha posterior was not requested (compute_alpha=False)")
    else:
        posterior_alpha, log_post_alpha = _posterior_alpha_or_invalid(track, acquisition, options)
    posterior_D_long = log_post_D_long = None
    if options.D_long_stride:
        posterior_D_long, log_post_D_long = _posterior_D_long(track, acquisition, options, posterior_D, log_post_D)
    return TrackPosterior(track_id, n, posterior_D, posterior_alpha, acquisition, options,
                          log_post_D, log_post_alpha, posterior_D_long, log_post_D_long)


def _analyze_group(group: pl.DataFrame, acquisition: Acquisition, options: GridPostOptions) -> TrackPosterior:
    """`analyze_track`, with an invalid track as an `invalid_input` result rather than a raise."""
    try:
        return analyze_track(group, acquisition, options)
    except ValueError as exc:
        message = str(exc)
        D_long = (PosteriorDTimescale(dict.fromkeys(PosteriorDTimescale.PARAMETERS), "invalid_input", message)
                  if options.D_long_stride else None)
        return TrackPosterior(int(group["track_id"][0]), group.height,
                              PosteriorD(dict.fromkeys(PosteriorD.PARAMETERS), "invalid_input", message),
                              PosteriorAlpha(dict.fromkeys(PosteriorAlpha.PARAMETERS), "invalid_input", message),
                              acquisition, options, posterior_D_long=D_long)


def analyze_tracks(table: pl.DataFrame, acquisition: Acquisition,
                   options: GridPostOptions = GridPostOptions(), *,
                   progress: Callable[[int, int], None] | None = None,
                   keep_posteriors: bool = False,
                   map_fn: Callable[..., Iterable] = map) -> GridPosteriorAnalysis:
    """Two fit rows per input track (posterior_D, posterior_alpha), three with
    `options.D_long_stride` set (posterior_D_timescale), including exclusions
    and invalid tracks.

    With `keep_posteriors`, `result.posteriors` also holds every "ok" track's
    normalized log posterior on `options`' grids (`GridPosteriors`).

    Schema/configuration errors raise before work starts. Invalid individual
    tracks get status='invalid_input' and do not prevent other tracks fitting.
    Progress runs in the caller's thread; no global state or worker is created.

    `map_fn(func, groups)` runs the per-track work and must yield results in
    input order, as the builtin `map` (the default) and
    `concurrent.futures.Executor.map` do. Passing a `ThreadPoolExecutor`'s
    `map` fits tracks concurrently -- the per-track linear algebra releases the
    GIL -- with the pool created and shut down by the caller; the result is
    identical to the serial one.
    """
    validate_acquisition(acquisition, allow_exposure=True)
    validate_table_schema(table)
    groups = table.sort("track_id", "frame").partition_by("track_id", maintain_order=True)
    fit_rows = []
    D_ids, log_post_D, alpha_ids, log_post_alpha, long_ids, log_post_long = [], [], [], [], [], []
    if progress is not None:
        progress(0, len(groups))
    results = map_fn(functools.partial(_analyze_group, acquisition=acquisition, options=options), groups)
    for done, result in enumerate(results, 1):
        track_id = result.track_id
        for post in filter(None, (result.posterior_D, result.posterior_alpha, result.posterior_D_long)):
            fit_rows.append({"track_id": track_id, "n_frames": result.n_frames, "model": post.model,
                             "method": post.method, "status": post.status, "message": post.message,
                             "uncertainty_method": post.uncertainty_method, **post.parameters})
        if keep_posteriors and result.log_post_D is not None:
            D_ids.append(track_id)
            log_post_D.append(result.log_post_D)
        if keep_posteriors and result.log_post_alpha is not None:
            alpha_ids.append(track_id)
            log_post_alpha.append(result.log_post_alpha)
        if keep_posteriors and result.log_post_D_long is not None:
            long_ids.append(track_id)
            log_post_long.append(result.log_post_D_long)
        if progress is not None:
            progress(done, len(groups))
    posteriors = None
    if keep_posteriors:
        posteriors = GridPosteriors(
            np.array(D_ids, dtype=np.int64), np.array(log_post_D).reshape(len(D_ids), options.n_D),
            np.array(alpha_ids, dtype=np.int64), np.array(log_post_alpha).reshape(len(alpha_ids), options.n_alpha),
            *((np.array(long_ids, dtype=np.int64), np.array(log_post_long).reshape(len(long_ids), options.n_D))
              if options.D_long_stride else (None, None)))
    return GridPosteriorAnalysis(pl.DataFrame(fit_rows, schema=FIT_SCHEMA), acquisition, options, posteriors)
