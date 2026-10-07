"""Functional single-track and table workflows for the D grid posterior."""
import functools
from collections.abc import Callable, Iterable

import numpy as np
import polars as pl

from ..data import Acquisition
from ..io import validate_table_schema, validated_track_frame
from ..validation import validate_acquisition
from . import posterior as posterior_mod
from .likelihood import _loglik, _prepared, _whiten
from .data import GridLikelihoods, GridPosteriorAnalysis, GridPostOptions, PosteriorD, TrackPosterior

FIT_SCHEMA = {
    "track_id": pl.Int64, "n_frames": pl.Int64, "model": pl.String,
    "method": pl.String, "status": pl.String, "message": pl.String,
    "uncertainty_method": pl.String,
    **{name: pl.Float64 for name in PosteriorD.PARAMETERS},
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
        # Flat prior: the normalized log posterior is the normalized log-likelihood, kept as `loglik_D`.
        lp = posterior_mod.log_posterior(_loglik(np.exp(u), w["lam"], w["y"][None], w["const"]), prior)
    except ValueError as exc:  # e.g. a zero localization SD
        return PosteriorD(dict.fromkeys(PosteriorD.PARAMETERS), "invalid_input", str(exc)), None
    p = np.exp(lp)
    s = posterior_mod.summary(p, u, options.level)
    return PosteriorD({"D_post_median_um2_s": s["median"], "D_post_lo_um2_s": s["lo"],
                       "D_post_hi_um2_s": s["hi"],
                       "D_post_info_bits": posterior_mod.information_bits(lp, prior),
                       "D_floor_um2_s": posterior_mod.localization_floor(track, acquisition)},
                      "ok", _edge_message(p, options)), lp


def analyze_track(track: pl.DataFrame, acquisition: Acquisition,
                  options: GridPostOptions = GridPostOptions()) -> TrackPosterior:
    """The grid posterior over D for one track's rows, on `options.u_D()`.

    Invalid input raises; short tracks are explicit results. The posterior models
    `acquisition.exposure_s` as box-shutter blur. An "ok" posterior cut by a grid edge
    (`posterior.edge_ratios`) says so in its message. The result carries an "ok"
    track's normalized log-likelihood on the grid (`loglik_D`).
    """
    validate_acquisition(acquisition, allow_exposure=True)
    track = validated_track_frame(track, acquisition, require_localization=True)
    track_id = int(track["track_id"][0])
    n = track.height
    if n < options.min_frames:
        message = f"{n} frames; min_frames={options.min_frames}"
        return TrackPosterior(track_id, n,
                              PosteriorD(dict.fromkeys(PosteriorD.PARAMETERS), "excluded", message),
                              acquisition, options)
    posterior_D, loglik_D = _posterior_D_or_invalid(track, acquisition, options)
    return TrackPosterior(track_id, n, posterior_D, acquisition, options, loglik_D)


def _analyze_group(group: pl.DataFrame, acquisition: Acquisition, options: GridPostOptions) -> TrackPosterior:
    """`analyze_track`, with an invalid track as an `invalid_input` result rather than a raise."""
    try:
        return analyze_track(group, acquisition, options)
    except ValueError as exc:
        message = str(exc)
        return TrackPosterior(int(group["track_id"][0]), group.height,
                              PosteriorD(dict.fromkeys(PosteriorD.PARAMETERS), "invalid_input", message),
                              acquisition, options)


def analyze_tracks(table: pl.DataFrame, acquisition: Acquisition,
                   options: GridPostOptions = GridPostOptions(), *,
                   progress: Callable[[int, int], None] | None = None,
                   keep_likelihoods: bool = False,
                   map_fn: Callable[..., Iterable] = map) -> GridPosteriorAnalysis:
    """One fit row per input track (model posterior_D), including exclusions and invalid tracks.

    With `keep_likelihoods`, `result.likelihoods` also holds every "ok" track's
    normalized log-likelihood on `options.u_D()` (`GridLikelihoods`): what tracks
    are combined from (`fit_shared_D`, `fit_lognormal`, `deconvolve`).

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
    D_ids, D_frames, loglik_D = [], [], []
    if progress is not None:
        progress(0, len(groups))
    results = map_fn(functools.partial(_analyze_group, acquisition=acquisition, options=options), groups)
    for done, result in enumerate(results, 1):
        track_id = result.track_id
        post = result.posterior_D
        fit_rows.append({"track_id": track_id, "n_frames": result.n_frames, "model": post.model,
                         "method": post.method, "status": post.status, "message": post.message,
                         "uncertainty_method": post.uncertainty_method, **post.parameters})
        if keep_likelihoods and result.loglik_D is not None:
            D_ids.append(track_id)
            D_frames.append(result.n_frames)
            loglik_D.append(result.loglik_D)
        if progress is not None:
            progress(done, len(groups))
    likelihoods = None
    if keep_likelihoods:
        likelihoods = GridLikelihoods(np.array(D_ids, dtype=np.int64), np.array(D_frames, dtype=np.int64),
                                      np.array(loglik_D).reshape(len(D_ids), options.n_D))
    return GridPosteriorAnalysis(pl.DataFrame(fit_rows, schema=FIT_SCHEMA), acquisition, options, likelihoods)
