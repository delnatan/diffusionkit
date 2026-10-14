"""Functional single-track and table workflows for the fBm (alpha, D) grid posterior."""
import functools
from collections.abc import Callable, Iterable

import numpy as np
import polars as pl

from ..data import Acquisition
from ..io import validate_table_schema, validated_track_frame
from ..validation import validate_acquisition
from . import posterior as P
from .data import (AlphaPosteriors, FbmPosteriorAnalysis, FbmPostOptions, JointGrid, JointLikelihoods, PosteriorAlphaD,
                   TrackPosteriorFbm)
from .population import joint_cells

FIT_SCHEMA = {
    "track_id": pl.Int64, "n_frames": pl.Int64, "model": pl.String,
    "method": pl.String, "status": pl.String, "message": pl.String,
    "uncertainty_method": pl.String,
    **{name: pl.Float64 for name in PosteriorAlphaD.PARAMETERS},
    "D_grid_edge": pl.String,
}

_EDGE_MESSAGE = {
    "low": "amplitude posterior cut by snr_min: the track only bounds D from above",
    "high": "amplitude posterior cut by snr_max: the track only bounds D from below",
    "both": "amplitude posterior cut by both grid edges: no information about D",
}


def _empty(status: str, message: str) -> PosteriorAlphaD:
    return PosteriorAlphaD(dict.fromkeys(PosteriorAlphaD.PARAMETERS), status, message)


def _posterior_or_invalid(track: pl.DataFrame, acquisition: Acquisition, options: FbmPostOptions,
                          joint: JointGrid | None) -> tuple[PosteriorAlphaD, np.ndarray | None, np.ndarray | None]:
    try:
        grid = P.log_likelihood_grid(track, acquisition, options)
    except ValueError as exc:  # e.g. a zero localization SD
        return _empty("invalid_input", str(exc)), None, None
    lp = P.log_posterior(grid.loglik)
    edge = P.snr_edge(lp)
    parameters = {**P.summary(lp, grid.ln_K, options),
                  "D_ref_floor_um2_s": P.localization_floor(track, options)}
    return (PosteriorAlphaD(parameters, "ok", _EDGE_MESSAGE.get(edge, ""), grid_edge=edge),
            P.alpha_log_marginal(lp), None if joint is None else joint_cells(grid, options, joint))


def analyze_track(track: pl.DataFrame, acquisition: Acquisition,
                  options: FbmPostOptions = FbmPostOptions(), joint: JointGrid | None = None) -> TrackPosteriorFbm:
    """The joint grid posterior over (alpha, D) for one track's rows, summarized per margin.

    Invalid input raises; short tracks are explicit results. The posterior models
    `acquisition.exposure_s` as box-shutter blur. An "ok" result carries the track's normalized
    marginal log posterior of alpha (`log_post_alpha`) and, given `joint`, its likelihood on those
    cells (`joint_loglik`, `population.joint_cells`).
    """
    validate_acquisition(acquisition, allow_exposure=True)
    track = validated_track_frame(track, acquisition, require_localization=True)
    track_id = int(track["track_id"][0])
    n = track.height
    if n < options.min_frames:
        return TrackPosteriorFbm(track_id, n, _empty("excluded", f"{n} frames; min_frames={options.min_frames}"),
                                 acquisition, options)
    posterior, log_post_alpha, joint_loglik = _posterior_or_invalid(track, acquisition, options, joint)
    return TrackPosteriorFbm(track_id, n, posterior, acquisition, options, log_post_alpha, joint_loglik)


def _analyze_group(group: pl.DataFrame, acquisition: Acquisition, options: FbmPostOptions,
                   joint: JointGrid | None) -> TrackPosteriorFbm:
    """`analyze_track`, with an invalid track as an `invalid_input` result rather than a raise."""
    try:
        return analyze_track(group, acquisition, options, joint)
    except ValueError as exc:
        return TrackPosteriorFbm(int(group["track_id"][0]), group.height, _empty("invalid_input", str(exc)),
                                 acquisition, options)


def analyze_tracks(table: pl.DataFrame, acquisition: Acquisition,
                   options: FbmPostOptions = FbmPostOptions(), *,
                   progress: Callable[[int, int], None] | None = None,
                   keep_posteriors: bool = False,
                   keep_likelihoods: bool = False,
                   joint: JointGrid = JointGrid(),
                   map_fn: Callable[..., Iterable] = map) -> FbmPosteriorAnalysis:
    """One fit row per input track (model posterior_alpha_D), including exclusions and invalid tracks.

    Same contract as `gridpost.analyze_tracks`: schema errors raise before work starts, invalid
    tracks get status='invalid_input', `map_fn` (e.g. a `ThreadPoolExecutor`'s `map`; the linear
    algebra releases the GIL) must yield results in input order. With `keep_posteriors`,
    `result.posteriors` holds every "ok" track's marginal log posterior of alpha (see
    `AlphaPosteriors` for why that is not a population model's input). With `keep_likelihoods`,
    `result.likelihoods` holds every "ok" track's likelihood on `joint`'s (alpha, log10 D_ref) cells,
    what `population.deconvolve_tracks` fits a population to.
    """
    validate_acquisition(acquisition, allow_exposure=True)
    validate_table_schema(table)
    groups = table.sort("track_id", "frame").partition_by("track_id", maintain_order=True)
    rows, ids, frames, log_post = [], [], [], []
    lik_ids, lik_frames, floors, cells = [], [], [], []
    if progress is not None:
        progress(0, len(groups))
    results = map_fn(functools.partial(_analyze_group, acquisition=acquisition, options=options,
                                       joint=joint if keep_likelihoods else None), groups)
    for done, result in enumerate(results, 1):
        post = result.posterior
        rows.append({"track_id": result.track_id, "n_frames": result.n_frames, "model": post.model,
                     "method": post.method, "status": post.status, "message": post.message,
                     "uncertainty_method": post.uncertainty_method, **post.parameters,
                     "D_grid_edge": post.grid_edge})
        if keep_posteriors and result.log_post_alpha is not None:
            ids.append(result.track_id)
            frames.append(result.n_frames)
            log_post.append(result.log_post_alpha)
        if keep_likelihoods and result.joint_loglik is not None:
            lik_ids.append(result.track_id)
            lik_frames.append(result.n_frames)
            floors.append(post.parameters["D_ref_floor_um2_s"])
            cells.append(result.joint_loglik)
        if progress is not None:
            progress(done, len(groups))
    posteriors = None
    if keep_posteriors:
        posteriors = AlphaPosteriors(np.array(ids, dtype=np.int64), np.array(frames, dtype=np.int64),
                                     np.array(log_post).reshape(len(ids), options.n_alpha))
    likelihoods = None
    if keep_likelihoods:
        alphas, D = joint.alphas(options), joint.log10_D_ref()
        likelihoods = JointLikelihoods(np.array(lik_ids, dtype=np.int64), np.array(lik_frames, dtype=np.int64),
                                       np.array(floors, dtype=float), alphas, D,
                                       np.array(cells, dtype=float).reshape(len(lik_ids), len(alphas), len(D)))
    return FbmPosteriorAnalysis(pl.DataFrame(rows, schema=FIT_SCHEMA), acquisition, options, posteriors, likelihoods)
