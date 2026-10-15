"""Functional single-track and table workflows for the fBm (alpha, D_ref) profile-likelihood fit."""
import functools
from collections.abc import Callable, Iterable

import numpy as np
import polars as pl

from ..data import Acquisition
from ..gridpost.likelihood import _prepared
from ..io import validate_table_schema, validated_track_frame
from ..tracks import TrackArrays, by_length, chunks
from ..validation import validate_acquisition
from . import likelihood as L
from . import profile as P
from .data import AlphaProfiles, FbmAnalysis, FbmPostOptions, JointGrid, JointLikelihoods, ProfileAlphaD, TrackFitFbm

CHUNK_TRACKS = 64  # tracks per unit of work handed to `map_fn`

FIT_SCHEMA = {
    "track_id": pl.Int64, "n_frames": pl.Int64, "model": pl.String,
    "method": pl.String, "status": pl.String, "message": pl.String,
    "uncertainty_method": pl.String,
    **{name: pl.Float64 for name in ProfileAlphaD.PARAMETERS},
    "alpha_edge": pl.String,
}

_NO_MOTION = "no motion above the localization noise: alpha is not identified and D_ref is only bounded above"
_EDGE_MESSAGE = {
    "low": "the alpha interval reaches alpha_min",
    "high": "the alpha interval reaches alpha_max",
    "both": "the alpha interval spans the whole grid",
}


def _empty(status: str, message: str) -> ProfileAlphaD:
    return ProfileAlphaD(dict.fromkeys(ProfileAlphaD.PARAMETERS), status, message)


def _excluded(n: int, options: FbmPostOptions) -> ProfileAlphaD:
    return _empty("excluded", f"{n} frames; min_frames={options.min_frames}")


def _number(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def _fit(s: P.TrackProfiles, k: int, floor: float, tau_ref_s: float) -> ProfileAlphaD:
    """Track k of `s` as an "ok" result."""
    identified = np.isfinite(s.alpha_hat[k])
    x_hat = s.log10_D_ref_hat[k]
    parameters = {
        "alpha_hat": _number(s.alpha_hat[k]), "alpha_lo": float(s.alpha_lo[k]), "alpha_hi": float(s.alpha_hi[k]),
        "alpha_dll_brownian": float(s.dll_brownian[k]), "alpha_p_brownian": float(s.p_brownian[k]),
        "K_hat": float(np.exp(s.ln_K_hat[k])), "D_ref_hat_um2_s": float(10. ** x_hat),
        "D_ref_lo_um2_s": float(10. ** s.log10_D_ref_lo[k]), "D_ref_hi_um2_s": float(10. ** s.log10_D_ref_hi[k]),
        "log10_D_ref_hat": _number(x_hat), "D_ref_floor_um2_s": floor, "loglik_max": float(s.loglik_max[k]),
    }
    edge = s.alpha_edge[k]
    message = _EDGE_MESSAGE.get(edge, "") if identified else _NO_MOTION
    return ProfileAlphaD(parameters, "ok", message, alpha_edge=edge)


def _fit_batch(xy: np.ndarray, sd: np.ndarray, acquisition: Acquisition, options: FbmPostOptions,
               joint: JointGrid | None) -> tuple[P.TrackProfiles, np.ndarray | None]:
    """Profile summaries, and given `joint` the population cells, of tracks of one length."""
    grid = options.alphas()
    alphas, i_one = P.alphas_with_one(grid)
    w = L.whiten_batch(xy, sd, float(acquisition.dt_s), float(acquisition.exposure_s), alphas)
    s = P.summarize(w, alphas, len(grid), i_one, options.level, options.tau_ref_s)
    return s, None if joint is None else L.cells(w, options, joint)


def _fit_or_invalid(track: pl.DataFrame, acquisition: Acquisition, options: FbmPostOptions,
                    joint: JointGrid | None) -> tuple[ProfileAlphaD, np.ndarray | None, np.ndarray | None]:
    try:
        prepared = _prepared(track, acquisition)
    except ValueError as exc:  # e.g. a zero localization SD
        return _empty("invalid_input", str(exc)), None, None
    xy, sd = (prepared.select(c).to_numpy()[None] for c in (["x_um", "y_um"], ["sigma_x_um", "sigma_y_um"]))
    s, cells = _fit_batch(xy, sd, acquisition, options, joint)
    return (_fit(s, 0, L.localization_floor(track, options), options.tau_ref_s), s.curve[0],
            None if cells is None else cells[0])


def analyze_track(track: pl.DataFrame, acquisition: Acquisition,
                  options: FbmPostOptions = FbmPostOptions(), joint: JointGrid | None = None) -> TrackFitFbm:
    """alpha and D_ref of one track's rows by profile likelihood, with likelihood-ratio intervals.

    Invalid input raises; short tracks are explicit results. The likelihood models
    `acquisition.exposure_s` as box-shutter blur. An "ok" result carries the track's profile of alpha
    (`profile`, maximum 0) and, given `joint`, its likelihood on those cells (`joint_loglik`).
    """
    validate_acquisition(acquisition, allow_exposure=True)
    track = validated_track_frame(track, acquisition, require_localization=True)
    track_id = int(track["track_id"][0])
    n = track.height
    if n < options.min_frames:
        return TrackFitFbm(track_id, n, _excluded(n, options), acquisition, options)
    fit, curve, cells = _fit_or_invalid(track, acquisition, options, joint)
    return TrackFitFbm(track_id, n, fit, acquisition, options, curve, cells)


def _analyze_group(group: pl.DataFrame, acquisition: Acquisition, options: FbmPostOptions,
                   joint: JointGrid | None) -> TrackFitFbm:
    """`analyze_track`, with an invalid track as an `invalid_input` result rather than a raise."""
    try:
        return analyze_track(group, acquisition, options, joint)
    except ValueError as exc:
        return TrackFitFbm(int(group["track_id"][0]), group.height, _empty("invalid_input", str(exc)),
                           acquisition, options)


def _analyze_chunk(index: range, arrays: TrackArrays, acquisition: Acquisition, options: FbmPostOptions,
                   joint: JointGrid | None) -> list[TrackFitFbm]:
    """`analyze_track` for tracks `index` of `arrays`, in order. Clean tracks of one length are fitted
    together (`likelihood.whiten_batch`); any other track takes the per-track path, so its result is that
    path's own."""
    out: dict[int, TrackFitFbm] = {}
    n_frames = arrays.n_frames
    fast = [i for i in index if arrays.clean[i] and n_frames[i] >= options.min_frames]
    for group in by_length(fast, n_frames):
        xy, sd = arrays.stack(group)
        s, cells = _fit_batch(xy, sd, acquisition, options, joint)
        floors = [np.mean(v ** 2) / options.tau_ref_s for v in sd]  # as `likelihood.localization_floor`
        for k, i in enumerate(group):
            out[i] = TrackFitFbm(int(arrays.track_id[i]), int(n_frames[i]),
                                 _fit(s, k, float(floors[k]), options.tau_ref_s), acquisition, options,
                                 s.curve[k], None if cells is None else cells[k])
    for i in index:
        if i in out:
            continue
        n = int(n_frames[i])
        if arrays.valid[i] and n < options.min_frames:
            out[i] = TrackFitFbm(int(arrays.track_id[i]), n, _excluded(n, options), acquisition, options)
        else:
            out[i] = _analyze_group(arrays.frame(i), acquisition, options, joint)
    return [out[i] for i in index]


def analyze_tracks(table: pl.DataFrame, acquisition: Acquisition,
                   options: FbmPostOptions = FbmPostOptions(), *,
                   progress: Callable[[int, int], None] | None = None,
                   keep_profiles: bool = False,
                   keep_likelihoods: bool = False,
                   joint: JointGrid = JointGrid(),
                   map_fn: Callable[..., Iterable] = map) -> FbmAnalysis:
    """One fit row per input track (model fbm_alpha_D), including exclusions and invalid tracks.

    Same contract as `gridpost.analyze_tracks`: schema errors raise before work starts, invalid
    tracks get status='invalid_input', and `map_fn` (e.g. a `ThreadPoolExecutor`'s `map`; the linear
    algebra releases the GIL) runs chunks of `CHUNK_TRACKS` consecutive tracks and must yield results
    in input order. With `keep_profiles`, `result.profiles` holds every "ok" track's profile of alpha,
    what `population.shared_alpha` sums. With `keep_likelihoods`, `result.likelihoods` holds every "ok"
    track's likelihood on `joint`'s (alpha, log10 D_ref) cells, what `population.deconvolve_tracks` fits a
    population to.
    """
    validate_acquisition(acquisition, allow_exposure=True)
    validate_table_schema(table)
    arrays = TrackArrays.from_table(table, acquisition)
    total = len(arrays)
    rows, ids, frames, curves = [], [], [], []
    lik_ids, lik_frames, floors, cells = [], [], [], []
    if progress is not None:
        progress(0, total)
    results = map_fn(functools.partial(_analyze_chunk, arrays=arrays, acquisition=acquisition, options=options,
                                       joint=joint if keep_likelihoods else None), chunks(total, CHUNK_TRACKS))
    done = 0
    for chunk in results:
        for result in chunk:
            fit = result.fit
            rows.append({"track_id": result.track_id, "n_frames": result.n_frames, "model": fit.model,
                         "method": fit.method, "status": fit.status, "message": fit.message,
                         "uncertainty_method": fit.uncertainty_method, **fit.parameters,
                         "alpha_edge": fit.alpha_edge})
            if keep_profiles and result.profile is not None:
                ids.append(result.track_id)
                frames.append(result.n_frames)
                curves.append(result.profile)
            if keep_likelihoods and result.joint_loglik is not None:
                lik_ids.append(result.track_id)
                lik_frames.append(result.n_frames)
                floors.append(fit.parameters["D_ref_floor_um2_s"])
                cells.append(result.joint_loglik)
            done += 1
            if progress is not None:
                progress(done, total)
    profiles = None
    if keep_profiles:
        profiles = AlphaProfiles(np.array(ids, dtype=np.int64), np.array(frames, dtype=np.int64), options.alphas(),
                                 np.array(curves, dtype=float).reshape(len(ids), options.n_alpha))
    likelihoods = None
    if keep_likelihoods:
        alphas, D = joint.alphas(options), joint.log10_D_ref()
        likelihoods = JointLikelihoods(np.array(lik_ids, dtype=np.int64), np.array(lik_frames, dtype=np.int64),
                                       np.array(floors, dtype=float), alphas, D,
                                       np.array(cells, dtype=float).reshape(len(lik_ids), len(alphas), len(D)))
    return FbmAnalysis(pl.DataFrame(rows, schema=FIT_SCHEMA), acquisition, options, profiles, likelihoods)
