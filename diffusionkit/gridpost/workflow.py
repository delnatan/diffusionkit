"""Functional single-track and table workflows for the D grid posterior."""
import functools
from collections.abc import Callable, Iterable

import numpy as np
import polars as pl
from scipy.special import logsumexp

from ..data import Acquisition
from ..io import validate_table_schema, validated_track_frame
from ..validation import validate_acquisition
from ..tracks import TrackArrays, by_length, chunks
from . import posterior as posterior_mod
from .likelihood import brownian_loglik_batch
from .data import GridLikelihoods, GridPosteriorAnalysis, GridPostOptions, PosteriorD, TrackPosterior

CHUNK_TRACKS = 256  # tracks per unit of work handed to `map_fn`

FIT_SCHEMA = {
    "track_id": pl.Int64, "n_frames": pl.Int64, "model": pl.String,
    "method": pl.String, "status": pl.String, "message": pl.String,
    "uncertainty_method": pl.String,
    **{name: pl.Float64 for name in PosteriorD.PARAMETERS},
    "D_grid_edge": pl.String,
}


def _edge_message(p: np.ndarray, options: GridPostOptions) -> str:
    """Why this D posterior's summary depends on the grid, or "" when it does not."""
    ratios = posterior_mod.edge_ratios(p)
    cut = [f"{name}={edge:g} um^2/s (edge/peak {r:.2g})"
           for name, edge, r in (("D_min_um2_s", options.D_min_um2_s, ratios[0]),
                                 ("D_max_um2_s", options.D_max_um2_s, ratios[1]))
           if r > posterior_mod.EDGE_RATIO_WARN]
    return "posterior cut by the grid edge at " + ", ".join(cut) if cut else ""


def _info_bits(lp: np.ndarray) -> np.ndarray:
    """`posterior.information_bits` against the flat prior, for every row of `lp` at once."""
    lq = -np.log(lp.shape[-1])
    p = np.exp(lp)
    return np.sum(np.where(p > 0, p * (lp - lq), 0.), axis=-1) / np.log(2)


def _posterior_D(lp: np.ndarray, floor: float, options: GridPostOptions, u: np.ndarray, bits: float) -> PosteriorD:
    """The "ok" summary of a normalized log posterior on `u` = `options.u_D()`, with its `_info_bits`."""
    p = np.exp(lp)
    s = posterior_mod.summary(p, u, options.level)
    return PosteriorD({"D_post_mean_um2_s": s["mean"], "D_post_lo_um2_s": s["lo"],
                       "D_post_hi_um2_s": s["hi"], "D_post_info_bits": float(bits), "D_floor_um2_s": floor},
                      "ok", _edge_message(p, options), grid_edge=posterior_mod.grid_edge(p))


def _posterior_D_or_invalid(track: pl.DataFrame, acquisition: Acquisition,
                            options: GridPostOptions) -> tuple[PosteriorD, np.ndarray | None]:
    try:
        u = options.u_D()
        # Flat prior: the normalized log posterior is the normalized log-likelihood, kept as `loglik_D`.
        lp = posterior_mod.log_posterior(posterior_mod.track_loglik(track, acquisition, u), posterior_mod.flat(u))
    except ValueError as exc:  # e.g. a zero localization SD
        return PosteriorD(dict.fromkeys(PosteriorD.PARAMETERS), "invalid_input", str(exc)), None
    return _posterior_D(lp, posterior_mod.localization_floor(track, acquisition), options, u, _info_bits(lp)), lp


def analyze_track(track: pl.DataFrame, acquisition: Acquisition,
                  options: GridPostOptions = GridPostOptions()) -> TrackPosterior:
    """The grid posterior over D for one track's rows, on `options.u_D()`.

    Invalid input raises; short tracks are explicit results. The posterior models
    `acquisition.exposure_s` as box-shutter blur. An "ok" posterior cut by a grid edge
    (`posterior.edge_ratios`) says so in its message and in `grid_edge` ("low": read the values
    as upper bounds; "high": as lower bounds). The result carries an "ok"
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


def _analyze_chunk(index: range, arrays: TrackArrays, acquisition: Acquisition,
                   options: GridPostOptions) -> list[TrackPosterior]:
    """`analyze_track` for tracks `index` of `arrays`, in order. Clean tracks are evaluated together,
    one length at a time; any other track takes the per-track path, so its result is that path's own."""
    out: dict[int, TrackPosterior] = {}
    n_frames = arrays.n_frames
    fast = [i for i in index if arrays.clean[i] and n_frames[i] >= options.min_frames]
    u = options.u_D()
    dt, exposure = float(acquisition.dt_s), float(acquisition.exposure_s)
    tau = dt - exposure / 3
    for group in by_length(fast, n_frames):
        xy, sd = arrays.stack(group)
        ll = brownian_loglik_batch(xy, sd, dt, exposure, np.exp(u))
        lp = ll - logsumexp(ll, axis=1, keepdims=True)
        floors = [np.mean(v ** 2) / tau for v in sd]  # as `posterior.localization_floor`
        for i, row, floor, bits in zip(group, lp, floors, _info_bits(lp)):
            out[i] = TrackPosterior(int(arrays.track_id[i]), int(n_frames[i]),
                                    _posterior_D(row, float(floor), options, u, bits), acquisition, options, row)
    for i in index:
        if i in out:
            continue
        n = int(n_frames[i])
        if arrays.valid[i] and n < options.min_frames:
            out[i] = TrackPosterior(int(arrays.track_id[i]), n,
                                    PosteriorD(dict.fromkeys(PosteriorD.PARAMETERS), "excluded",
                                               f"{n} frames; min_frames={options.min_frames}"),
                                    acquisition, options)
        else:
            out[i] = _analyze_group(arrays.frame(i), acquisition, options)
    return [out[i] for i in index]


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
    Progress runs in the caller's thread, once per track; no global state or worker is created.

    The table is read once into arrays (`tracks.TrackArrays`), and tracks are evaluated
    together in chunks of `CHUNK_TRACKS` consecutive tracks, one length at a time within a
    chunk. `map_fn(func, chunks)` runs the chunks and must yield results in input order, as the
    builtin `map` (the default) and `concurrent.futures.Executor.map` do. Passing a
    `ThreadPoolExecutor`'s `map` runs chunks concurrently (numpy releases the GIL), with the
    pool created and shut down by the caller; the result is identical to the serial one.
    """
    validate_acquisition(acquisition, allow_exposure=True)
    validate_table_schema(table)
    arrays = TrackArrays.from_table(table, acquisition)
    total = len(arrays)
    fit_rows = []
    D_ids, D_frames, loglik_D = [], [], []
    if progress is not None:
        progress(0, total)
    results = map_fn(functools.partial(_analyze_chunk, arrays=arrays, acquisition=acquisition, options=options),
                     chunks(total, CHUNK_TRACKS))
    done = 0
    for chunk in results:
        for result in chunk:
            post = result.posterior_D
            fit_rows.append({"track_id": result.track_id, "n_frames": result.n_frames, "model": post.model,
                             "method": post.method, "status": post.status, "message": post.message,
                             "uncertainty_method": post.uncertainty_method, **post.parameters,
                             "D_grid_edge": post.grid_edge})
            if keep_likelihoods and result.loglik_D is not None:
                D_ids.append(result.track_id)
                D_frames.append(result.n_frames)
                loglik_D.append(result.loglik_D)
            done += 1
            if progress is not None:
                progress(done, total)
    likelihoods = None
    if keep_likelihoods:
        likelihoods = GridLikelihoods(np.array(D_ids, dtype=np.int64), np.array(D_frames, dtype=np.int64),
                                      np.array(loglik_D).reshape(len(D_ids), options.n_D))
    return GridPosteriorAnalysis(pl.DataFrame(fit_rows, schema=FIT_SCHEMA), acquisition, options, likelihoods)
