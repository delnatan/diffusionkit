"""The spatial run: classes for C = 1..max_classes, the suggested C, beta's shuffle null, the per-track columns."""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import polars as pl

from ..gridpost import posterior as posterior_mod
from ..gridpost.data import GridLikelihoods, GridPosteriorAnalysis
from .data import ShuffleNull, SpatialAnalysis, SpatialOptions
from .potts import PottsFit, fit, knn


def centroids(tracks: pl.DataFrame) -> pl.DataFrame:
    """Each track's mean position: `track_id`, `x_um`, `y_um`."""
    return tracks.group_by("track_id").agg(pl.col("x_um").mean(), pl.col("y_um").mean()).sort("track_id")


def _selection(fits: dict[int, PottsFit], scores: dict[int, float], n: int, gain_per_track: float
               ) -> tuple[pl.DataFrame, int]:
    """The score table and the suggested C: the last C whose gain over C - 1 clears `gain_per_track` per track,
    counting up from C = 1 and stopping at the first that does not."""
    cs = sorted(fits)
    gains = [None] + [scores[c] - scores[c - 1] for c in cs[1:]]
    suggested = 1
    for c, g in zip(cs[1:], gains[1:]):
        if g / n < gain_per_track:
            break
        suggested = c
    table = pl.DataFrame({
        "n_classes": cs,
        "score": [scores[c] for c in cs],
        "gain": gains,
        "gain_per_track": [None if g is None else g / n for g in gains],
        "beta": [fits[c].beta for c in cs],
        "suggested": [c == suggested for c in cs],
    }, schema={"n_classes": pl.Int64, "score": pl.Float64, "gain": pl.Float64, "gain_per_track": pl.Float64,
               "beta": pl.Float64, "suggested": pl.Boolean})
    return table, suggested


def analyze_likelihoods(likelihoods: GridLikelihoods, u_D: np.ndarray, positions: pl.DataFrame,
                        options: SpatialOptions = SpatialOptions(), *,
                        map_fn: Callable[..., Iterable] = map,
                        progress: Callable[[int, int], None] | None = None) -> SpatialAnalysis:
    """Spatial classes of the tracks in `likelihoods` (gridpost's kept rows on the grid `u_D`) at `positions`
    (`track_id`, `x_um`, `y_um`; `centroids` makes it from a track table). Tracks without a position are left
    out. `map_fn` runs the shuffles (a pool's `map`); `progress(done, total)` counts fits."""
    pos = positions.select("track_id", "x_um", "y_um").unique("track_id")
    ids = np.asarray(likelihoods.track_ids)
    keep = np.isin(ids, pos["track_id"].to_numpy())
    if not keep.any():
        raise ValueError("no track in the likelihoods has a position")
    ids = ids[keep]
    xy = (pl.DataFrame({"track_id": ids}).join(pos, on="track_id", how="left")
          .select("x_um", "y_um").to_numpy().astype(float))
    if not np.isfinite(xy).all():
        raise ValueError("track positions must be finite")
    ll = np.asarray(likelihoods.loglik_D, float)[keep]
    offset = ll.max(axis=1)
    lik = np.exp(ll - offset[:, None])
    u = np.asarray(u_D, float)
    nb = knn(xy, options.k)
    n = len(ids)

    total = options.max_classes + (options.n_shuffles if options.max_classes > 1 else 0)
    done = 0

    def tick():
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, total)

    kw = dict(max_iter=options.max_iter, tol=options.tol)
    plains, fits, scores = {}, {}, {}
    for c in range(1, options.max_classes + 1):
        plains[c] = plain = fit(lik, u, nb, c, beta=0., **kw)
        fits[c] = plain if c == 1 else fit(lik, u, nb, c, beta=None, init=plain, **kw)
        scores[c] = fits[c].score(nb, offset)
        tick()
    selection, suggested = _selection(fits, scores, n, options.gain_per_track)
    chosen = options.n_classes or suggested

    null = None
    if chosen > 1 and options.n_shuffles:
        # Each shuffle repeats the observed fit's own path: the plain mixture's classes, then beta from 0.
        observed, start = fits[chosen], plains[chosen]
        seeds = np.random.SeedSequence(options.seed).spawn(options.n_shuffles)

        base = start.score(nb, offset)  # beta = 0: no graph in it

        def shuffled(seed) -> tuple[float, float]:
            graph = knn(xy[np.random.default_rng(seed).permutation(n)], options.k)
            f = fit(lik, u, graph, chosen, beta=None, init=start, **kw)
            return f.score(graph, offset) - base, f.beta

        out = []
        for r in map_fn(shuffled, seeds):
            out.append(r)
            tick()
        gains, betas = np.asarray(out, float).reshape(-1, 2).T
        gain = scores[chosen] - base
        null = ShuffleNull(gain, observed.beta, gains, betas, float((1 + (gains >= gain).sum()) / (1 + len(gains))))
    elif progress is not None:
        progress(total, total)

    post = fits[chosen].track_posteriors(lik, u)
    D = np.array([[s["mean"], s["lo"], s["hi"]] for s in
                  (posterior_mod.summary(p, u, options.level) for p in post)])
    return SpatialAnalysis(ids, np.asarray(likelihoods.n_frames)[keep], xy, nb, u, options, fits, selection,
                           chosen, null, D)


def analyze(result: GridPosteriorAnalysis, positions: pl.DataFrame, options: SpatialOptions = SpatialOptions(),
            **kw) -> SpatialAnalysis:
    """`analyze_likelihoods` on a gridpost run kept with `keep_likelihoods=True`."""
    if result.likelihoods is None:
        raise ValueError("the gridpost run kept no likelihoods: run it with keep_likelihoods=True")
    return analyze_likelihoods(result.likelihoods, result.options.u_D(), positions, options, **kw)
