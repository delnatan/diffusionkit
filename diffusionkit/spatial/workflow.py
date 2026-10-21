"""The spatial run: classes for C = 1..max_classes, the suggested C, the shuffle null, the per-track columns."""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import polars as pl

from ..gridpost import posterior as posterior_mod
from ..gridpost.data import GridLikelihoods, GridPosteriorAnalysis
from .data import MODEL, ShuffleNull, SpatialAnalysis, SpatialOptions
from .potts import PottsFit, agreement_table, densities, fit, graph, splits


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
    adjacency = graph(xy, options.k)
    n = len(ids)

    total = options.max_classes + (options.n_shuffles if options.max_classes > 1 else 0)
    done = 0

    def tick():
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, total)

    kw = dict(max_iter=options.max_iter, tol=options.tol)
    # beta's M-step reads E_beta[S] off a table that depends only on the graph and C. A shuffle permutes the tracks
    # over the same point set, so its graph is this one with the nodes relabelled: the tables serve it too.
    tables = {c: agreement_table(adjacency, c, options.seed) for c in range(2, options.max_classes + 1)}
    plains, fits, scores, starts = {}, {}, {}, {}
    for c in range(1, options.max_classes + 1):
        plains[c] = plain = fit(lik, u, adjacency, c, beta=0., **kw)
        if c == 1:
            fits[c] = plain
        else:
            # From the plain mixture and from the C - 1 fit with each class split; the best score is kept.
            for start in [(plain.centre, plain.sd), *splits(fits[c - 1])]:
                f = fit(lik, u, adjacency, c, tables[c], init=start, **kw)
                if c not in fits or f.score(adjacency, offset) > fits[c].score(adjacency, offset):
                    fits[c], starts[c] = f, start
        scores[c] = fits[c].score(adjacency, offset)
        tick()
    selection, suggested = _selection(fits, scores, n, options.gain_per_track)
    chosen = options.n_classes or suggested

    null = None
    if chosen > 1 and options.n_shuffles:
        # Each shuffle repeats the observed fit's own path, from its start with beta from 0. Shuffling the
        # positions is permuting the tracks over the graph's nodes.
        observed, start = fits[chosen], starts[chosen]
        seeds = np.random.SeedSequence(options.seed).spawn(options.n_shuffles)

        base = plains[chosen].score(adjacency, offset)  # beta = 0: no graph in it

        def shuffled(seed) -> tuple[float, float]:
            perm = np.random.default_rng(seed).permutation(n)
            f = fit(lik[perm], u, adjacency, chosen, tables[chosen], init=start, **kw)
            return f.score(adjacency, offset[perm]) - base, f.beta

        out = []
        for r in map_fn(shuffled, seeds):
            out.append(r)
            tick()
        gains, betas = np.asarray(out, float).reshape(-1, 2).T
        gain = scores[chosen] - base
        null = ShuffleNull(gain, observed.beta, gains, betas, float((1 + (gains >= gain).sum()) / (1 + len(gains))))
    elif progress is not None:
        progress(total, total)

    return SpatialAnalysis(ids, np.asarray(likelihoods.n_frames)[keep], xy, adjacency, u, options, fits, selection,
                           chosen, null, _spatial_D(fits[chosen], lik, u, options.level))


def _spatial_D(f: PottsFit, lik: np.ndarray, u: np.ndarray, level: float) -> np.ndarray:
    return np.array([[s["mean"], s["lo"], s["hi"]] for s in
                     (posterior_mod.summary(p, u, level) for p in f.track_posteriors(lik, u))])


def analyze(result: GridPosteriorAnalysis, positions: pl.DataFrame, options: SpatialOptions = SpatialOptions(),
            **kw) -> SpatialAnalysis:
    """`analyze_likelihoods` on a gridpost run kept with `keep_likelihoods=True`."""
    if result.likelihoods is None:
        raise ValueError("the gridpost run kept no likelihoods: run it with keep_likelihoods=True")
    return analyze_likelihoods(result.likelihoods, result.options.u_D(), positions, options, **kw)


def restore(likelihoods: GridLikelihoods, u_D: np.ndarray, record: dict, probabilities: pl.DataFrame
            ) -> SpatialAnalysis:
    """The `SpatialAnalysis` saved as `record()` and `probabilities()`, over the same likelihoods. Only the fit of
    the C used comes back (`fits` has that one key); the selection table and the null are as saved."""
    if record.get("model") != MODEL:
        raise ValueError("this spatial run was saved by an older model (with class weights) and cannot be rebuilt "
                         "under the current one: run the spatial classes again")
    opts = SpatialOptions(**record["options"])
    probs = probabilities
    ids = probs["track_id"].to_numpy()
    rows = {int(t): i for i, t in enumerate(np.asarray(likelihoods.track_ids))}
    missing = [int(t) for t in ids if int(t) not in rows]
    if missing:
        raise ValueError(f"{len(missing)} saved tracks have no likelihood, e.g. {missing[:3]}")
    idx = np.array([rows[int(t)] for t in ids], int)
    ll = np.asarray(likelihoods.loglik_D, float)[idx]
    lik = np.exp(ll - ll.max(axis=1, keepdims=True))
    u = np.asarray(u_D, float)
    c, n = record["n_classes"], record["classes"]
    centre, sd = np.asarray(n["centre_ln_D"], float), np.asarray(n["sd_ln_D"], float)
    q = probs.select([f"q_{k}" for k in range(1, c + 1)]).to_numpy()
    f = PottsFit(centre, sd, float(n["beta"]), float(n["agreement"]), q, lik @ densities(u, centre, sd),
                 int(n["n_iter"]), bool(n["converged"]))
    xy = probs.select("x_um", "y_um").to_numpy().astype(float)
    null = record["null"]
    if null is not None:
        null = ShuffleNull(null["gain"], null["beta"], np.asarray(null["shuffled_gain"], float),
                           np.asarray(null["shuffled_beta"], float), null["p"])
    selection = pl.DataFrame(record["selection"], schema={
        "n_classes": pl.Int64, "score": pl.Float64, "gain": pl.Float64, "gain_per_track": pl.Float64,
        "beta": pl.Float64, "suggested": pl.Boolean})
    return SpatialAnalysis(ids, np.asarray(likelihoods.n_frames)[idx], xy, graph(xy, opts.k), u, opts, {c: f},
                           selection, c, null, _spatial_D(f, lik, u, opts.level))
