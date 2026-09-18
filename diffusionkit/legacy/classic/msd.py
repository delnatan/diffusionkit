"""Time-averaged and ensemble-averaged mean squared displacement (MSD).

The numerical kernel (`_track_tamsd_arrays`) is a pure function on plain
numpy arrays: no classes, no hidden state, one job. Everything else wires it
into polars group-wise operations.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def _track_tamsd_arrays(
    x: np.ndarray, y: np.ndarray, dt_s: float, max_lag: int
) -> dict[str, np.ndarray]:
    """Time-averaged MSD for a single track, assumed uniformly sampled, no gaps.

    MSD(lag) = mean_i[ (x[i+lag]-x[i])^2 + (y[i+lag]-y[i])^2 ]

    Returns arrays keyed by lag (frames): lag, tau_s, msd_um2, n_pairs.
    """
    lags = np.arange(1, max_lag + 1)
    msd = np.empty(max_lag)
    n_pairs = np.empty(max_lag, dtype=np.int64)
    for i, lag in enumerate(lags):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        sq = dx * dx + dy * dy
        msd[i] = sq.mean()
        n_pairs[i] = sq.size
    return {"lag": lags, "tau_s": lags * dt_s, "msd_um2": msd, "n_pairs": n_pairs}


def compute_all_tamsd(
    tracks: pl.DataFrame, dt_s: float, max_lag_frac: float = 1.0
) -> pl.DataFrame:
    """Per-track TAMSD for every track, stacked into one long-format table.

    max_lag_frac caps the largest lag computed per track, as a fraction of
    (track_length - 1). Default 1.0 computes the full curve; downstream
    fitting functions decide how many points to actually use.
    """

    def _per_track(group: pl.DataFrame) -> pl.DataFrame:
        group = group.sort("frame")
        n = group.height
        max_lag = max(1, min(n - 1, int(np.floor((n - 1) * max_lag_frac))))
        arrays = _track_tamsd_arrays(
            group["x_um"].to_numpy(), group["y_um"].to_numpy(), dt_s, max_lag
        )
        return pl.DataFrame(
            {
                "track_id": np.full(max_lag, group["track_id"][0], dtype=np.int64),
                "track_length": np.full(max_lag, n, dtype=np.int64),
                **arrays,
            }
        )

    return tracks.group_by("track_id", maintain_order=True).map_groups(_per_track)


def ensemble_average_msd(tamsd: pl.DataFrame, min_tracks: int = 5) -> pl.DataFrame:
    """Ensemble average of per-track TAMSD curves at each common lag.

    Each track's MSD(lag) is weighted by its own n_pairs (more independent
    displacement pairs -> more reliable per-track estimate at that lag).
    msd_sem is the weighted standard error of the mean *across tracks*
    (between-track spread), not per-pair noise within a track.

    Lags supported by fewer than `min_tracks` tracks are dropped, since the
    ensemble average becomes unreliable (and eventually a single long track
    dominates) as fewer tracks reach that lag.
    """
    with_group_mean = tamsd.with_columns(
        (
            (pl.col("msd_um2") * pl.col("n_pairs")).sum().over(["lag", "tau_s"])
            / pl.col("n_pairs").sum().over(["lag", "tau_s"])
        ).alias("msd_mean_um2")
    )

    out = (
        with_group_mean.with_columns(
            (
                pl.col("n_pairs") * (pl.col("msd_um2") - pl.col("msd_mean_um2")) ** 2
            ).alias("_weighted_sq_dev")
        )
        .group_by(["lag", "tau_s"])
        .agg(
            n_tracks=pl.len(),
            n_pairs_total=pl.col("n_pairs").sum(),
            msd_um2=pl.col("msd_mean_um2").first(),
            msd_sem=(
                pl.col("_weighted_sq_dev").sum()
                / pl.col("n_pairs").sum()
                / pl.len()
            ).sqrt(),
        )
        .filter(pl.col("n_tracks") >= min_tracks)
        .sort("lag")
    )
    return out
