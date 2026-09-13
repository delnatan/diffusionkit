"""Model-free per-track shape descriptors.

Numbers read straight off a trajectory's positions, no diffusion model and
no noise term: the things worth sorting and filtering tracks by *before*
(or alongside) a fit, and the axes a fitted D or alpha is usefully plotted
against -- a low alpha on a track that also has a small radius of gyration
is confinement; a low alpha on a track that spans the field is more likely
a fit artifact.

All of these are raw geometry, so localization error inflates them, most
on short or slow tracks where the true extent is comparable to the error.
They describe what was measured, not the underlying motion.
"""
from __future__ import annotations

import polars as pl


def track_geometry(tracks: pl.DataFrame) -> pl.DataFrame:
    """One row per track from `load_tracks`'s tidy table.

    - `radius_of_gyration_um`: RMS distance of the positions from their own
      centroid -- the track's spatial extent, the usual confinement-size read.
    - `net_displacement_um`: straight-line distance from first to last position.
    - `straightness`: net displacement over the summed step lengths, in [0, 1];
      near 1 for directed motion, small for diffusive or confined motion.
    - `gyration_asymmetry`: `(l1 - l2)^2 / (l1 + l2)^2` from the 2D gyration
      tensor's eigenvalues, in [0, 1]; 0 for an isotropic cloud of positions,
      1 for positions on a line. A descriptor of the track's shape, not a test
      for anisotropic diffusion (see `bayes.anisotropy` for that).

    Columns are null where undefined: every one of them for a single-point
    track, and `straightness`/`gyration_asymmetry` for a track that never moved.
    """
    ordered = tracks.sort(["track_id", "frame"]).with_columns(
        (
            pl.col("x_um").diff().over("track_id") ** 2
            + pl.col("y_um").diff().over("track_id") ** 2
        )
        .sqrt()
        .alias("_step_um")
    )
    per_track = ordered.group_by("track_id").agg(
        pl.len().alias("_n"),
        pl.col("x_um").var(ddof=0).alias("_txx"),
        pl.col("y_um").var(ddof=0).alias("_tyy"),
        (
            (pl.col("x_um") - pl.col("x_um").mean()) * (pl.col("y_um") - pl.col("y_um").mean())
        )
        .mean()
        .alias("_txy"),
        (
            (pl.col("x_um").last() - pl.col("x_um").first()) ** 2
            + (pl.col("y_um").last() - pl.col("y_um").first()) ** 2
        )
        .sqrt()
        .alias("net_displacement_um"),
        pl.col("_step_um").sum().alias("_path_um"),
    )
    trace = pl.col("_txx") + pl.col("_tyy")
    multi = pl.col("_n") > 1
    return per_track.select(
        "track_id",
        pl.when(multi).then(trace.sqrt()).alias("radius_of_gyration_um"),
        pl.when(multi).then(pl.col("net_displacement_um")).alias("net_displacement_um"),
        pl.when(multi & (pl.col("_path_um") > 0))
        .then(pl.col("net_displacement_um") / pl.col("_path_um"))
        .alias("straightness"),
        pl.when(multi & (trace > 0))
        .then(((pl.col("_txx") - pl.col("_tyy")) ** 2 + 4 * pl.col("_txy") ** 2) / trace**2)
        .alias("gyration_asymmetry"),
    ).sort("track_id")
