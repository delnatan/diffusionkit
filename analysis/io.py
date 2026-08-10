"""Load raw SPT localization tables and convert to physical units.

Everything downstream operates on one tidy, long-format polars DataFrame:
one row per (particle, frame) localization, columns in physical units.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class AcquisitionParams:
    """Fixed imaging parameters needed to convert pixels/frames to physical units."""

    pixel_size_um: float
    dt_s: float


def load_tracks(csv_path: str | Path, params: AcquisitionParams) -> pl.DataFrame:
    """Load a localization table and convert to a tidy, physical-unit DataFrame.

    Returns columns: particle, frame, t_s, x_um, y_um, x_std_um, y_std_um,
    track_length. Sorted by (particle, frame).
    """
    raw = pl.read_csv(csv_path)
    tracks = raw.select(
        pl.col("particle").cast(pl.Int64),
        pl.col("frame").cast(pl.Int64),
        (pl.col("frame") * params.dt_s).alias("t_s"),
        (pl.col("x") * params.pixel_size_um).alias("x_um"),
        (pl.col("y") * params.pixel_size_um).alias("y_um"),
        (pl.col("x_std") * params.pixel_size_um).alias("x_std_um"),
        (pl.col("y_std") * params.pixel_size_um).alias("y_std_um"),
        pl.col("track_length").cast(pl.Int64),
    ).sort(["particle", "frame"])
    return tracks


def assert_contiguous_tracks(tracks: pl.DataFrame) -> None:
    """Raise if any track has frame gaps.

    Time-averaged MSD as implemented here assumes each track is sampled on a
    uniform frame grid with no missing frames (lag n <-> tau = n * dt_s).
    """
    bad = (
        tracks.group_by("particle")
        .agg(is_contiguous=(pl.col("frame").diff().drop_nulls() == 1).all())
        .filter(~pl.col("is_contiguous"))
    )
    if bad.height > 0:
        ids = bad["particle"].to_list()
        raise ValueError(f"{len(ids)} tracks have frame gaps: {ids[:10]}")
