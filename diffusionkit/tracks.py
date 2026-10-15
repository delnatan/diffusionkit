"""A whole track table as flat numpy arrays, for algorithms that work on many tracks at once.

`validated_track_frame` checks one track at a time with polars, which costs more than the likelihood of a short
track. `TrackArrays` sorts and reads the table once and marks the tracks that pass every check
`validated_track_frame` makes (`valid`), and those that also have positive localization SDs and at least three
frames, which the likelihoods need (`clean`). Those tracks go to the vectorized likelihoods in length buckets; any other track keeps going through
`validated_track_frame`, so its result and message are the per-track path's own.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .data import Acquisition


@dataclass(frozen=True)
class TrackArrays:
    """Every track of `table` (sorted by track_id, then frame), as flat arrays indexed by row offsets."""

    table: pl.DataFrame     # the sorted table: track i is table.slice(start[i], n_frames[i])
    track_id: np.ndarray    # (T,)
    start: np.ndarray       # (T,) first row of each track
    n_frames: np.ndarray    # (T,)
    xy: np.ndarray          # (rows, 2) um
    sd: np.ndarray | None   # (rows, 2) um, None without sigma columns
    valid: np.ndarray       # (T,) passes every check of `validated_track_frame` (localization required)
    clean: np.ndarray       # (T,) valid, with positive SDs and >= 3 frames: what the likelihoods need

    @classmethod
    def from_table(cls, table: pl.DataFrame, acquisition: Acquisition) -> "TrackArrays":
        """`table` must already have passed `io.validate_table_schema`."""
        table = table.sort("track_id", "frame")
        ids = table["track_id"].to_numpy()
        T = len(ids)
        if T == 0:
            empty = np.zeros(0, np.int64)
            return cls(table, empty, empty, empty, np.zeros((0, 2)), None, np.zeros(0, bool), np.zeros(0, bool))
        first = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        n = np.diff(np.r_[first, T])
        frames = table["frame"].to_numpy().astype(np.int64)
        xy = table.select("x_um", "y_um").to_numpy().astype(float)
        sd = (table.select("sigma_x_um", "sigma_y_um").to_numpy().astype(float)
              if "sigma_x_um" in table.columns else None)

        bad = ~np.isfinite(xy).all(axis=1)  # per row, then reduced per track
        step = np.r_[1, np.diff(frames)]
        step[first] = 1
        bad |= step != 1
        bad[first] |= frames[first] < 0
        if sd is None:
            bad[:] = True
            zero = bad
        else:
            bad |= ~(np.isfinite(sd).all(axis=1) & (sd >= 0).all(axis=1))
            zero = (sd == 0).any(axis=1)
        if "track_length" in table.columns:
            lengths = table["track_length"].to_numpy().astype(float)
            bad |= ~(lengths == np.repeat(n, n))
        if "t_s" in table.columns:
            times = table["t_s"].to_numpy().astype(float)
            expected = frames * float(acquisition.dt_s)
            with np.errstate(invalid="ignore"):
                bad |= ~(np.abs(times - expected) <= 1e-12 + 1e-9 * np.abs(expected))
        valid = ~np.logical_or.reduceat(bad, first)
        clean = valid & ~np.logical_or.reduceat(zero, first) & (n >= 3)
        return cls(table, ids[first], first, n, xy, sd, valid, clean)

    def __len__(self) -> int:
        return len(self.track_id)

    def frame(self, i: int) -> pl.DataFrame:
        """Track i's rows as a table, for the per-track path."""
        return self.table.slice(int(self.start[i]), int(self.n_frames[i]))

    def stack(self, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positions and SDs of tracks `index`, which must all have one length: two (len(index), n, 2) arrays."""
        n = int(self.n_frames[index[0]])
        rows = self.start[index][:, None] + np.arange(n)[None, :]
        return self.xy[rows], self.sd[rows]


def chunks(n_tracks: int, size: int) -> list[range]:
    """Consecutive track ranges of at most `size`: the unit of work a caller's `map_fn` runs."""
    return [range(s, min(s + size, n_tracks)) for s in range(0, n_tracks, size)]


def by_length(index, n_frames: np.ndarray) -> list[np.ndarray]:
    """`index` split into groups of equal track length."""
    index = np.asarray(index)
    if not len(index):
        return []
    lengths = n_frames[index]
    return [index[lengths == n] for n in np.unique(lengths)]
