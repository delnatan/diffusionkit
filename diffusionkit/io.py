"""Physical-unit table adapters; no inference or fitting in this module."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .data import Acquisition
from .validation import validate_acquisition


@dataclass(frozen=True)
class AcquisitionParams:
    """Pixel-table conversion settings, retained for existing callers."""
    pixel_size_um: float
    dt_s: float


def validate_table_schema(table: pl.DataFrame) -> None:
    required = {"track_id", "frame", "x_um", "y_um"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    for name in ("track_id", "frame"):
        if not table.schema[name].is_integer() or table[name].null_count():
            raise ValueError(f"{name} must contain non-null integers")
    if table.height and table["track_id"].max() > np.iinfo(np.int64).max:
        raise ValueError("track_id must fit in int64")
    for name in ("x_um", "y_um", "sigma_x_um", "sigma_y_um", "t_s", "track_length"):
        if name in table.columns and not table.schema[name].is_numeric():
            raise ValueError(f"{name} must be numeric")
    if ("sigma_x_um" in table.columns) != ("sigma_y_um" in table.columns):
        raise ValueError("Provide both sigma_x_um and sigma_y_um, or neither")


def validated_track_frame(table: pl.DataFrame, acquisition: Acquisition, *,
                          require_localization: bool = True) -> pl.DataFrame:
    """One track's rows: schema-checked, sorted by frame, contiguous, finite.

    Validates any redundant time/length metadata and, if `require_localization`,
    that `sigma_x_um`/`sigma_y_um` are present, finite, and nonnegative. This is
    the single validation boundary every per-track algorithm (`classic.analysis`,
    `gridpost.likelihood`) calls before touching the data.
    """
    validate_acquisition(acquisition, allow_exposure=True)
    validate_table_schema(table)
    if table.height == 0 or table["track_id"].n_unique() != 1:
        raise ValueError("Expected a nonempty table containing exactly one track_id")
    if "track_length" in table.columns:
        lengths = table["track_length"].to_numpy()
        if not np.all(lengths == table.height):
            raise ValueError("track_length disagrees with the number of rows")
    if "t_s" in table.columns:
        times = table["t_s"].to_numpy()
        expected = table["frame"].to_numpy() * acquisition.dt_s
        if not np.all(np.isfinite(times)) or not np.allclose(times, expected, rtol=1e-9, atol=1e-12):
            raise ValueError("t_s must equal frame * dt_s")
    table = table.sort("frame")
    frames = table["frame"].to_numpy()
    if np.any(frames < 0):
        raise ValueError("frame must be nonnegative")
    if np.any(np.diff(frames.astype(np.int64)) != 1):
        raise ValueError("frames must be consecutive and unique; gaps are not supported")
    positions = table.select("x_um", "y_um").to_numpy()
    if not np.all(np.isfinite(positions)):
        raise ValueError("x_um/y_um must be finite")
    if require_localization:
        if "sigma_x_um" not in table.columns:
            raise ValueError("Localization SDs are required; choose localization='ignore' explicitly to omit the correction")
        errors = table.select("sigma_x_um", "sigma_y_um").to_numpy()
        if not np.all(np.isfinite(errors)) or np.any(errors < 0):
            raise ValueError("sigma_x_um/sigma_y_um must be finite and nonnegative")
    return table


def load_tracks(csv_path: str | Path, params: AcquisitionParams) -> pl.DataFrame:
    """Read pixel coordinates and position SDs; return sorted physical units."""
    validate_acquisition(Acquisition(params.dt_s))
    if not np.isfinite(params.pixel_size_um) or params.pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be finite and positive")
    raw = pl.read_csv(csv_path)
    for name in ("track_id", "frame"):
        if name not in raw.columns or not raw.schema[name].is_integer():
            raise ValueError(f"{name} must contain integers (fractional values are not truncated)")
    table = raw.select(
        "track_id", "frame",
        (pl.col("frame") * params.dt_s).alias("t_s"),
        (pl.col("x") * params.pixel_size_um).alias("x_um"),
        (pl.col("y") * params.pixel_size_um).alias("y_um"),
        (pl.col("sigma_x") * params.pixel_size_um).alias("sigma_x_um"),
        (pl.col("sigma_y") * params.pixel_size_um).alias("sigma_y_um"),
    ).with_columns(pl.len().over("track_id").alias("track_length")).sort("track_id", "frame")
    validate_table_schema(table)
    return table


def assert_contiguous_tracks(tracks: pl.DataFrame) -> None:
    """Validate consecutive unique frames independent of input row order."""
    for group in tracks.sort("track_id", "frame").partition_by("track_id", maintain_order=True):
        frames = group["frame"].to_numpy()
        if frames.dtype.kind not in "iu" or np.any(np.diff(frames.astype(np.int64)) != 1):
            raise ValueError(f"Track {group['track_id'][0]} has gaps or duplicate frames")
