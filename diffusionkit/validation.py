"""Validation and defensive copies at algorithm boundaries."""
from numbers import Integral

import numpy as np

from .data import Acquisition, Track


def validate_acquisition(acquisition: Acquisition, *, allow_exposure: bool = False) -> None:
    """allow_exposure=True only for algorithms that model motion blur."""
    if not np.isfinite(acquisition.dt_s) or acquisition.dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if not np.isfinite(acquisition.exposure_s) or acquisition.exposure_s < 0:
        raise ValueError("exposure_s must be finite and nonnegative")
    if acquisition.exposure_s > acquisition.dt_s:
        raise ValueError("exposure_s cannot exceed dt_s")
    if acquisition.exposure_s > 0 and not allow_exposure:
        raise ValueError("Motion blur is not implemented; this analysis assumes instantaneous positions (exposure_s=0)")


def validated_track(track: Track, *, require_localization: bool = True) -> Track:
    """Return sorted, copied, read-only arrays; reject gaps and duplicate frames."""
    if isinstance(track.track_id, bool) or not isinstance(track.track_id, Integral):
        raise ValueError("track_id must be an integer")
    if not np.iinfo(np.int64).min <= track.track_id <= np.iinfo(np.int64).max:
        raise ValueError("track_id must fit in int64")
    frames = np.asarray(track.frames)
    positions = np.array(track.positions_um, dtype=float, copy=True)
    if frames.ndim != 1 or frames.size == 0 or frames.dtype.kind not in "iu":
        raise ValueError("frames must be a nonempty one-dimensional integer array")
    if positions.shape != (frames.size, 2) or not np.all(np.isfinite(positions)):
        raise ValueError("positions_um must be a finite (n_frames, 2) array")
    if np.any(frames < 0) or np.any(frames > np.iinfo(np.int64).max):
        raise ValueError("frames must be nonnegative int64 values")
    frames = np.array(frames, dtype=np.int64, copy=True)
    order = np.argsort(frames)
    frames, positions = frames[order], positions[order]
    if np.any(np.diff(frames) != 1):
        raise ValueError("frames must be consecutive and unique; gaps are not supported")
    errors = track.localization_sd_um
    if errors is None:
        if require_localization:
            raise ValueError("Localization SDs are required; choose localization='ignore' explicitly to omit the correction")
    else:
        errors = np.array(errors, dtype=float, copy=True)
        if errors.shape != positions.shape or not np.all(np.isfinite(errors)) or np.any(errors < 0):
            raise ValueError("localization_sd_um must be a finite nonnegative (n_frames, 2) array")
        errors = errors[order]
        errors.setflags(write=False)
    frames.setflags(write=False)
    positions.setflags(write=False)
    return Track(int(track.track_id), frames, positions, errors)
