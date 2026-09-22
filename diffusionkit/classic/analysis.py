"""Per-track MSD with pair-specific, independent static localization errors."""
from numbers import Integral

import numpy as np
import polars as pl

from ..data import Acquisition
from ..io import validated_track_frame
from ..validation import validate_acquisition
from .data import MSDCurve, MSDOptions


def validate_options(options: MSDOptions) -> None:
    for name, lower in (("max_lag", 1), ("min_frames", 2), ("max_nfev", 1)):
        value = getattr(options, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if options.localization not in ("provided", "ignore"):
        raise ValueError("localization must be 'provided' or 'ignore'")


def compute_msd(track: pl.DataFrame, acquisition: Acquisition,
                options: MSDOptions = MSDOptions()) -> MSDCurve:
    """Compute only requested lags. Negative corrected MSD values are retained.

    At lag l, subtract mean_i sum_axis(s_i² + s_(i+l)²). This assumes
    independent, zero-mean localization errors with the supplied SDs.
    """
    require_localization = options.localization == "provided"
    validate_acquisition(acquisition)  # no motion-blur model here; excluded upstream when exposure_s > 0
    validate_options(options)
    track = validated_track_frame(track, acquisition, require_localization=require_localization)
    n_frames = track.height
    positions = track.select("x_um", "y_um").to_numpy()
    variances = (track.select("sigma_x_um", "sigma_y_um").to_numpy()**2 if require_localization
                 else np.zeros_like(positions))
    lags = np.arange(1, min(options.max_lag, n_frames - 1) + 1)
    observed, offsets = [], []
    for lag in lags:
        displacement = positions[lag:] - positions[:-lag]
        observed.append(np.mean(np.sum(displacement**2, axis=1)))
        offsets.append(np.mean(np.sum(variances[lag:] + variances[:-lag], axis=1)))
    arrays = (lags, lags * acquisition.dt_s, n_frames - lags,
              np.asarray(observed), np.asarray(offsets))
    for array in arrays:
        array.setflags(write=False)
    return MSDCurve(*arrays)
