"""Per-track MSD with pair-specific, independent static localization errors."""
from numbers import Integral

import numpy as np

from ..data import Acquisition, Track
from ..validation import validate_acquisition, validated_track
from .data import MSDCurve, MSDOptions


def validate_options(options: MSDOptions) -> None:
    for name, lower in (("max_lag", 1), ("min_frames", 2), ("max_nfev", 1)):
        value = getattr(options, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if options.localization not in ("provided", "ignore"):
        raise ValueError("localization must be 'provided' or 'ignore'")


def compute_msd(track: Track, acquisition: Acquisition,
                options: MSDOptions = MSDOptions()) -> MSDCurve:
    """Compute only requested lags. Negative corrected MSD values are retained.

    At lag l, subtract mean_i sum_axis(s_i² + s_(i+l)²). This assumes
    independent, zero-mean localization errors with the supplied SDs.
    """
    validate_acquisition(acquisition)
    validate_options(options)
    track = validated_track(track, require_localization=options.localization == "provided")
    lags = np.arange(1, min(options.max_lag, len(track.frames) - 1) + 1)
    observed, offsets = [], []
    variances = (track.localization_sd_um**2 if options.localization == "provided"
                 else np.zeros_like(track.positions_um))
    for lag in lags:
        displacement = track.positions_um[lag:] - track.positions_um[:-lag]
        observed.append(np.mean(np.sum(displacement**2, axis=1)))
        offsets.append(np.mean(np.sum(variances[lag:] + variances[:-lag], axis=1)))
    arrays = (lags, lags * acquisition.dt_s, len(track.frames) - lags,
              np.asarray(observed), np.asarray(offsets))
    for array in arrays:
        array.setflags(write=False)
    return MSDCurve(*arrays)
