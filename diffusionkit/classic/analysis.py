"""Per-track MSD with pair-specific, independent static localization errors."""
from numbers import Integral

import numpy as np
import polars as pl

from ..data import Acquisition
from ..io import validated_track_frame
from ..validation import validate_acquisition
from .data import MSDCurve, MSDOptions


MIN_LAGS = 3  # both textbook fits have two parameters; a third point is the first with a residual


def window_lags(n_available: int, fraction: float = .3, min_lags: int = MIN_LAGS) -> int:
    """How many of a curve's first lags to fit: `fraction` of the `n_available`, at least `min_lags`.

    The usual rule is to fit the first 25-40% of the curve: the MSD estimates at long lags rest on few,
    heavily overlapping pairs and are the noisiest. It is a heuristic window, not an optimized one. The
    result never exceeds `n_available`. The same rule serves a track (n_available = n_frames - 1) and an
    ensemble curve (its number of lags).
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return int(min(n_available, max(min_lags, round(fraction * n_available))))


def validate_options(options: MSDOptions) -> None:
    for name, lower in (("min_frames", 2), ("max_nfev", 1)):
        value = getattr(options, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if (options.max_lag is None) == (options.lag_fraction is None):
        raise ValueError("set exactly one of max_lag and lag_fraction (use max_lag=None with lag_fraction)")
    if options.max_lag is not None and (isinstance(options.max_lag, bool) or not isinstance(options.max_lag, Integral)
                                        or options.max_lag < 1):
        raise ValueError("max_lag must be an integer >= 1")
    if options.lag_fraction is not None and not 0 < options.lag_fraction <= 1:
        raise ValueError("lag_fraction must be in (0, 1]")
    if options.localization not in ("provided", "ignore"):
        raise ValueError("localization must be 'provided' or 'ignore'")


def compute_msd(track: pl.DataFrame, acquisition: Acquisition,
                options: MSDOptions = MSDOptions()) -> MSDCurve:
    """One track's time-averaged MSD at the requested lags only. Negative corrected values are retained.

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
    n_lags = (min(options.max_lag, n_frames - 1) if options.lag_fraction is None
              else window_lags(n_frames - 1, options.lag_fraction))
    lags = np.arange(1, n_lags + 1)
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
