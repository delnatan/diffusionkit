"""Validation and defensive copies at algorithm boundaries."""
import numpy as np

from .data import Acquisition


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
