"""Data shared by track-analysis algorithms. Positions and errors are in µm."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Acquisition:
    dt_s: float
    exposure_s: float = 0.0


@dataclass(frozen=True)
class Track:
    track_id: int
    frames: np.ndarray
    positions_um: np.ndarray  # (n_frames, 2), columns x, y
    localization_sd_um: np.ndarray | None = None  # same shape, position SDs, not PSF widths
