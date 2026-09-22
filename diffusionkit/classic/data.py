"""Classical-analysis inputs and outputs. Algorithms live in other modules."""
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import polars as pl

from ..data import Acquisition


@dataclass(frozen=True)
class MSDOptions:
    max_lag: int = 3  # explicit comparison window, not an optimized cutoff
    min_frames: int = 5
    localization: Literal["provided", "ignore"] = "provided"
    max_nfev: int = 200


@dataclass(frozen=True)
class MSDCurve:
    lag: np.ndarray
    tau_s: np.ndarray
    n_pairs: np.ndarray
    msd_um2: np.ndarray
    localization_offset_um2: np.ndarray


@dataclass(frozen=True)
class MSDFit:
    model: str
    method: str
    parameters: dict[str, float | None]
    status: str
    message: str
    n_lags: int
    residual_sum_squares_um4: float | None = None
    optimizer_status: int | None = None
    nfev: int = 0
    uncertainty_method: str = "not_estimated"


@dataclass(frozen=True)
class PosteriorD:
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        "D_post_median_um2_s", "D_post_lo_um2_s", "D_post_hi_um2_s")
    parameters: dict[str, float | None]
    status: str
    message: str
    model: str = "posterior_D"
    method: str = "grid_posterior"
    uncertainty_method: str = "credible_interval"


@dataclass(frozen=True)
class TrackAnalysis:
    track_id: int
    n_frames: int
    msd: MSDCurve | None
    brownian: MSDFit
    anomalous: MSDFit
    acquisition: Acquisition
    options: MSDOptions
    posterior_D: PosteriorD


@dataclass(frozen=True)
class ClassicAnalysis:
    fits: pl.DataFrame  # one row per (track_id, model), including failed/excluded fits
    msd: pl.DataFrame  # one row per (track_id, lag)
    acquisition: Acquisition
    options: MSDOptions
