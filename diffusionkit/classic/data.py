"""Classical MSD-analysis inputs and outputs. Algorithms live in other modules."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from ..data import Acquisition


@dataclass(frozen=True)
class MSDOptions:
    # The lag window per track: a fixed number of lags (an explicit comparison window, not an optimized
    # cutoff), or a fraction of the track's longest lag (`window_lags`; 0.25-0.4 is the usual rule, so
    # longer tracks use more of their curve). Exactly one is set.
    max_lag: int | None = 3
    lag_fraction: float | None = None
    min_frames: int = 5
    localization: Literal["provided", "ignore"] = "provided"
    max_nfev: int = 200
    # The power law's own window, the same two ways (at most one set; neither: the D window above). Alpha
    # usually wants more lags than D: a log-log slope needs a span of lags to show curvature, while D is best
    # read off the first, best-measured ones. The curve is computed to the larger of the two windows.
    alpha_max_lag: int | None = None
    alpha_lag_fraction: float | None = None
    # How alpha is fitted: "nls", bounded least squares in linear MSD space (`fit_anomalous_msd`), or
    # "loglog", OLS of log(MSD - offset) on log(tau) (`fit_loglog_msd`), the textbook fit the ensemble uses.
    alpha_fit: Literal["nls", "loglog"] = "nls"


@dataclass(frozen=True)
class MSDCurve:
    """An MSD at increasing lags: one track's time average, or an average over tracks (the ensemble)."""
    lag: np.ndarray
    tau_s: np.ndarray
    n_pairs: np.ndarray
    msd_um2: np.ndarray
    localization_offset_um2: np.ndarray

    def head(self, n_points: int) -> "MSDCurve":
        """The first `n_points` lags: the fitting window."""
        if isinstance(n_points, bool) or int(n_points) != n_points or not 1 <= n_points <= len(self.lag):
            raise ValueError(f"n_points must be an integer in [1, {len(self.lag)}], got {n_points!r}")
        return MSDCurve(*(a[:int(n_points)] for a in (self.lag, self.tau_s, self.n_pairs, self.msd_um2,
                                                     self.localization_offset_um2)))


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
class TrackAnalysis:
    track_id: int
    n_frames: int
    msd: MSDCurve | None
    brownian: MSDFit
    anomalous: MSDFit
    acquisition: Acquisition
    options: MSDOptions


@dataclass(frozen=True)
class ClassicAnalysis:
    fits: pl.DataFrame  # one row per (track_id, model), including failed/excluded fits
    msd: pl.DataFrame  # one row per (track_id, lag)
    acquisition: Acquisition
    options: MSDOptions
