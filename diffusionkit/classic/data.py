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
class MLEOptions:
    n_boot: int = 500  # parametric-bootstrap replicates for z_nonbrownian; 0 disables
    seed: int = 0  # combined with track_id, so results do not depend on track order
    upper_level: float = .95  # one-sided profile-likelihood upper limit on D


@dataclass(frozen=True)
class BrownianMLE:
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        "D_um2_s", "D_upper_um2_s", "log_likelihood", "lr_motion", "p_motion",
        "z_nonbrownian", "p_nonbrownian", "z_nonbrownian_asymptotic",
        "alpha_1step", "alpha_1step_se")
    parameters: dict[str, float | None]
    status: str
    message: str
    n_boot: int = 0
    n_boot_valid: int = 0
    model: str = "brownian_mle"
    method: str = "displacement_mle"
    uncertainty_method: str = "profile_likelihood_asymptotic"


@dataclass(frozen=True)
class TrackAnalysis:
    track_id: int
    n_frames: int
    msd: MSDCurve | None
    brownian: MSDFit
    anomalous: MSDFit
    acquisition: Acquisition
    options: MSDOptions
    brownian_mle: BrownianMLE
    mle_options: MLEOptions


@dataclass(frozen=True)
class ClassicAnalysis:
    fits: pl.DataFrame  # one row per (track_id, model), including failed/excluded fits
    msd: pl.DataFrame  # one row per (track_id, lag)
    acquisition: Acquisition
    options: MSDOptions
    mle_options: MLEOptions
