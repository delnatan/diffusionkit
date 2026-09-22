"""Grid-posterior inputs and outputs. Algorithms live in other modules."""
from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from ..data import Acquisition


@dataclass(frozen=True)
class GridPostOptions:
    min_frames: int = 3  # the whitening step's own hard minimum (>= 3 frames -> >= 2 displacements)
    level: float = .9  # credible-interval mass


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
class PosteriorAlpha:
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        "alpha_post_median", "alpha_post_lo", "alpha_post_hi")
    parameters: dict[str, float | None]
    status: str
    message: str
    model: str = "posterior_alpha"
    method: str = "grid_posterior_marginal_K"
    uncertainty_method: str = "credible_interval"


@dataclass(frozen=True)
class TrackPosterior:
    track_id: int
    n_frames: int
    posterior_D: PosteriorD
    posterior_alpha: PosteriorAlpha
    acquisition: Acquisition
    options: GridPostOptions


@dataclass(frozen=True)
class GridPosteriorAnalysis:
    fits: pl.DataFrame  # one row per (track_id, model): posterior_D, posterior_alpha
    acquisition: Acquisition
    options: GridPostOptions
