"""Grid-posterior inputs and outputs. Algorithms live in other modules."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import polars as pl

from ..data import Acquisition


@dataclass(frozen=True)
class GridPostOptions:
    """Everything a grid-posterior run depends on besides the tracks and the acquisition.

    The grids are part of the analysis, not a numerical detail: the default
    prior is flat in ln D over [D_min_um2_s, D_max_um2_s] and zero outside
    it, so a posterior that reaches an edge is cut there, and its median and
    interval move with the edge. Every public function that evaluates a grid
    takes these options (or the arrays they build) -- there is no module-level
    grid to fall back on.
    """
    min_frames: int = 3  # the whitening step's own hard minimum (>= 3 frames -> >= 2 displacements)
    level: float = .9  # credible-interval mass
    D_min_um2_s: float = 1e-5
    D_max_um2_s: float = 10.
    n_D: int = 601  # ~2.3% steps in D over the default range

    def __post_init__(self):
        if not (np.isfinite(self.D_min_um2_s) and np.isfinite(self.D_max_um2_s)
                and 0 < self.D_min_um2_s < self.D_max_um2_s):
            raise ValueError(f"need 0 < D_min_um2_s < D_max_um2_s, got {self.D_min_um2_s}, {self.D_max_um2_s}")
        if self.n_D < 2:
            raise ValueError("n_D must be at least 2")
        if not 0 < self.level < 1:
            raise ValueError(f"level must be in (0, 1), got {self.level}")

    def u_D(self) -> np.ndarray:
        """The ln D grid (D in um^2/s) the D posterior is evaluated on."""
        return np.linspace(np.log(self.D_min_um2_s), np.log(self.D_max_um2_s), self.n_D)


@dataclass(frozen=True)
class PosteriorD:
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        "D_post_median_um2_s", "D_post_lo_um2_s", "D_post_hi_um2_s", "D_post_info_bits",
        "D_floor_um2_s")
    parameters: dict[str, float | None]
    status: str
    message: str
    model: str = "posterior_D"
    method: str = "grid_posterior"
    uncertainty_method: str = "credible_interval"


@dataclass(frozen=True)
class TrackPosterior:
    track_id: int
    n_frames: int
    posterior_D: PosteriorD
    acquisition: Acquisition
    options: GridPostOptions
    # The normalized log posterior on `options.u_D()`, None unless the status is "ok".
    log_post_D: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class GridPosteriors:
    """Every "ok" track's normalized log posterior on `options.u_D()`, one row per track in the order of
    `track_ids`: what a population read (summed, pooled or deconvolved) is built from. The prior is
    flat in ln D, so each row is also the track's log-likelihood up to a constant."""
    track_ids: np.ndarray
    n_frames: np.ndarray  # (len(track_ids),) frames per track: a track's detections
    log_post_D: np.ndarray  # (len(track_ids), n_D)


@dataclass(frozen=True)
class GridPosteriorAnalysis:
    fits: pl.DataFrame  # one row per track (model posterior_D)
    acquisition: Acquisition
    options: GridPostOptions
    posteriors: GridPosteriors | None = field(default=None, compare=False, repr=False)  # keep_posteriors=True
