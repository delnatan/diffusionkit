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
    # The normalized log-likelihood on `options.u_D()` (`GridLikelihoods`), None unless the status is "ok".
    loglik_D: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class GridLikelihoods:
    """Every "ok" track's log-likelihood of D on `options.u_D()`, one row per track in the order of `track_ids`.

    Each row is normalized (its logsumexp over the grid is 0), which makes it also the track's
    posterior under the flat prior in ln D. It is kept as a likelihood because that is how tracks
    combine: a population model adds the rows' logs (`lognormal`, `deconvolve`), where adding
    posteriors would count the prior once per track. The per-row constant never matters.
    """
    track_ids: np.ndarray
    n_frames: np.ndarray  # (len(track_ids),) frames per track: a track's detections
    loglik_D: np.ndarray  # (len(track_ids), n_D)


@dataclass(frozen=True)
class GridDistribution:
    """A distribution of the grid parameter across tracks (a population), on `u`, with posterior draws.

    `weights` is a point estimate and each row of `samples` a posterior draw; every one sums to 1
    over the grid. Any functional's interval is read off the draws: `band` pointwise, `mass` over a
    range, `samples @ a` for anything linear.
    """

    u: np.ndarray
    weights: np.ndarray       # (len(u),)
    samples: np.ndarray       # (n_samples, len(u))

    def band(self, level: float = .68, cumulative: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise (lower, upper) equal-tailed band of the weights, or of their cumulative sum."""
        s = np.cumsum(self.samples, axis=1) if cumulative else self.samples
        q = (1 - level) / 2
        lo, hi = np.quantile(s, [q, 1 - q], axis=0)
        return lo, hi

    def mass(self, lo: float = 0., hi: float = np.inf) -> np.ndarray:
        """(n_samples,) draws of the mass with lo <= D < hi (the parameter's own units, not ln), whole grid cells."""
        with np.errstate(divide="ignore"):
            inside = (self.u >= np.log(lo)) & (self.u < np.log(hi))
        return self.samples[:, inside].sum(axis=1)


@dataclass(frozen=True)
class GridPosteriorAnalysis:
    fits: pl.DataFrame  # one row per track (model posterior_D)
    acquisition: Acquisition | None  # None for a selection across experiments with different acquisitions
    options: GridPostOptions
    likelihoods: GridLikelihoods | None = field(default=None, compare=False, repr=False)  # keep_likelihoods=True
