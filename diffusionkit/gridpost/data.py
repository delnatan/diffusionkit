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
    grid to fall back on. The alpha posterior integrates its scale, D at the
    frame interval, over this same D grid and prior.
    """
    min_frames: int = 3  # the whitening step's own hard minimum (>= 3 frames -> >= 2 displacements)
    level: float = .9  # credible-interval mass
    D_min_um2_s: float = 1e-4
    D_max_um2_s: float = 10.
    n_D: int = 501  # ~2.3% steps in D over the default range
    # Avoid alpha's exact 0/2 edges, where the fGn covariance degenerates.
    alpha_min: float = .05
    alpha_max: float = 1.95
    n_alpha: int = 39
    compute_alpha: bool = True  # False: every alpha row is `excluded` ("not requested")
    # The alpha likelihood: "exact" (Gaussian, one eigendecomposition per alpha,
    # O(m^3)), "whittle" (debiased Whittle, one FFT per alpha, O(m log m)), or
    # "auto": exact below `alpha_whittle_min_frames`, Whittle from there on.
    # Whittle is calibrated but ~1-6% less accurate in alpha with constant
    # localization SDs and ~5-13% with per-frame varying ones, at any length;
    # from 40 frames it is >= 2.4x faster, ~10x at 200
    # (scripts/validate_alpha_whittle.py, audit/alpha_whittle_validation.json).
    alpha_method: str = "auto"
    alpha_whittle_min_frames: int = 40

    ALPHA_METHODS: ClassVar[tuple[str, ...]] = ("auto", "exact", "whittle")

    def __post_init__(self):
        if not (np.isfinite(self.D_min_um2_s) and np.isfinite(self.D_max_um2_s)
                and 0 < self.D_min_um2_s < self.D_max_um2_s):
            raise ValueError(f"need 0 < D_min_um2_s < D_max_um2_s, got {self.D_min_um2_s}, {self.D_max_um2_s}")
        if not 0 < self.alpha_min < self.alpha_max < 2:
            raise ValueError(f"need 0 < alpha_min < alpha_max < 2, got {self.alpha_min}, {self.alpha_max}")
        for name in ("n_D", "n_alpha"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least 2")
        if not 0 < self.level < 1:
            raise ValueError(f"level must be in (0, 1), got {self.level}")
        if self.alpha_method not in self.ALPHA_METHODS:
            raise ValueError(f"alpha_method must be one of {self.ALPHA_METHODS}, got {self.alpha_method!r}")
        if self.alpha_whittle_min_frames < 3:
            raise ValueError(f"alpha_whittle_min_frames must be at least 3, got {self.alpha_whittle_min_frames}")

    def u_D(self) -> np.ndarray:
        """The ln D grid (D in um^2/s): the D posterior's, and the alpha posterior's nuisance scale."""
        return np.linspace(np.log(self.D_min_um2_s), np.log(self.D_max_um2_s), self.n_D)

    def alpha_likelihood(self, n_frames: int) -> str:
        """"exact" or "whittle": the alpha likelihood a track of `n_frames` gets."""
        if self.alpha_method == "auto":
            return "whittle" if n_frames >= self.alpha_whittle_min_frames else "exact"
        return self.alpha_method

    def alphas(self) -> np.ndarray:
        """The alpha grid the alpha posterior is evaluated on."""
        return np.linspace(self.alpha_min, self.alpha_max, self.n_alpha)


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
class PosteriorAlpha:
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        "alpha_post_median", "alpha_post_lo", "alpha_post_hi", "alpha_post_info_bits")
    parameters: dict[str, float | None]
    status: str
    message: str
    model: str = "posterior_alpha"
    method: str = "grid_posterior_marginal_D"
    uncertainty_method: str = "credible_interval"


@dataclass(frozen=True)
class TrackPosterior:
    track_id: int
    n_frames: int
    posterior_D: PosteriorD
    posterior_alpha: PosteriorAlpha
    acquisition: Acquisition
    options: GridPostOptions
    # Normalized log posteriors on `options.u_D()` / `options.alphas()`, None
    # unless that posterior's status is "ok".
    log_post_D: np.ndarray | None = field(default=None, compare=False, repr=False)
    log_post_alpha: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class GridPosteriors:
    """Every "ok" track's normalized log posterior, one row per track.

    `log_post_D` rows are on `options.u_D()`, in the order of `D_track_ids`;
    `log_post_alpha` rows on `options.alphas()`, in the order of
    `alpha_track_ids`. The per-track vectors a population read (summed,
    pooled or deconvolved) is built from.
    """
    D_track_ids: np.ndarray
    log_post_D: np.ndarray  # (len(D_track_ids), n_D)
    alpha_track_ids: np.ndarray
    log_post_alpha: np.ndarray  # (len(alpha_track_ids), n_alpha)


@dataclass(frozen=True)
class GridPosteriorAnalysis:
    fits: pl.DataFrame  # one row per (track_id, model): posterior_D, posterior_alpha
    acquisition: Acquisition
    options: GridPostOptions
    posteriors: GridPosteriors | None = field(default=None, compare=False, repr=False)  # keep_posteriors=True
