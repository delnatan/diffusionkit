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
        "D_post_mean_um2_s", "D_post_lo_um2_s", "D_post_hi_um2_s", "D_post_info_bits",
        "D_floor_um2_s")
    parameters: dict[str, float | None]
    status: str
    message: str
    model: str = "posterior_D"
    method: str = "grid_posterior"
    uncertainty_method: str = "credible_interval"
    grid_edge: str | None = None  # `posterior.grid_edge`: "low", "high", "both", or None


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

    def partially_pooled_means(self, loglik_D: np.ndarray, n_draws: int = 200) -> np.ndarray:
        """(n_tracks,) each track's E[D] with this population as its prior (partial pooling), in the
        parameter's own units: the mean of E[D | track, g] over `n_draws` evenly spaced draws of g,
        so the population's own uncertainty is in it.

        `loglik_D` are log-likelihood rows on `u` (any per-row constant). Unlike the flat-prior
        E[D], it does not depend on the grid's edges (the population has no mass there), and its
        average over the tracks the population was fitted to is the population's mean. It is not
        the track's own number, though: it borrows from the population, so it moves with which
        tracks make up the population and with the population model.
        """
        lls = np.asarray(loglik_D, float)
        if lls.ndim != 2 or lls.shape[1] != len(self.u):
            raise ValueError(f"loglik_D must be (n_tracks, {len(self.u)}), got {lls.shape}")
        L = np.exp(lls - lls.max(axis=1, keepdims=True))
        pick = np.linspace(0, len(self.samples) - 1, min(n_draws, len(self.samples))).round().astype(int)
        G = self.samples[pick].T  # (len(u), n_draws)
        return ((L @ (G * np.exp(self.u)[:, None])) / np.maximum(L @ G, 1e-300)).mean(axis=1)


@dataclass(frozen=True)
class GridPosteriorAnalysis:
    fits: pl.DataFrame  # one row per track (model posterior_D)
    acquisition: Acquisition | None  # None for a selection across experiments with different acquisitions
    options: GridPostOptions
    likelihoods: GridLikelihoods | None = field(default=None, compare=False, repr=False)  # keep_likelihoods=True
