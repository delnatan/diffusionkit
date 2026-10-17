"""fBm profile-likelihood inputs and outputs. Algorithms live in other modules."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import polars as pl

from ..data import Acquisition


@dataclass(frozen=True)
class FbmPostOptions:
    """Everything an fBm profile-likelihood run depends on besides the tracks and the acquisition.

    alpha is evaluated on `alphas()`, n_alpha points over [alpha_min, alpha_max]; the amplitude is
    profiled out at each (`profile`), so no prior enters. `level` is the likelihood-ratio intervals'
    coverage: alpha within chi2_1(level) / 2 of the maximum (1.92 nats at 0.95).

    `tau_ref_s` only sets what is reported: D_ref = MSD_1D(tau_ref) / (2 tau_ref), the D of the
    true (unblurred) motion at one lag shared by every track and movie. At alpha = 1 it is the
    Brownian D at any tau_ref; otherwise it is part of D_ref's definition, so state it in seconds
    and keep it fixed across datasets with different frame intervals.
    """
    min_frames: int = 3
    level: float = .95
    tau_ref_s: float = .1
    alpha_min: float = .02
    alpha_max: float = 1.98
    n_alpha: int = 99  # steps of 0.02 over the default range, with alpha = 1 on the grid

    def __post_init__(self):
        if not 0 < self.alpha_min < self.alpha_max < 2:
            raise ValueError(f"need 0 < alpha_min < alpha_max < 2, got {self.alpha_min}, {self.alpha_max}")
        if self.n_alpha < 3:
            raise ValueError("n_alpha must be at least 3")
        if not 0 < self.level < 1:
            raise ValueError(f"level must be in (0, 1), got {self.level}")
        if not self.tau_ref_s > 0:
            raise ValueError(f"tau_ref_s must be positive, got {self.tau_ref_s}")
        if self.min_frames < 3:
            raise ValueError("min_frames must be at least 3")

    def alphas(self) -> np.ndarray:
        """The alpha grid."""
        return np.linspace(self.alpha_min, self.alpha_max, self.n_alpha)


@dataclass(frozen=True)
class ProfileAlphaD:
    """One track's profile-likelihood fit. `alpha_hat` is None when the track shows no motion above its
    localization noise (its likelihood peaks at K = 0, the same at every alpha); D_ref_lo is 0 when the track
    only bounds D_ref from above."""
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        "alpha_hat", "alpha_lo", "alpha_hi", "alpha_dll_brownian", "alpha_p_brownian",
        "K_hat", "D_ref_hat_um2_s", "D_ref_lo_um2_s", "D_ref_hi_um2_s", "log10_D_ref_hat",
        "D_ref_floor_um2_s", "loglik_max")
    parameters: dict[str, float | None]
    status: str
    message: str
    model: str = "fbm_alpha_D"
    method: str = "profile_likelihood"
    uncertainty_method: str = "likelihood_ratio_interval"
    alpha_edge: str | None = None  # which end of the alpha grid the interval reaches: "low", "high", "both", None


@dataclass(frozen=True)
class TrackFitFbm:
    track_id: int
    n_frames: int
    fit: ProfileAlphaD
    acquisition: Acquisition
    options: FbmPostOptions
    # l_p(alpha) - max l_p on `options.alphas()` (`AlphaProfiles`), None unless "ok".
    profile: np.ndarray | None = field(default=None, compare=False, repr=False)
    # The likelihood on a `JointGrid`'s cells (`likelihood.cells`), when asked for and "ok".
    joint_loglik: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class AlphaProfiles:
    """Every "ok" track's profile log-likelihood of alpha on `alphas`, one row per track, each row's maximum 0.

    A likelihood with the track's own amplitude maximized at each alpha, so rows add: their sum is the
    profile of one alpha shared by every track (`population.shared_alpha`). They are not a population model's
    input: a distribution of alpha across tracks is fitted jointly with D_ref (`JointLikelihoods`).
    """
    track_ids: np.ndarray
    n_frames: np.ndarray
    alphas: np.ndarray
    loglik: np.ndarray  # (len(track_ids), n_alpha)

    def rows(self, index) -> "AlphaProfiles":
        """The tracks at `index` (an integer index array or a boolean mask)."""
        return AlphaProfiles(self.track_ids[index], self.n_frames[index], self.alphas, self.loglik[index])


@dataclass(frozen=True)
class SharedAlpha:
    """One alpha shared by every track (complete pooling): the maximum of the summed profiles and its
    likelihood-ratio interval. `alphas` and `loglik` are the summed curve on a fine grid (maximum 0)."""
    alpha_hat: float
    alpha_lo: float
    alpha_hi: float
    dll_brownian: float  # summed max - summed value at alpha = 1
    p_brownian: float
    n_tracks: int
    level: float
    at_grid_edge: bool  # the interval reaches an end of the alpha grid
    alphas: np.ndarray = field(compare=False, repr=False)
    loglik: np.ndarray = field(compare=False, repr=False)

    def summary(self) -> dict[str, float]:
        return {"alpha": self.alpha_hat, "lo": self.alpha_lo, "hi": self.alpha_hi,
                "dll_brownian": self.dll_brownian, "p_brownian": self.p_brownian}


@dataclass(frozen=True)
class JointGrid:
    """The (alpha, log10 D) cells a track's likelihood is kept on for a population (`population`).

    Alpha cells group `alpha_group` consecutive rows of `options.alphas()` (0.06 wide on the default
    grid, with alpha = 1 a cell centre); log10 D cells are `d_log10_D_ref` wide over
    [log10_D_ref_min, log10_D_ref_max]. A population is fitted on the part of this range some track
    tells apart from its neighbours (`population.informative_range`), so the range only has to be
    wide enough.

    D on the cells is the D of the motion at the lag `tau_s` (default: the frame interval), not at the
    reported `options.tau_ref_s`: D = K tau_s^(alpha - 1). A track's likelihood is a ridge in
    (alpha, D) along which the MSD at the lags it measures stays fixed; at the frame interval that ridge
    is nearly flat in D, so neighbouring D cells hold the same alphas. At tau_ref = 0.1 s, far from a
    23 ms frame, the ridge is steep, and a narrow population (one bead size, alpha = 1) that falls
    between two D cells is shared between them at two alphas (0.82 and 1.06 on 0.2-decade cells).
    `population.deconvolve_tracks` reports the population at tau_ref.

    The cells must also be narrower than the sharpest tracks' likelihood across D: a long track pins D at
    fixed alpha to ~0.07 decades, hence 0.05. `population.deconvolve_tracks` keeps that affordable by
    refining only where a coarse fit puts mass.
    """
    alpha_group: int = 3
    log10_D_ref_min: float = -6.
    log10_D_ref_max: float = 2.
    d_log10_D_ref: float = .05
    tau_s: float | None = None

    def __post_init__(self):
        if self.alpha_group < 1:
            raise ValueError(f"alpha_group must be at least 1, got {self.alpha_group}")
        if not (self.d_log10_D_ref > 0 and self.log10_D_ref_max - self.log10_D_ref_min >= 3 * self.d_log10_D_ref):
            raise ValueError("need d_log10_D_ref > 0 and at least 3 cells over [log10_D_ref_min, log10_D_ref_max]")
        if self.tau_s is not None and not self.tau_s > 0:
            raise ValueError(f"tau_s must be positive, got {self.tau_s}")

    def lag(self, acquisition: Acquisition) -> float:
        """The lag D is set at on the cells: `tau_s`, or the frame interval."""
        return float(acquisition.dt_s) if self.tau_s is None else self.tau_s

    def alphas(self, options: FbmPostOptions) -> np.ndarray:
        """Alpha cell centres: the mean of each group of `alpha_group` grid rows (a remainder is dropped)."""
        fine = options.alphas()
        n = len(fine) // self.alpha_group
        if n < 3:
            raise ValueError(f"{len(fine)} alpha rows make fewer than 3 cells of {self.alpha_group}")
        return fine[:n * self.alpha_group].reshape(n, self.alpha_group).mean(axis=1)

    def log10_D_ref(self) -> np.ndarray:
        """log10 D cell centres (D at `lag`)."""
        n = int(round((self.log10_D_ref_max - self.log10_D_ref_min) / self.d_log10_D_ref))
        return self.log10_D_ref_min + self.d_log10_D_ref * (np.arange(n) + .5)


@dataclass(frozen=True)
class JointLikelihoods:
    """Every "ok" track's likelihood over (alpha, log10 D) cells, one (n_alpha, n_D) slab per track.

    Free of priors: the likelihood is evaluated on points of each cell and averaged over it
    (`likelihood.cells`). Each slab's maximum is 0. D on the cells is at the lag `tau_s` (`JointGrid`);
    the floors and the population are at `tau_ref_s`.
    """
    track_ids: np.ndarray
    n_frames: np.ndarray
    D_ref_floor_um2_s: np.ndarray  # (n_tracks,) `likelihood.localization_floor`, at tau_ref_s
    alphas: np.ndarray             # (n_alpha,) cell centres
    log10_D_ref: np.ndarray        # (n_D,) cell centres of log10 D at tau_s, evenly spaced
    loglik: np.ndarray             # (n_tracks, n_alpha, n_D)
    tau_s: float                   # the lag D is set at on the cells
    tau_ref_s: float               # the lag the population is reported at (`FbmPostOptions.tau_ref_s`)

    def rows(self, index) -> "JointLikelihoods":
        """The tracks at `index` (an integer index array or a boolean mask)."""
        return JointLikelihoods(self.track_ids[index], self.n_frames[index], self.D_ref_floor_um2_s[index],
                                self.alphas, self.log10_D_ref, self.loglik[index], self.tau_s, self.tau_ref_s)


@dataclass(frozen=True)
class AlphaDPopulation:
    """The distribution of (alpha, log10 D_ref) across tracks (`population.deconvolve_tracks`).

    `weights` is the posterior mode and each of `samples` a posterior draw; each sums to 1 over the
    cells. Every method returning an array of length n_samples gives one value per draw, so its
    interval is read off with quantiles. Below `log10_floor` the tracks barely tell D_ref values
    apart, and alpha there is mostly the smoothness prior's continuation of alpha above: the
    `above_floor` reads leave that part out.
    """
    alphas: np.ndarray
    log10_D_ref: np.ndarray
    weights: np.ndarray       # (n_alpha, n_D)
    samples: np.ndarray       # (n_samples, n_alpha, n_D)
    lam: tuple[float, float]  # smoothness along alpha and log10 D_ref at the evidence maximum
    n_tracks: int
    log10_floor: float        # median over the tracks of log10 `D_ref_floor_um2_s`

    def _draws(self, above_floor: bool) -> np.ndarray:
        if not above_floor:
            return self.samples
        s = self.samples * (self.log10_D_ref > self.log10_floor)
        return s / s.sum(axis=(1, 2), keepdims=True)

    def alpha_weights(self, above_floor: bool = False) -> np.ndarray:
        """(n_alpha,) alpha's margin at the mode."""
        w = self.weights * (self.log10_D_ref > self.log10_floor) if above_floor else self.weights
        return w.sum(axis=1) / w.sum()

    def alpha_samples(self, above_floor: bool = False) -> np.ndarray:
        """(n_samples, n_alpha) alpha's margin per draw."""
        return self._draws(above_floor).sum(axis=2)

    def alpha_band(self, level: float = .9, above_floor: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise equal-tailed band of alpha's margin."""
        q = (1 - level) / 2
        lo, hi = np.quantile(self.alpha_samples(above_floor), [q, 1 - q], axis=0)
        return lo, hi

    def p_alpha_below(self, cut: float = 1., above_floor: bool = False) -> np.ndarray:
        """(n_samples,) mass with alpha < cut, cells split linearly at the cut."""
        half = (self.alphas[1] - self.alphas[0]) / 2
        share = np.clip((cut - (self.alphas - half)) / (2 * half), 0, 1)
        return self.alpha_samples(above_floor) @ share

    def alpha_mean(self, above_floor: bool = False) -> np.ndarray:
        return self.alpha_samples(above_floor) @ self.alphas

    def alpha_sd(self, above_floor: bool = False) -> np.ndarray:
        p = self.alpha_samples(above_floor)
        m = p @ self.alphas
        return np.sqrt(np.maximum(p @ self.alphas ** 2 - m ** 2, 0.))

    def corr(self, above_floor: bool = False) -> np.ndarray:
        """(n_samples,) correlation of alpha and log10 D_ref across the population."""
        s = self._draws(above_floor)
        a, d = self.alphas[None, :, None], self.log10_D_ref[None, None, :]
        ma, md = (s * a).sum((1, 2)), (s * d).sum((1, 2))
        va, vd = (s * a ** 2).sum((1, 2)) - ma ** 2, (s * d ** 2).sum((1, 2)) - md ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return ((s * a * d).sum((1, 2)) - ma * md) / np.sqrt(va * vd)

    def mass_below_floor(self) -> np.ndarray:
        """(n_samples,) the share of the population with D_ref below the floor."""
        return self.samples[:, :, self.log10_D_ref <= self.log10_floor].sum(axis=(1, 2))


@dataclass(frozen=True)
class FbmAnalysis:
    fits: pl.DataFrame  # one row per track (model fbm_alpha_D)
    acquisition: Acquisition
    options: FbmPostOptions
    profiles: AlphaProfiles | None = field(default=None, compare=False, repr=False)  # keep_profiles=True
    likelihoods: JointLikelihoods | None = field(default=None, compare=False, repr=False)  # keep_likelihoods=True
