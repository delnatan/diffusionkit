"""fBm grid-posterior inputs and outputs. Algorithms live in other modules."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import polars as pl

from ..data import Acquisition


@dataclass(frozen=True)
class FbmPostOptions:
    """Everything an fBm grid-posterior run depends on besides the tracks and the acquisition.

    The prior is flat in alpha over [alpha_min, alpha_max] and flat in ln SNR over [snr_min, snr_max],
    where SNR = K ||L^-1 C_alpha L^-t|| is the amplitude in units of the track's own localization
    noise (`posterior.log_likelihood_grid`). Within one alpha that is flat in ln K, ln D or any other
    amplitude, so it changes nothing for a track that measures its amplitude; what it fixes is how
    the bounds fall across alpha. Below detection the likelihood depends on K only through K lam,
    so in SNR units the noise-only stretch is equally long at every alpha, and a track that sees only
    noise returns the flat alpha prior. (Bounds in D at dt lean such tracks to low alpha, and in D at
    the longest lag they still lean, unless the range is many decades wide.)

    `tau_ref_s` only sets what is reported: D_ref = MSD_1D(tau_ref) / (2 tau_ref), the D of the
    true (unblurred) motion at one lag shared by every track and movie. At alpha = 1 it is the
    Brownian D at any tau_ref; otherwise it is part of D_ref's definition, so state it in seconds
    and keep it fixed across datasets with different frame intervals.
    """
    min_frames: int = 3
    level: float = .9  # credible-interval mass
    tau_ref_s: float = .1
    alpha_min: float = .02
    alpha_max: float = 1.98
    n_alpha: int = 99  # steps of 0.02 over the default range, with alpha = 1 on the grid
    snr_min: float = 1e-4
    snr_max: float = 1e8
    n_snr: int = 553  # steps of 0.05 in ln SNR over the default range

    def __post_init__(self):
        if not 0 < self.alpha_min < self.alpha_max < 2:
            raise ValueError(f"need 0 < alpha_min < alpha_max < 2, got {self.alpha_min}, {self.alpha_max}")
        if not (np.isfinite(self.snr_min) and np.isfinite(self.snr_max) and 0 < self.snr_min < self.snr_max):
            raise ValueError(f"need 0 < snr_min < snr_max, got {self.snr_min}, {self.snr_max}")
        if self.n_alpha < 2 or self.n_snr < 2:
            raise ValueError("n_alpha and n_snr must be at least 2")
        if not 0 < self.level < 1:
            raise ValueError(f"level must be in (0, 1), got {self.level}")
        if not self.tau_ref_s > 0:
            raise ValueError(f"tau_ref_s must be positive, got {self.tau_ref_s}")
        if self.min_frames < 3:
            raise ValueError("min_frames must be at least 3")

    def alphas(self) -> np.ndarray:
        """The alpha grid."""
        return np.linspace(self.alpha_min, self.alpha_max, self.n_alpha)

    def u_snr(self) -> np.ndarray:
        """The ln SNR grid the amplitude is evaluated on."""
        return np.linspace(np.log(self.snr_min), np.log(self.snr_max), self.n_snr)


@dataclass(frozen=True)
class PosteriorAlphaD:
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        "alpha_post_mean", "alpha_post_lo", "alpha_post_hi", "alpha_post_sd", "alpha_post_info_bits",
        "alpha_post_p_below_1",
        "D_ref_post_mean_um2_s", "D_ref_post_lo_um2_s", "D_ref_post_hi_um2_s",
        "log10_D_ref_post_mean", "log10_D_ref_post_sd", "alpha_log10_D_ref_corr", "D_ref_floor_um2_s")
    parameters: dict[str, float | None]
    status: str
    message: str
    model: str = "posterior_alpha_D"
    method: str = "grid_posterior"
    uncertainty_method: str = "credible_interval"
    grid_edge: str | None = None  # `gridpost.posterior.grid_edge` of the SNR margin: "low", "high", "both", None


@dataclass(frozen=True)
class TrackPosteriorFbm:
    track_id: int
    n_frames: int
    posterior: PosteriorAlphaD
    acquisition: Acquisition
    options: FbmPostOptions
    # The normalized log marginal posterior of alpha on `options.alphas()` (`AlphaPosteriors`), None unless "ok".
    log_post_alpha: np.ndarray | None = field(default=None, compare=False, repr=False)
    # The likelihood on a `JointGrid`'s cells (`population.joint_cells`), when asked for and "ok".
    joint_loglik: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class AlphaPosteriors:
    """Every "ok" track's marginal log posterior of alpha on `options.alphas()`, one row per track.

    A posterior, not a likelihood: the amplitude is summed out under its flat ln SNR prior, and
    near the localization floor that prior decides how the row leans. Tracks don't combine by
    adding these rows: on simulated populations a deconvolution of them put the share of alpha
    < 0.75 at 0.43 against a true 0.60 (0.05 against 0.49 when alpha and D were correlated).
    A population model needs each track's likelihood over (alpha, D) jointly (`JointLikelihoods`).
    """
    track_ids: np.ndarray
    n_frames: np.ndarray
    log_post_alpha: np.ndarray  # (len(track_ids), n_alpha), each row's logsumexp is 0




@dataclass(frozen=True)
class JointGrid:
    """The (alpha, log10 D_ref) cells a track's likelihood is kept on for a population (`population`).

    Alpha cells group `alpha_group` consecutive rows of `options.alphas()` (0.06 wide on the default
    grid, with alpha = 1 a cell centre); log10 D_ref cells are `d_log10_D_ref` wide over
    [log10_D_ref_min, log10_D_ref_max]. A population is fitted on the part of this range some track
    tells apart from its neighbours (`population.informative_range`), so the range only has to be
    wide enough.
    """
    alpha_group: int = 3
    log10_D_ref_min: float = -6.
    log10_D_ref_max: float = 2.
    d_log10_D_ref: float = .2

    def __post_init__(self):
        if self.alpha_group < 1:
            raise ValueError(f"alpha_group must be at least 1, got {self.alpha_group}")
        if not (self.d_log10_D_ref > 0 and self.log10_D_ref_max - self.log10_D_ref_min >= 3 * self.d_log10_D_ref):
            raise ValueError("need d_log10_D_ref > 0 and at least 3 cells over [log10_D_ref_min, log10_D_ref_max]")

    def alphas(self, options: FbmPostOptions) -> np.ndarray:
        """Alpha cell centres: the mean of each group of `alpha_group` grid rows (a remainder is dropped)."""
        fine = options.alphas()
        n = len(fine) // self.alpha_group
        if n < 3:
            raise ValueError(f"{len(fine)} alpha rows make fewer than 3 cells of {self.alpha_group}")
        return fine[:n * self.alpha_group].reshape(n, self.alpha_group).mean(axis=1)

    def log10_D_ref(self) -> np.ndarray:
        """log10 D_ref cell centres."""
        n = int(round((self.log10_D_ref_max - self.log10_D_ref_min) / self.d_log10_D_ref))
        return self.log10_D_ref_min + self.d_log10_D_ref * (np.arange(n) + .5)


@dataclass(frozen=True)
class JointLikelihoods:
    """Every "ok" track's likelihood over (alpha, log10 D_ref) cells, one (n_alpha, n_D) slab per track.

    Free of priors: each fine grid row (`posterior.log_likelihood_grid`) is moved to the common
    log10 D_ref axis (a shift, no Jacobian) and averaged over each cell. Each slab's maximum is 0.
    """
    track_ids: np.ndarray
    n_frames: np.ndarray
    D_ref_floor_um2_s: np.ndarray  # (n_tracks,) `posterior.localization_floor`
    alphas: np.ndarray             # (n_alpha,) cell centres
    log10_D_ref: np.ndarray        # (n_D,) cell centres, evenly spaced
    loglik: np.ndarray             # (n_tracks, n_alpha, n_D)

    def rows(self, index) -> "JointLikelihoods":
        """The tracks at `index` (an integer index array or a boolean mask)."""
        return JointLikelihoods(self.track_ids[index], self.n_frames[index], self.D_ref_floor_um2_s[index],
                                self.alphas, self.log10_D_ref, self.loglik[index])


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
class FbmPosteriorAnalysis:
    fits: pl.DataFrame  # one row per track (model posterior_alpha_D)
    acquisition: Acquisition
    options: FbmPostOptions
    posteriors: AlphaPosteriors | None = field(default=None, compare=False, repr=False)  # keep_posteriors=True
    likelihoods: JointLikelihoods | None = field(default=None, compare=False, repr=False)  # keep_likelihoods=True
