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


@dataclass(frozen=True)
class AlphaPosteriors:
    """Every "ok" track's marginal log posterior of alpha on `options.alphas()`, one row per track.

    A posterior, not a likelihood: the amplitude is summed out under its flat ln SNR prior, and
    near the localization floor that prior decides how the row leans. Tracks don't combine by
    adding these rows: on simulated populations a deconvolution of them put the share of alpha
    < 0.75 at 0.43 against a true 0.60 (0.05 against 0.49 when alpha and D were correlated).
    What a population model needs is each track's likelihood over (alpha, D) jointly.
    """
    track_ids: np.ndarray
    n_frames: np.ndarray
    log_post_alpha: np.ndarray  # (len(track_ids), n_alpha), each row's logsumexp is 0


@dataclass(frozen=True)
class FbmPosteriorAnalysis:
    fits: pl.DataFrame  # one row per track (model posterior_alpha_D)
    acquisition: Acquisition
    options: FbmPostOptions
    posteriors: AlphaPosteriors | None = field(default=None, compare=False, repr=False)  # keep_posteriors=True
