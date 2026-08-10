"""Fit conventional diffusion models to MSD(tau) curves.

Two models, both linear after a transform (so plain weighted least squares,
no nonlinear optimizer needed):

  Normal diffusion:    MSD(tau) = 4*D*tau + b            (fit in linear space)
  Anomalous diffusion: MSD(tau) = 4*D_alpha * tau^alpha   (fit in log-log space)

`b` in the normal-diffusion fit is the static-localization/motion-blur
offset: b = 4*sigma_loc^2 - 4*D*R*dt_frame (Michalet & Berglund 2012), where
R=0 for negligible exposure duty cycle (our default assumption; camera
exposure/duty-cycle isn't recorded here). `localization_offset_by_track`
gives the R=0 expected value of b directly from the measured x_std/y_std, as
an independent check on the fitted intercept.

All fit functions are pure: numpy arrays in, an immutable result out. Batch
fitting over many tracks is just this kernel called once per particle via
polars `map_groups`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class NormalDiffusionFit:
    D_um2_s: float
    D_stderr_um2_s: float
    intercept_um2: float
    intercept_stderr_um2: float
    n_points: int
    r_squared: float


@dataclass(frozen=True)
class AnomalousDiffusionFit:
    alpha: float
    alpha_stderr: float
    D_alpha_um2_s_alpha: float
    n_points: int
    r_squared: float


def n_fit_points(
    n_lags: int, frac: float = 0.25, min_points: int = 3, max_points: int = 10
) -> int:
    """Standard heuristic: fit using only the first `frac` of available lags.

    MSD(tau) at large lag is estimated from few, highly overlapping (and thus
    correlated) displacement pairs, so its variance grows with lag; fitting
    too far out biases both D and the intercept (Saxton 1997; Michalet 2010).
    `frac` alone is not enough to prevent this on long tracks, since a fixed
    fraction of a long track can still include many noisy large-lag points --
    hence the `max_points` cap on top of the fractional rule. See FINDINGS.md
    for a worked example of how large the resulting bias can get.
    """
    return max(
        min_points, min(n_lags, max_points, int(np.floor(n_lags * frac)))
    )


def fit_normal_diffusion(
    tau_s: np.ndarray,
    msd_um2: np.ndarray,
    n_points: int,
    weights: np.ndarray | None = None,
) -> NormalDiffusionFit:
    """Weighted least-squares fit of MSD = 4*D*tau + b over the first n_points."""
    tau = np.asarray(tau_s[:n_points], dtype=float)
    msd = np.asarray(msd_um2[:n_points], dtype=float)
    w = (
        np.ones_like(tau)
        if weights is None
        else np.asarray(weights[:n_points], dtype=float)
    )

    X = np.column_stack([tau, np.ones_like(tau)])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], msd * sw, rcond=None)
    slope, intercept = beta

    resid = msd - X @ beta
    ss_res = float(np.sum(w * resid**2))
    wmean = np.average(msd, weights=w)
    ss_tot = float(np.sum(w * (msd - wmean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dof = max(len(tau) - 2, 1)
    sigma2 = ss_res / dof
    XtWX_inv = np.linalg.inv((X * w[:, None]).T @ X)
    se = np.sqrt(np.diag(XtWX_inv) * sigma2)

    return NormalDiffusionFit(
        D_um2_s=slope / 4.0,
        D_stderr_um2_s=se[0] / 4.0,
        intercept_um2=float(intercept),
        intercept_stderr_um2=float(se[1]),
        n_points=n_points,
        r_squared=r2,
    )


def fit_anomalous_diffusion(
    tau_s: np.ndarray,
    msd_um2: np.ndarray,
    n_points: int,
    min_valid_points: int = 3,
) -> AnomalousDiffusionFit:
    """OLS fit of log(MSD) = alpha*log(tau) + log(4*D_alpha) over the first n_points.

    Non-positive MSD values (e.g. after subtracting a localization-offset
    correction -- see `fit_all_tracks`) are dropped before the fit, since
    log() of them is undefined; `n_points` on the returned result reflects
    how many points actually survived. If fewer than `min_valid_points`
    remain, returns an all-NaN result rather than fitting an
    under-determined or degenerate line.
    """
    tau = np.asarray(tau_s[:n_points], dtype=float)
    msd = np.asarray(msd_um2[:n_points], dtype=float)
    valid = msd > 0
    tau, msd = tau[valid], msd[valid]

    if len(tau) < min_valid_points:
        return AnomalousDiffusionFit(
            alpha=float("nan"),
            alpha_stderr=float("nan"),
            D_alpha_um2_s_alpha=float("nan"),
            n_points=len(tau),
            r_squared=float("nan"),
        )

    log_tau = np.log(tau)
    log_msd = np.log(msd)

    X = np.column_stack([log_tau, np.ones_like(log_tau)])
    beta, *_ = np.linalg.lstsq(X, log_msd, rcond=None)
    alpha, log4D = beta

    resid = log_msd - X @ beta
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((log_msd - log_msd.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dof = max(len(tau) - 2, 1)
    sigma2 = ss_res / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)

    return AnomalousDiffusionFit(
        alpha=float(alpha),
        alpha_stderr=float(se[0]),
        D_alpha_um2_s_alpha=float(np.exp(log4D) / 4.0),
        n_points=len(tau),
        r_squared=r2,
    )


def localization_offset_by_track(tracks: pl.DataFrame) -> pl.DataFrame:
    """Per-track expected static-localization MSD offset (R=0, isotropic noise).

    offset = 2*(mean(x_std_um^2) + mean(y_std_um^2))

    This is the theoretical intercept b of MSD(tau) = 4*D*tau + b when the
    only source of offset is per-frame localization noise with no motion-blur
    correction. Compare it against the empirical fit intercept from
    `fit_normal_diffusion` as a self-consistency check.
    """
    return tracks.group_by("particle").agg(
        offset_um2=(
            2 * (pl.col("x_std_um") ** 2 + pl.col("y_std_um") ** 2)
        ).mean()
    )


def weighted_expected_offset(
    tamsd: pl.DataFrame, localization_offset: pl.DataFrame, n_points: int
) -> float:
    """Expected localization offset, weighted the same way the ensemble MSD
    curve is (each track weighted by its total n_pairs across the first
    n_points lags).

    A plain per-track mean of `localization_offset_by_track` is dominated by
    short, poorly-localized tracks, while `ensemble_average_msd` weights each
    lag by n_pairs (favoring long, well-tracked particles). Comparing a fitted
    ensemble intercept against a plain mean offset is then apples-to-oranges;
    this puts both on the same weighting.
    """
    weights = (
        tamsd.filter(pl.col("lag") <= n_points)
        .group_by("particle")
        .agg(w=pl.col("n_pairs").sum())
        .join(localization_offset, on="particle")
    )
    w = weights["w"].to_numpy()
    o = weights["offset_um2"].to_numpy()
    return float(np.average(o, weights=w))


def fit_all_tracks(
    tamsd: pl.DataFrame,
    min_track_length: int = 10,
    frac_points: float = 0.25,
    min_points: int = 3,
    max_points: int = 10,
    localization_offset: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Fit normal- and anomalous-diffusion models to every eligible track.

    Tracks shorter than `min_track_length` are excluded (too few lags for a
    meaningful fit).

    No row is dropped or clamped for being unphysical (e.g. D < 0, which the
    linear model can return on a noisy short curve) -- garbage-in-garbage-out
    fits are kept and flagged instead (`D_negative`, `intercept_negative`,
    `at_min_points`), so silent filtering doesn't bias downstream summary
    statistics without a record of what was excluded and why.

    If `localization_offset` is given (see `localization_offset_by_track`),
    the anomalous fit is *additionally* computed on offset-subtracted MSD
    (MSD - 2*(sigma_x^2+sigma_y^2)), in `*_corrected` columns, alongside the
    uncorrected `alpha` -- always both, never a mode switch, so the two can
    be compared directly per track rather than requiring two separate runs.
    Taking log(MSD) conflates the tau^alpha signal with the tau-independent
    localization offset (log(a*x+b) != log(a*x) + const); subtracting a
    known offset first is the fix, but is only as good as that offset
    estimate, and can push points non-positive for slow/short tracks --
    `fit_anomalous_diffusion` drops those points and flags tracks where too
    few survive via `n_points_used_alpha_corrected`.
    """
    eligible = tamsd.filter(pl.col("track_length") >= min_track_length)

    offset_lookup: dict[int, float] = {}
    if localization_offset is not None:
        offset_lookup = dict(
            zip(
                localization_offset["particle"].to_list(),
                localization_offset["offset_um2"].to_list(),
            )
        )

    def _fit_one(group: pl.DataFrame) -> pl.DataFrame:
        particle = group["particle"][0]
        track_length = group["track_length"][0]
        tau = group["tau_s"].to_numpy()
        msd = group["msd_um2"].to_numpy()
        n_pairs = group["n_pairs"].to_numpy()
        npts = n_fit_points(
            len(tau),
            frac=frac_points,
            min_points=min_points,
            max_points=max_points,
        )

        normal = fit_normal_diffusion(tau, msd, npts, weights=n_pairs)
        anomalous = fit_anomalous_diffusion(tau, msd, npts)

        result = {
            "particle": [particle],
            "track_length": [track_length],
            "n_points_used": [npts],
            "at_min_points": [npts == min_points],
            "D_um2_s": [normal.D_um2_s],
            "D_stderr_um2_s": [normal.D_stderr_um2_s],
            "intercept_um2": [normal.intercept_um2],
            "intercept_stderr_um2": [normal.intercept_stderr_um2],
            "r2_normal": [normal.r_squared],
            "D_negative": [normal.D_um2_s < 0],
            "intercept_negative": [normal.intercept_um2 < 0],
            "alpha": [anomalous.alpha],
            "alpha_stderr": [anomalous.alpha_stderr],
            "n_points_used_alpha": [anomalous.n_points],
            "D_alpha_um2_s_alpha": [anomalous.D_alpha_um2_s_alpha],
            "r2_anomalous": [anomalous.r_squared],
        }

        if localization_offset is not None:
            offset = offset_lookup.get(particle, float("nan"))
            if np.isfinite(offset):
                anom_corr = fit_anomalous_diffusion(tau, msd - offset, npts)
            else:
                anom_corr = AnomalousDiffusionFit(
                    float("nan"), float("nan"), float("nan"), 0, float("nan")
                )
            result.update(
                {
                    "alpha_corrected": [anom_corr.alpha],
                    "alpha_corrected_stderr": [anom_corr.alpha_stderr],
                    "n_points_used_alpha_corrected": [anom_corr.n_points],
                    "D_alpha_corrected_um2_s_alpha": [
                        anom_corr.D_alpha_um2_s_alpha
                    ],
                    "r2_anomalous_corrected": [anom_corr.r_squared],
                }
            )

        return pl.DataFrame(result)

    return eligible.group_by("particle", maintain_order=True).map_groups(
        _fit_one
    )
