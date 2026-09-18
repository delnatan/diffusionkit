"""Fit conventional diffusion models to MSD(tau) curves.

Two models, both linear after a transform (so plain weighted least squares,
no nonlinear optimizer needed):

  Normal diffusion:    MSD(tau) = 4*D*tau + b        (fit in linear space)
  Anomalous diffusion: MSD(tau) = 4*K*tau^alpha      (fit in log-log space)

`b` in the normal-diffusion fit is the static-localization/motion-blur
offset: b = 4*sigma_loc^2 - 4*D*R*dt_frame (Michalet & Berglund 2012), where
R=0 for negligible exposure duty cycle (our default assumption; camera
exposure/duty-cycle isn't recorded here). `localization_offset_by_track`
gives the R=0 expected value of b directly from the measured sigma_x/sigma_y, as
an independent check on the fitted intercept.

All fit functions are pure: numpy arrays in, an immutable result out. Batch
fitting over many tracks is just this kernel called once per track via
polars `map_groups`.

`fit_normal_diffusion_gls`/`fit_anomalous_diffusion_gls` are a second,
covariance-weighted way to fit the same two models, using every available
lag (down-weighted by `covariance.py`'s analytic TAMSD covariance) instead
of `n_fit_points`'s ad-hoc truncation. See their docstrings and README for
why, and `n_fit_points`'s docstring for why the OLS path still truncates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .covariance import step_covariance, tamsd_covariance


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
    K_um2_s_alpha: float
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
    """OLS fit of log(MSD) = alpha*log(tau) + log(4*K) over the first n_points.

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
            K_um2_s_alpha=float("nan"),
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
        K_um2_s_alpha=float(np.exp(log4D) / 4.0),
        n_points=len(tau),
        r_squared=r2,
    )


@dataclass(frozen=True)
class NormalDiffusionGLSFit:
    D_um2_s: float
    D_stderr_um2_s: float
    intercept_um2: float
    intercept_stderr_um2: float
    n_points: int
    r_squared: float
    singular: bool


@dataclass(frozen=True)
class AnomalousDiffusionGLSFit:
    alpha: float
    alpha_stderr: float
    K_um2_s_alpha: float
    n_points: int
    r_squared: float
    n_iterations: int
    singular: bool


def n_gls_fit_points(
    track_length: int, n_pairs: np.ndarray, min_points: int = 3
) -> int:
    """Largest number of lags with at least 2 displacement pairs, no cap.

    Unlike `n_fit_points`, this isn't protecting against correlated,
    high-variance tail lags biasing an *unweighted* fit -- the GLS/FGLS
    fits below account for that correlation directly via `covariance.py`.
    The only thing this guards against is a lag with too few pairs (1) to
    contribute a meaningful covariance term at all. Floored at
    `min_points` so degenerate short tracks still get *some* attempt (and
    a chance to be flagged `singular` if that attempt fails), matching
    `n_fit_points`'s floor.
    """
    n_pairs = np.asarray(n_pairs)
    valid = np.nonzero(n_pairs >= 2)[0]
    n_valid = int(valid[-1]) + 1 if len(valid) > 0 else 0
    return max(min_points, min(len(n_pairs), n_valid))


_SINGULAR_COND_THRESHOLD = 1e12


def _gls_solve(
    X: np.ndarray, y: np.ndarray, Sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Generalized least squares: beta = (X'Sigma^-1 X)^-1 X'Sigma^-1 y.

    Returns (beta, stderr, pseudo_r2), or None if `Sigma` is singular or
    too ill-conditioned to trust (see `_SINGULAR_COND_THRESHOLD`) -- a
    per-track signal to flag and skip, not to paper over with a ridge
    term (see fitting.py module docstring / README).

    Misspecifying `Sigma` (it's built from a pilot parameter estimate,
    see the callers below) costs *efficiency*, not correctness: weighted
    least squares is unbiased for any positive-definite weight matrix,
    it's only minimum-variance when the weights match the true
    covariance. So an imperfect pilot degrades this fit's precision, not
    its validity.
    """
    if np.linalg.cond(Sigma) > _SINGULAR_COND_THRESHOLD:
        return None
    try:
        Z = np.linalg.solve(Sigma, X)  # Sigma^-1 X
        v = np.linalg.solve(Sigma, y)  # Sigma^-1 y
    except np.linalg.LinAlgError:
        return None

    A = X.T @ Z
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    beta = A_inv @ (X.T @ v)
    se = np.sqrt(np.diag(A_inv))

    resid = y - X @ beta
    ss_res = float(resid @ np.linalg.solve(Sigma, resid))
    ones = np.ones_like(y)
    w1 = np.linalg.solve(Sigma, ones)
    y_wmean = float(ones @ np.linalg.solve(Sigma, y)) / float(ones @ w1)
    dev = y - y_wmean
    ss_tot = float(dev @ np.linalg.solve(Sigma, dev))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return beta, se, r2


def fit_normal_diffusion_gls(
    tau_s: np.ndarray,
    msd_um2: np.ndarray,
    track_length: int,
    n_pairs: np.ndarray,
    dt_s: float,
    sigma2_um2: float,
    n_points: int | None = None,
) -> NormalDiffusionGLSFit:
    """Covariance-weighted (feasible GLS) fit of MSD = 4*D*tau + b.

    Alternative to `fit_normal_diffusion` + `n_fit_points`'s truncation:
    uses `n_gls_fit_points` lags (no fixed cap) and weights them by the
    analytic TAMSD covariance (`covariance.tamsd_covariance`) instead of
    hard-cutting the fit range.

    One-pass FGLS, not iterated: at alpha=1 (assumed here), the motion
    part of the step covariance is exactly diagonal (independent
    Brownian increments), so the *pilot* D only rescales that diagonal --
    it doesn't change the off-diagonal correlation shape driven by
    `sigma2_um2`. Low sensitivity to the pilot, unlike the anomalous case
    below.
    """
    n = track_length - 1
    npts = (
        n_gls_fit_points(track_length, n_pairs)
        if n_points is None
        else n_points
    )
    tau = np.asarray(tau_s[:npts], dtype=float)
    msd = np.asarray(msd_um2[:npts], dtype=float)
    lags = np.arange(1, npts + 1)

    pilot = fit_normal_diffusion(tau_s, msd_um2, npts, weights=n_pairs[:npts])
    D_pilot = pilot.D_um2_s if pilot.D_um2_s > 0 else 1e-6

    Gamma = step_covariance(n, K=D_pilot, dt_s=dt_s, alpha=1.0, sigma2_um2=sigma2_um2)
    Sigma = tamsd_covariance(Gamma, lags)

    X = np.column_stack([tau, np.ones_like(tau)])
    result = _gls_solve(X, msd, Sigma)
    if result is None:
        return NormalDiffusionGLSFit(
            D_um2_s=float("nan"),
            D_stderr_um2_s=float("nan"),
            intercept_um2=float("nan"),
            intercept_stderr_um2=float("nan"),
            n_points=npts,
            r_squared=float("nan"),
            singular=True,
        )

    beta, se, r2 = result
    slope, intercept = beta
    return NormalDiffusionGLSFit(
        D_um2_s=slope / 4.0,
        D_stderr_um2_s=se[0] / 4.0,
        intercept_um2=float(intercept),
        intercept_stderr_um2=float(se[1]),
        n_points=npts,
        r_squared=r2,
        singular=False,
    )


def fit_anomalous_diffusion_gls(
    tau_s: np.ndarray,
    msd_um2: np.ndarray,
    track_length: int,
    n_pairs: np.ndarray,
    dt_s: float,
    sigma2_um2: float,
    n_points: int | None = None,
    frac_points: float = 0.25,
    min_points: int = 3,
    max_points: int = 10,
    max_iter: int = 2,
    tol: float = 1e-3,
    min_valid_points: int = 3,
) -> AnomalousDiffusionGLSFit:
    """Covariance-weighted (feasible GLS) fit of log(MSD) = alpha*log(tau) + log(4K).

    Genuinely circular, unlike the normal-diffusion case: the TAMSD
    covariance's off-diagonal terms depend on `alpha` itself (fGn
    autocovariance), so the pilot estimate isn't just a scale factor.
    Handled as standard feasible GLS / IRLS: pilot (alpha, K) from the
    unweighted `fit_anomalous_diffusion` -> build Sigma -> delta-method
    into log-space -> solve -> re-pilot from the new alpha and repeat, up
    to `max_iter` times or until alpha changes by less than `tol`.

    The delta method (`Cov[log rho] ~= Cov[rho] / (rho_n * rho_m)`) is a
    first-order approximation that requires each lag's *relative*
    fluctuation to be small -- and it isn't, at the large-lag/few-pair tail
    `fit_normal_diffusion_gls` happily uses the whole track for. Measured
    (see `scripts/validate_gls_recovery.py`, `FINDINGS.md`): confined to
    the same `n_fit_points`-style truncated window OLS uses, this GLS fit
    beats OLS on both alpha bias and variance; extended to the full track
    like the normal-diffusion fit, the approximation breaks down enough at
    the tail to make alpha *worse* than OLS (bias grows with D instead of
    shrinking, variance increases). So unlike `fit_normal_diffusion_gls`,
    this one truncates by default (`frac_points`/`min_points`/`max_points`,
    same rule and defaults as `n_fit_points`) -- the covariance weighting
    inside that window is still a real improvement over OLS, just not a
    license to drop the window entirely for *this* fit.

    The plug-in Sigma is also itself only as good as the current pilot --
    another honest approximation, not exact GLS. This is a cheap step up
    from OLS-on-a-truncated-window, not a substitute for the Bayesian
    arm's joint, no-plug-in estimate of (K, alpha, sigma_loc) from one
    likelihood.
    """
    n = track_length - 1
    npts = (
        n_fit_points(
            len(tau_s), frac=frac_points, min_points=min_points, max_points=max_points
        )
        if n_points is None
        else n_points
    )
    tau_full = np.asarray(tau_s[:npts], dtype=float)
    msd_full = np.asarray(msd_um2[:npts], dtype=float)
    lags_full = np.arange(1, npts + 1)

    pilot = fit_anomalous_diffusion(tau_s, msd_um2, npts, min_valid_points=min_valid_points)
    if not np.isfinite(pilot.alpha) or not np.isfinite(pilot.K_um2_s_alpha):
        return AnomalousDiffusionGLSFit(
            alpha=float("nan"),
            alpha_stderr=float("nan"),
            K_um2_s_alpha=float("nan"),
            n_points=0,
            r_squared=float("nan"),
            n_iterations=0,
            singular=False,
        )

    valid = msd_full > 0
    if valid.sum() < min_valid_points:
        return AnomalousDiffusionGLSFit(
            alpha=float("nan"),
            alpha_stderr=float("nan"),
            K_um2_s_alpha=float("nan"),
            n_points=int(valid.sum()),
            r_squared=float("nan"),
            n_iterations=0,
            singular=False,
        )

    tau = tau_full[valid]
    msd = msd_full[valid]
    lags = lags_full[valid]
    valid_idx = np.nonzero(valid)[0]

    alpha_pilot = float(np.clip(pilot.alpha, 0.02, 1.98))
    K_pilot = pilot.K_um2_s_alpha if pilot.K_um2_s_alpha > 0 else 1e-6

    log_tau = np.log(tau)
    log_msd = np.log(msd)
    X = np.column_stack([log_tau, np.ones_like(log_tau)])

    result = None
    n_iterations = 0
    for _ in range(max_iter):
        n_iterations += 1
        Gamma = step_covariance(
            n, K=K_pilot, dt_s=dt_s, alpha=alpha_pilot, sigma2_um2=sigma2_um2
        )
        Sigma_lin_full = tamsd_covariance(Gamma, lags_full)
        Sigma_lin = Sigma_lin_full[np.ix_(valid_idx, valid_idx)]
        Sigma_log = Sigma_lin / np.outer(msd, msd)

        result = _gls_solve(X, log_msd, Sigma_log)
        if result is None:
            break

        beta, se, r2 = result
        alpha_new, log4K_new = beta
        alpha_new_clipped = float(np.clip(alpha_new, 0.02, 1.98))
        K_new = np.exp(log4K_new) / 4.0
        K_new = K_new if K_new > 0 else 1e-6

        converged = abs(alpha_new_clipped - alpha_pilot) < tol
        alpha_pilot, K_pilot = alpha_new_clipped, K_new
        if converged:
            break

    if result is None:
        return AnomalousDiffusionGLSFit(
            alpha=float("nan"),
            alpha_stderr=float("nan"),
            K_um2_s_alpha=float("nan"),
            n_points=len(tau),
            r_squared=float("nan"),
            n_iterations=n_iterations,
            singular=True,
        )

    beta, se, r2 = result
    alpha_final, log4K_final = beta
    return AnomalousDiffusionGLSFit(
        alpha=float(alpha_final),
        alpha_stderr=float(se[0]),
        K_um2_s_alpha=float(np.exp(log4K_final) / 4.0),
        n_points=len(tau),
        r_squared=r2,
        n_iterations=n_iterations,
        singular=False,
    )


def _weighted_nonlinear_gls_solve(
    tau: np.ndarray,
    y: np.ndarray,
    Sigma: np.ndarray,
    theta0: np.ndarray,
    max_inner_iter: int = 50,
    inner_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Levenberg-Marquardt solve of y ~= 4*exp(log_K)*tau**alpha, weighted by
    Sigma^-1 -- the nonlinear-GLS analog of `_gls_solve`, staying in linear
    MSD space instead of log-linearizing the power law.

    `theta = (log_K, alpha)`; fitting log_K instead of K keeps K positive
    for free (no clipping needed inside the solve, unlike the log-linear
    GLS fit's post-hoc K clamp). Returns (theta, stderr, pseudo_r2), or
    None if `Sigma` is singular/ill-conditioned or no improving step is
    found -- same "flag it, don't force it" contract as `_gls_solve`.
    """
    if np.linalg.cond(Sigma) > _SINGULAR_COND_THRESHOLD:
        return None

    def model(theta: np.ndarray) -> np.ndarray:
        log_K, alpha = theta
        return 4.0 * np.exp(log_K) * tau**alpha

    def wsse(theta: np.ndarray) -> tuple[float, np.ndarray] | tuple[None, None]:
        r = y - model(theta)
        try:
            return float(r @ np.linalg.solve(Sigma, r)), r
        except np.linalg.LinAlgError:
            return None, None

    theta = np.asarray(theta0, dtype=float)
    cur_wsse, resid = wsse(theta)
    if cur_wsse is None:
        return None

    lam = 1e-3
    for _ in range(max_inner_iter):
        f_val = model(theta)
        J = np.column_stack([f_val, f_val * np.log(tau)])  # d f / d(log_K, alpha)
        try:
            JtSinvJ = J.T @ np.linalg.solve(Sigma, J)
            JtSinvr = J.T @ np.linalg.solve(Sigma, resid)
        except np.linalg.LinAlgError:
            return None
        diagJ = np.diag(np.diag(JtSinvJ))

        step_found = False
        new_theta = new_wsse = new_resid = None
        for _ in range(10):  # damping backtrack
            try:
                delta = np.linalg.solve(JtSinvJ + lam * diagJ, JtSinvr)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            new_theta = theta + delta
            new_wsse, new_resid = wsse(new_theta)
            if new_wsse is not None and new_wsse < cur_wsse:
                step_found = True
                break
            lam *= 10

        if not step_found:
            break  # no improving step within budget -- treat as converged

        improvement = cur_wsse - new_wsse
        theta, resid, cur_wsse = new_theta, new_resid, new_wsse
        lam = max(lam / 10, 1e-12)
        if improvement < inner_tol * max(1.0, cur_wsse):
            break

    f_val = model(theta)
    J = np.column_stack([f_val, f_val * np.log(tau)])
    try:
        cov = np.linalg.inv(J.T @ np.linalg.solve(Sigma, J))
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.diag(cov))

    ones = np.ones_like(y)
    try:
        w1 = np.linalg.solve(Sigma, ones)
        y_wmean = float(ones @ np.linalg.solve(Sigma, y)) / float(ones @ w1)
        dev = y - y_wmean
        ss_tot = float(dev @ np.linalg.solve(Sigma, dev))
        r2 = 1.0 - cur_wsse / ss_tot if ss_tot > 0 else float("nan")
    except np.linalg.LinAlgError:
        r2 = float("nan")

    return theta, se, r2


def fit_anomalous_diffusion_nlgls(
    tau_s: np.ndarray,
    msd_um2: np.ndarray,
    track_length: int,
    n_pairs: np.ndarray,
    dt_s: float,
    sigma2_um2: float,
    offset_um2: float | None = None,
    n_points: int | None = None,
    frac_points: float = 0.25,
    min_points: int = 3,
    max_points: int = 10,
    max_iter: int = 2,
    tol: float = 1e-3,
) -> AnomalousDiffusionGLSFit:
    """Covariance-weighted nonlinear GLS fit of MSD = 4*K*tau^alpha directly
    in linear space -- no log transform, unlike `fit_anomalous_diffusion`/
    `fit_anomalous_diffusion_gls`.

    Exists because the log-linear GLS fit's delta-method covariance
    (`Cov[log rho] ~= Cov[rho]/rho^2`) compounds badly with offset
    correction: subtracting the localization offset shrinks the corrected
    MSD's magnitude (most at low D, where the offset *is* most of the raw
    signal), which inflates its *relative* fluctuation right when the
    delta method needs that fluctuation to be small. Measured
    (`scripts/validate_gls_recovery.py` / `FINDINGS.md`): offset-corrected
    log-linear GLS is worse than uncorrected. Fitting the untransformed
    power law directly sidesteps the linearization entirely -- weighting
    is done with the same linear-space `Sigma` the normal-diffusion GLS
    fit already uses, and `offset_um2` (pass the track's estimated
    localization offset, e.g. from `localization_offset_by_track`) can
    make MSD go negative at some lags without the log-fit's "drop
    non-positive points" workaround, since there's no log to take.

    `offset_um2=None` (default) fits the raw, uncorrected MSD -- still
    covariance-weighted, still no log transform, a direct comparison
    point for whether the offset correction itself is doing the work.
    Same feasible-GLS structure as `fit_anomalous_diffusion_gls`
    otherwise: pilot from unweighted `fit_anomalous_diffusion`, build
    Sigma, solve (here: `_weighted_nonlinear_gls_solve`, Levenberg-
    Marquardt instead of a closed-form linear solve), re-pilot and
    repeat up to `max_iter` times.
    """
    n = track_length - 1
    npts = (
        n_fit_points(
            len(tau_s), frac=frac_points, min_points=min_points, max_points=max_points
        )
        if n_points is None
        else n_points
    )
    tau = np.asarray(tau_s[:npts], dtype=float)
    msd = np.asarray(msd_um2[:npts], dtype=float)
    lags = np.arange(1, npts + 1)
    offset = offset_um2 if offset_um2 is not None and np.isfinite(offset_um2) else 0.0
    y = msd - offset

    pilot = fit_anomalous_diffusion(tau_s, msd_um2, npts)
    if not np.isfinite(pilot.alpha) or not np.isfinite(pilot.K_um2_s_alpha):
        return AnomalousDiffusionGLSFit(
            alpha=float("nan"),
            alpha_stderr=float("nan"),
            K_um2_s_alpha=float("nan"),
            n_points=0,
            r_squared=float("nan"),
            n_iterations=0,
            singular=False,
        )

    alpha_pilot = float(np.clip(pilot.alpha, 0.02, 1.98))
    K_pilot = pilot.K_um2_s_alpha if pilot.K_um2_s_alpha > 0 else 1e-6

    result = None
    n_iterations = 0
    for _ in range(max_iter):
        n_iterations += 1
        Gamma = step_covariance(
            n, K=K_pilot, dt_s=dt_s, alpha=alpha_pilot, sigma2_um2=sigma2_um2
        )
        Sigma = tamsd_covariance(Gamma, lags)

        theta0 = np.array([np.log(K_pilot), alpha_pilot])
        result = _weighted_nonlinear_gls_solve(tau, y, Sigma, theta0)
        if result is None:
            break

        theta, se, r2 = result
        log_K_new, alpha_new = theta
        alpha_new_clipped = float(np.clip(alpha_new, 0.02, 1.98))
        K_new = float(np.exp(log_K_new))
        K_new = K_new if K_new > 0 else 1e-6

        converged = abs(alpha_new_clipped - alpha_pilot) < tol
        alpha_pilot, K_pilot = alpha_new_clipped, K_new
        if converged:
            break

    if result is None:
        return AnomalousDiffusionGLSFit(
            alpha=float("nan"),
            alpha_stderr=float("nan"),
            K_um2_s_alpha=float("nan"),
            n_points=npts,
            r_squared=float("nan"),
            n_iterations=n_iterations,
            singular=True,
        )

    theta, se, r2 = result
    log_K_final, alpha_final = theta
    return AnomalousDiffusionGLSFit(
        alpha=float(alpha_final),
        alpha_stderr=float(se[1]),
        K_um2_s_alpha=float(np.exp(log_K_final)),
        n_points=npts,
        r_squared=r2,
        n_iterations=n_iterations,
        singular=False,
    )


def localization_offset_by_track(tracks: pl.DataFrame) -> pl.DataFrame:
    """Per-track expected static-localization MSD offset (R=0, isotropic noise).

    offset = 2*(mean(sigma_x_um^2) + mean(sigma_y_um^2))

    This is the theoretical intercept b of MSD(tau) = 4*D*tau + b when the
    only source of offset is per-frame localization noise with no motion-blur
    correction. Compare it against the empirical fit intercept from
    `fit_normal_diffusion` as a self-consistency check.
    """
    return tracks.group_by("track_id").agg(
        offset_um2=(
            2 * (pl.col("sigma_x_um") ** 2 + pl.col("sigma_y_um") ** 2)
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
        .group_by("track_id")
        .agg(w=pl.col("n_pairs").sum())
        .join(localization_offset, on="track_id")
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

    Also (when `localization_offset` is given): `D_gls_um2_s`/`alpha_gls`
    (covariance-weighted GLS, see `fit_normal_diffusion_gls`/
    `fit_anomalous_diffusion_gls`) and `alpha_nlgls`/`alpha_nlgls_corrected`
    (nonlinear GLS in linear MSD space, see `fit_anomalous_diffusion_nlgls`).
    `alpha_nlgls_corrected` -- covariance-weighted *and* offset-corrected,
    without the log-linear GLS fit's delta-method approximation -- is the
    most accurate classic alpha estimate this module produces (see
    README's "Classic, done properly" and FINDINGS.md for the measurement);
    it's still an addition alongside `alpha`, not a replacement.
    """
    eligible = tamsd.filter(pl.col("track_length") >= min_track_length)

    offset_lookup: dict[int, float] = {}
    if localization_offset is not None:
        offset_lookup = dict(
            zip(
                localization_offset["track_id"].to_list(),
                localization_offset["offset_um2"].to_list(),
            )
        )

    def _fit_one(group: pl.DataFrame) -> pl.DataFrame:
        track_id = group["track_id"][0]
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
            "track_id": [track_id],
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
            "K_um2_s_alpha": [anomalous.K_um2_s_alpha],
            "r2_anomalous": [anomalous.r_squared],
        }

        if localization_offset is not None:
            offset = offset_lookup.get(track_id, float("nan"))
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
                    "K_corrected_um2_s_alpha": [
                        anom_corr.K_um2_s_alpha
                    ],
                    "r2_anomalous_corrected": [anom_corr.r_squared],
                }
            )

            # GLS fits (covariance-weighted, no truncation cap) need
            # sigma_loc^2 -- only computed here since that's what
            # `offset` (= 4*sigma_loc^2, see `localization_offset_by_track`)
            # gives us. dt_s = tau[0] since lag[0] == 1.
            if np.isfinite(offset):
                dt_s = float(tau[0])
                sigma2_um2 = offset / 4.0
                normal_gls = fit_normal_diffusion_gls(
                    tau, msd, track_length, n_pairs, dt_s, sigma2_um2
                )
                # Same truncated window as the OLS anomalous fit above
                # (`npts`), not GLS's own full-range default -- see
                # `fit_anomalous_diffusion_gls`'s docstring for why alpha
                # needs the window and D doesn't.
                anom_gls = fit_anomalous_diffusion_gls(
                    tau, msd, track_length, n_pairs, dt_s, sigma2_um2,
                    n_points=npts,
                )
            else:
                normal_gls = NormalDiffusionGLSFit(
                    float("nan"), float("nan"), float("nan"), float("nan"),
                    0, float("nan"), True,
                )
                anom_gls = AnomalousDiffusionGLSFit(
                    float("nan"), float("nan"), float("nan"), 0,
                    float("nan"), 0, True,
                )
            result.update(
                {
                    "D_gls_um2_s": [normal_gls.D_um2_s],
                    "D_gls_stderr_um2_s": [normal_gls.D_stderr_um2_s],
                    "intercept_gls_um2": [normal_gls.intercept_um2],
                    "n_points_used_gls": [normal_gls.n_points],
                    "r2_gls_normal": [normal_gls.r_squared],
                    "gls_singular": [normal_gls.singular],
                    "alpha_gls": [anom_gls.alpha],
                    "alpha_gls_stderr": [anom_gls.alpha_stderr],
                    "K_gls_um2_s_alpha": [anom_gls.K_um2_s_alpha],
                    "n_points_used_alpha_gls": [anom_gls.n_points],
                    "n_iterations_alpha_gls": [anom_gls.n_iterations],
                    "r2_gls_anomalous": [anom_gls.r_squared],
                    "alpha_gls_singular": [anom_gls.singular],
                }
            )

            # Nonlinear GLS (no log transform, see fit_anomalous_diffusion_nlgls):
            # raw and offset-corrected, same truncated window as alpha/alpha_gls.
            # *_nlgls_corrected is the recommended alpha estimate (see README /
            # FINDINGS.md) -- bias close to the Bayesian fit's, at a fraction of
            # the cost -- but *_nlgls (uncorrected) rides along too, same
            # "always both" convention as alpha/alpha_corrected.
            if np.isfinite(offset):
                nlgls = fit_anomalous_diffusion_nlgls(
                    tau, msd, track_length, n_pairs, dt_s, sigma2_um2,
                    n_points=npts,
                )
                nlgls_corr = fit_anomalous_diffusion_nlgls(
                    tau, msd, track_length, n_pairs, dt_s, sigma2_um2,
                    offset_um2=offset, n_points=npts,
                )
            else:
                nlgls = AnomalousDiffusionGLSFit(
                    float("nan"), float("nan"), float("nan"), 0,
                    float("nan"), 0, True,
                )
                nlgls_corr = nlgls
            result.update(
                {
                    "alpha_nlgls": [nlgls.alpha],
                    "alpha_nlgls_stderr": [nlgls.alpha_stderr],
                    "K_nlgls_um2_s_alpha": [nlgls.K_um2_s_alpha],
                    "n_points_used_nlgls": [nlgls.n_points],
                    "n_iterations_nlgls": [nlgls.n_iterations],
                    "r2_nlgls": [nlgls.r_squared],
                    "nlgls_singular": [nlgls.singular],
                    "alpha_nlgls_corrected": [nlgls_corr.alpha],
                    "alpha_nlgls_corrected_stderr": [nlgls_corr.alpha_stderr],
                    "K_nlgls_corrected_um2_s_alpha": [nlgls_corr.K_um2_s_alpha],
                    "n_points_used_nlgls_corrected": [nlgls_corr.n_points],
                    "n_iterations_nlgls_corrected": [nlgls_corr.n_iterations],
                    "r2_nlgls_corrected": [nlgls_corr.r_squared],
                    "nlgls_corrected_singular": [nlgls_corr.singular],
                }
            )

        return pl.DataFrame(result)

    return eligible.group_by("track_id", maintain_order=True).map_groups(
        _fit_one
    )
