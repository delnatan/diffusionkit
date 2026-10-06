"""Classical MSD point estimates. No priors or confidence-interval claims."""
from dataclasses import replace
from numbers import Integral

import numpy as np
from scipy.optimize import least_squares

from .data import MSDCurve, MSDFit


def _arrays(curve: MSDCurve) -> tuple[np.ndarray, np.ndarray]:
    tau, raw, offset = (np.asarray(x, dtype=float) for x in
                        (curve.tau_s, curve.msd_um2, curve.localization_offset_um2))
    if tau.ndim != 1 or raw.shape != tau.shape or offset.shape != tau.shape:
        raise ValueError("MSD time, value, and offset arrays must have matching one-dimensional shapes")
    if not all(np.all(np.isfinite(x)) for x in (tau, raw, offset)):
        raise ValueError("MSD time, value, and offset arrays must be finite")
    if np.any(tau <= 0) or np.any(np.diff(tau) <= 0) or np.any(raw < 0) or np.any(offset < 0):
        raise ValueError("Times must be positive and increasing; raw MSD and offsets must be nonnegative")
    return tau, raw - offset


def empty_fit(model: str, status: str, message: str, n_lags: int = 0) -> MSDFit:
    if model == "linear":
        return MSDFit(model, "msd_ols_intercept",
                      {"D_um2_s": None, "offset_um2": None, "localization_sd_um": None}, status, message, n_lags)
    normal = model == "brownian"
    return MSDFit(model, "msd_ols" if normal else "msd_nls",
                  {"D_um2_s": None} if normal else {"K_um2_s_alpha": None, "alpha": None},
                  status, message, n_lags)


def fit_brownian_msd(curve: MSDCurve) -> MSDFit:
    """OLS through the origin: MSD(lag) - known offset(lag) = 4 D tau.

    Negative D is retained and flagged; clipping would change the estimator.
    Correlated MSD residuals are not used as an uncertainty estimate.
    """
    tau, y = _arrays(curve)
    if len(tau) == 0:
        return empty_fit("brownian", "insufficient_data", "At least one lag is required")
    q = tau / tau[0]
    amplitude = float(np.dot(q, y) / np.dot(q, q))
    D = float(amplitude / (4 * tau[0]))
    status = "nonphysical" if D < 0 else "boundary" if D == 0 else "ok"
    message = "Negative diffusion estimate" if D < 0 else "Zero diffusion estimate" if D == 0 else ""
    return MSDFit("brownian", "msd_ols", {"D_um2_s": D}, status, message, len(tau),
                  float(np.sum((y - amplitude * q)**2)))


def fit_anomalous_msd(curve: MSDCurve, *, max_nfev: int = 200) -> MSDFit:
    """Bounded least squares in linear MSD space: corrected MSD = 4 K tau^alpha.

    Internally fit amplitude at the geometric-mean lag, with K >= 0 and
    0 <= alpha <= 2. Boundary/zero-amplitude solutions are flagged.
    The bounds are model constraints, not a prior. No log(MSD) is taken.
    """
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, Integral) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    tau, y = _arrays(curve)
    if len(tau) < 3:
        return empty_fit("power_law", "insufficient_data", "At least three lags are required", len(tau))
    scale = float(np.max(np.abs(y)))
    if scale == 0 or np.all(y <= 0):
        return replace(empty_fit("power_law", "unidentified", "No positive corrected MSD signal to identify alpha", len(tau)),
                       residual_sum_squares_um4=float(y @ y))
    t_ref = float(np.exp(np.mean(np.log(tau))))
    log_t = np.log(tau / t_ref)
    target = y / scale

    def residual(theta):
        return theta[0] * np.exp(theta[1] * log_t) - target

    def jacobian(theta):
        q = np.exp(theta[1] * log_t)
        return np.column_stack((q, theta[0] * q * log_t))

    # Two-dimensional problem; three starts reduce sensitivity to initial alpha.
    candidates = []
    for alpha0 in (0.5, 1.0, 1.5):
        q = np.exp(alpha0 * log_t)
        amplitude0 = max(float(q @ target / (q @ q)), 1e-3)
        candidates.append(least_squares(residual, [amplitude0, alpha0], jac=jacobian,
                                       bounds=([0., 0.], [np.inf, 2.]), max_nfev=max_nfev,
                                       ftol=1e-10, xtol=1e-10, gtol=1e-10))
    result = min(candidates, key=lambda r: r.cost)
    amplitude, alpha = result.x
    cost = result.cost
    # Evaluate model endpoints exactly: a bounded optimizer can stop just
    # inside a boundary as its scaled gradient tends to zero.
    for endpoint in (0., 2.):
        q = np.exp(endpoint * log_t)
        a = max(float(q @ target / (q @ q)), 0.)
        endpoint_cost = .5 * np.sum((a*q-target)**2)
        if endpoint_cost <= cost:
            amplitude, alpha, cost = a, endpoint, endpoint_cost
    if not result.success or not np.all(np.isfinite(result.x)):
        status, message = "optimizer_failed", result.message
    elif amplitude <= 1e-8 or np.linalg.cond(jacobian([amplitude, alpha])) > 1e10:
        return replace(empty_fit("power_law", "unidentified", "Amplitude or Jacobian does not identify alpha", len(tau)),
                       residual_sum_squares_um4=float(2*cost*scale**2),
                       optimizer_status=int(result.status), nfev=sum(r.nfev for r in candidates))
    elif alpha < 1e-6 or alpha > 2 - 1e-6:
        status, message = "boundary", "Alpha reached a model boundary (0 or 2)"
    else:
        status, message = "ok", ""
    K = float(amplitude * scale / (4 * t_ref**alpha))
    if not np.isfinite(K):
        return empty_fit("power_law", "optimizer_failed", "Nonfinite generalized diffusion coefficient", len(tau))
    return MSDFit("power_law", "msd_nls", {"K_um2_s_alpha": K, "alpha": float(alpha)},
                  status, str(message), len(tau), float(2 * cost * scale**2),
                  int(result.status), sum(r.nfev for r in candidates))


def fit_linear_msd(curve: MSDCurve, n_points: int | None = None) -> MSDFit:
    """Textbook linear fit with a free intercept, MSD = 4 D tau + b, to the curve's raw MSD.

    `n_points` fits the first n lags (default all of the curve). b is the localization offset, 4 sigma^2
    for an isotropic 2D SD sigma; the SDs a curve may carry are not used. Reports `D_um2_s` = slope / 4,
    `offset_um2` = b and `localization_sd_um` = sqrt(b / 4). A negative b is nonphysical: flagged, with the
    SD null. Three lags at least (two parameters). Hand `offset_um2` to `fit_loglog_msd`.
    """
    curve = curve if n_points is None else curve.head(n_points)
    tau, raw = (np.asarray(x, dtype=float) for x in (curve.tau_s, curve.msd_um2))
    if len(tau) < 3:
        return empty_fit("linear", "insufficient_data", "At least three lags are required to fit an intercept", len(tau))
    design = np.column_stack((tau, np.ones_like(tau)))
    (slope, b), *_ = np.linalg.lstsq(design, raw, rcond=None)
    D = float(slope / 4)
    status = "nonphysical" if D < 0 or b < 0 else "boundary" if D == 0 else "ok"
    message = ("Negative diffusion estimate" if D < 0 else "Negative MSD intercept" if b < 0 else
               "Zero diffusion estimate" if D == 0 else "")
    return MSDFit("linear", "msd_ols_intercept",
                  {"D_um2_s": D, "offset_um2": float(b), "localization_sd_um": float(np.sqrt(b / 4)) if b >= 0 else None},
                  status, message, len(tau), float(np.sum((raw - design @ [slope, b]) ** 2)))


def fit_loglog_msd(curve: MSDCurve, offset_um2: float | None = None, n_points: int | None = None) -> MSDFit:
    """Textbook power law: OLS of log(MSD - offset) on log(tau); alpha = slope, 4K = exp(intercept).

    `offset_um2` is subtracted from the raw MSD first: pass `fit_linear_msd(...).parameters["offset_um2"]`
    (clipped at 0 if negative), or leave it None to use the curve's own offset (supplied SDs). It is a
    number, so the two fits stay separate steps on the same curve. `n_points` fits the first n lags.
    Lags whose corrected MSD is not positive cannot be logged; they are dropped and the message counts
    them, which biases alpha upward when they are the informative short lags. Alpha is not constrained to
    [0, 2]: outside it the status is `nonphysical`. At least three usable points. The log of a noisy,
    correlated mean is biased, which is why `fit_anomalous_msd` (linear space) is the per-track default.
    """
    curve = curve if n_points is None else curve.head(n_points)
    tau = np.asarray(curve.tau_s, dtype=float)
    offset = np.asarray(curve.localization_offset_um2, float) if offset_um2 is None else np.full(len(tau), float(offset_um2))
    y = np.asarray(curve.msd_um2, float) - offset
    keep = y > 0
    if keep.sum() < 3:
        return empty_fit("power_law", "insufficient_data",
                         f"{int(keep.sum())} lags with positive corrected MSD; at least three are required", int(keep.sum()))
    x, ly = np.log(tau[keep]), np.log(y[keep])
    (alpha, intercept), *_ = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), ly, rcond=None)
    dropped = int((~keep).sum())
    message = f"{dropped} lag(s) with corrected MSD <= 0 dropped from the log fit" if dropped else ""
    status = "ok"
    if not 0 <= alpha <= 2:
        status, message = "nonphysical", "; ".join(m for m in (f"alpha {alpha:.3g} outside [0, 2]", message) if m)
    return MSDFit("power_law", "msd_loglog", {"K_um2_s_alpha": float(np.exp(intercept) / 4), "alpha": float(alpha)},
                  status, message, int(keep.sum()), float(np.sum((ly - alpha * x - intercept) ** 2)))
