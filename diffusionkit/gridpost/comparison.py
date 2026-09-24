"""Brownian motion versus localization noise with an unknown global SD scale."""
import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ..data import Acquisition
from .likelihood import _prepared, _whiten


def _motion_lrt(lam: np.ndarray, y: np.ndarray) -> float | None:
    """Twice the profile log-likelihood gain for variances c² + D * lam.

    Both models fit c > 0; the alternative also fits D >= 0. Include the
    c -> 0 limit when taking the alternative's supremum. Profile the overall
    variance analytically, then search the remaining covariance shape in
    log(D/c²). Scan before refining to accommodate multiple local optima.
    This calculation does not use the posterior's D grid or its prior.

    Exactly zero displacements have unbounded likelihood as c -> 0 under
    both models, so their likelihood ratio is undefined (returned as None).
    """
    lam, y = np.ravel(lam), np.ravel(y)
    amplitude = np.max(np.abs(y))
    if amplitude == 0:
        return None
    # Remove the arbitrary data amplitude and eigenvalue units. The fitted
    # common variance absorbs both; this also avoids overflow in y**2.
    energy = (y / amplitude) ** 2
    energy /= energy.mean()
    with np.errstate(divide="ignore"):
        log_energy = np.log(energy)
    log_lam = np.log(lam) - np.log(lam).mean()
    n = len(y)

    def cost(log_variance):
        # Twice negative profile log likelihood, relative to the noise-only
        # model (whose normalized energy has mean one).
        return float(np.sum(log_variance) + n * (
            logsumexp(log_energy - log_variance) - np.log(n)))

    def objective(log_ratio):
        return cost(np.logaddexp(0., log_ratio + log_lam))

    # At either end every mode is within exp(-24) of its limiting shape.
    # Include the exact limits as candidates, rather than bounding D or c.
    grid = np.linspace(-log_lam.max() - 24., -log_lam.min() + 24., 257)
    log_variances = np.logaddexp(0., grid[:, None] + log_lam)
    values = np.sum(log_variances, axis=1) + n * (
        logsumexp(log_energy - log_variances, axis=1) - np.log(n))
    best = min(0., cost(log_lam), float(values.min()))
    for i in range(1, len(grid) - 1):
        if values[i] <= values[i-1] and values[i] <= values[i+1] and (
                values[i] < values[i-1] or values[i] < values[i+1]):
            fit = minimize_scalar(objective, bounds=(grid[i-1], grid[i+1]),
                                  method="bounded", options={"xatol": 1e-9})
            if not fit.success:
                raise RuntimeError("Motion likelihood-ratio optimization failed")
            best = min(best, float(fit.fun))
    return max(0., -best)


def brownian_motion_lrt(track: pl.DataFrame, acquisition: Acquisition) -> float | None:
    """Nonnegative 2*log likelihood ratio: D*A+c²*B versus c²*B.

    One global localization SD multiplier c is shared across axes/frames and
    fitted independently in each hypothesis. Near zero means little fit
    improvement from Brownian motion after rescaling noise. This is neither
    a Bayes factor nor a p-value; short-track thresholds require calibration.
    An exactly constant-position track returns None (unbounded scale fit).
    """
    w = _whiten(_prepared(track, acquisition), acquisition)
    return _motion_lrt(w["lam"], w["y"])
