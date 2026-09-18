"""Brownian displacement likelihood with known per-frame localization SDs.

Per axis, the m = n - 1 displacements of one track are Gaussian with

    Sigma(D) = D A + B,

where A is Brownian motion averaged over the exposure (Berglund 2010) and B
comes from independent localization errors: B_ii = s_i² + s_(i+1)²,
B_(i,i+1) = -s_(i+1)². D >= 0; D = 0 means localization noise alone.

With B = L Lᵀ and L⁻¹ A L⁻ᵀ = Q diag(lam) Qᵀ, the whitened data
y = Qᵀ L⁻¹ Δ have independent components of variance 1 + D lam_k. The
likelihood, its derivatives and parametric-bootstrap replicates are then
cheap functions of D.

The non-Brownian score is the efficient score for the fBm exponent alpha at
alpha = 1 (the Brownian model), evaluated at the MLE of D. It is calibrated
per track by a parametric bootstrap from the fitted Brownian model.
"""
from functools import lru_cache
from numbers import Integral

import numpy as np
from scipy.integrate import quad
from scipy.linalg import cholesky, eigh, solve_triangular, toeplitz
from scipy.optimize import brentq
from scipy.stats import chi2, norm

from ..data import Acquisition, Track
from ..validation import validate_acquisition, validated_track
from .data import BrownianMLE, MLEOptions

_GRID = np.concatenate([[0.], np.logspace(-9, 0, 181)])
_GOLDEN = (np.sqrt(5) - 1) / 2
_CHUNK = 256


def validate_mle_options(options: MLEOptions) -> None:
    for name, lower in (("n_boot", 0), ("seed", 0)):
        value = getattr(options, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if not 0.5 < options.upper_level < 1:
        raise ValueError("upper_level must be between 0.5 and 1")


def _xlogx(u):
    u = np.abs(u)
    return np.where(u > 0, u * np.log(np.where(u > 0, u, 1.)), 0.)


def _exposure_mean(f, x: float, exposure_s: float) -> float:
    """E f(x + w), w = u - v with u, v uniform on [0, exposure_s] (triangular)."""
    if exposure_s == 0:
        return float(f(x))
    density = lambda w: (1 - abs(w) / exposure_s) / exposure_s  # noqa: E731
    inner = [p for p in (-x, 0.) if -exposure_s < p < exposure_s]
    return quad(lambda w: density(w) * f(x + w), -exposure_s, exposure_s,
                points=inner or None, limit=200, epsabs=0, epsrel=1e-12)[0]


@lru_cache(maxsize=256)
def motion_kernels(n_disp: int, dt_s: float, exposure_s: float = 0.) -> tuple[np.ndarray, np.ndarray]:
    """(A, H) for n_disp displacements of exposure-averaged positions.

    fBm per axis has Cov(X_s, X_t) = K(|s|^a + |t|^a - |t - s|^a), so
    displacement covariance at lag k is K[phi((k+1)dt) + phi((k-1)dt) - 2 phi(k dt)]
    with phi(x) = E|x + w|^a. A is this at a = 1 per unit D; H is its
    derivative in a at a = 1 per unit K, using E[|x + w| ln|x + w|].
    """
    if exposure_s == 0:
        # Closed form; the quadrature path is checked against it in tests.
        A = 2 * dt_s * np.eye(n_disp)
        x = np.arange(n_disp + 1) * dt_s
        psi = _xlogx(x)
    else:
        x = np.arange(n_disp + 1) * dt_s
        phi = np.array([_exposure_mean(np.abs, xi, exposure_s) for xi in x])
        psi = np.array([_exposure_mean(_xlogx, xi, exposure_s) for xi in x])
        k = np.arange(n_disp)
        A = toeplitz(phi[k + 1] + phi[np.abs(k - 1)] - 2 * phi[k])
    k = np.arange(n_disp)
    H = toeplitz(psi[k + 1] + psi[np.abs(k - 1)] - 2 * psi[k])
    A.setflags(write=False)
    H.setflags(write=False)
    return A, H


def localization_covariance(sd_um: np.ndarray) -> np.ndarray:
    """(2, m, m) displacement covariance from independent per-frame position SDs."""
    var = np.asarray(sd_um, dtype=float).T ** 2  # (2, n)
    m = var.shape[1] - 1
    B = np.zeros((2, m, m))
    i = np.arange(m)
    B[:, i, i] = var[:, :-1] + var[:, 1:]
    B[:, i[:-1], i[1:]] = B[:, i[1:], i[:-1]] = -var[:, 1:-1]
    return B


def _prepared(track: Track, acquisition: Acquisition):
    validate_acquisition(acquisition, allow_exposure=True)
    track = validated_track(track, require_localization=True)
    if len(track.frames) < 3:
        raise ValueError("At least three frames are required")
    if np.any(track.localization_sd_um <= 0):
        raise ValueError("Brownian MLE requires positive localization SDs")
    return track


def _whiten(track: Track, acquisition: Acquisition) -> dict:
    """Whitened data and kernels for one validated track."""
    delta = np.diff(track.positions_um, axis=0).T  # (2, m)
    m = delta.shape[1]
    A, H = motion_kernels(m, float(acquisition.dt_s), float(acquisition.exposure_s))
    lam, y, Hw, logdet_B = [], [], [], 0.
    for axis, B in enumerate(localization_covariance(track.localization_sd_um)):
        L = cholesky(B, lower=True)
        LA = solve_triangular(L, A, lower=True)
        M = solve_triangular(L, LA.T, lower=True)  # L⁻¹ A L⁻ᵀ
        values, Q = eigh((M + M.T) / 2)
        LH = solve_triangular(L, H, lower=True)
        W = solve_triangular(L, LH.T, lower=True)
        lam.append(values)
        y.append(Q.T @ solve_triangular(L, delta[axis], lower=True))
        Hw.append(Q.T @ ((W + W.T) / 2) @ Q)
        logdet_B += 2 * np.sum(np.log(np.diag(L)))
    const = -.5 * logdet_B - m * np.log(2 * np.pi)
    return {"lam": np.array(lam), "y": np.array(y), "Hw": np.array(Hw), "const": const}


def _loglik(D, lam, y, const):
    """D (r,), y (r, 2, m) -> (r,)."""
    d = 1 + D[:, None, None] * lam
    return const - .5 * np.sum(np.log(d) + y**2 / d, axis=(1, 2))


def _maximize(lam, y, const):
    """Vectorized MLE over replicates y (r, 2, m) with D in [0, inf).

    Log grid search, golden-section refinement in the best bracket, then an
    explicit comparison with the boundary D = 0. Returns (D_hat, at_upper).
    """
    out, at_upper = np.empty(len(y)), np.zeros(len(y), bool)
    for start in range(0, len(y), _CHUNK):
        yc = y[start:start + _CHUNK]
        r = len(yc)
        # Unconstrained D lam_k is roughly y_k² - 1, so this bounds the maximizer.
        D_max = 10 * np.maximum(np.sum(yc**2, axis=(1, 2)), 1.) / lam.min()
        grid = D_max[:, None] * _GRID
        ll = _loglik(grid.ravel(), lam, np.repeat(yc, len(_GRID), axis=0), const).reshape(r, -1)
        best = np.argmax(ll, axis=1)
        at_upper[start:start + r] = best == len(_GRID) - 1
        rows = np.arange(r)
        lo = grid[rows, np.maximum(best - 1, 0)]
        hi = grid[rows, np.minimum(best + 1, len(_GRID) - 1)]
        a, b = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
        fa, fb = _loglik(a, lam, yc, const), _loglik(b, lam, yc, const)
        for _ in range(100):
            # Keep the half-bracket containing the larger value; reuse one point.
            left = fa > fb
            hi, lo = np.where(left, b, hi), np.where(left, lo, a)
            a, b = (np.where(left, hi - _GOLDEN * (hi - lo), b),
                    np.where(left, a, lo + _GOLDEN * (hi - lo)))
            f_new = _loglik(np.where(left, a, b), lam, yc, const)
            fa, fb = np.where(left, f_new, fb), np.where(left, fa, f_new)
            if np.all(hi - lo <= 1e-12 * hi):
                break
        D = np.where(fa > fb, a, b)
        zero = _loglik(np.zeros(r), lam, yc, const) >= _loglik(D, lam, yc, const)
        out[start:start + r] = np.where(zero, 0., D)
    return out, at_upper


def _score(D, lam, y, Hw):
    """Efficient alpha score and information at D (r,), y (r, 2, m).

    U_eff = U_alpha - (I_aD/I_DD) U_D removes the D direction, including the
    ln(dt) part of d/d(alpha), so it does not depend on the time unit. At an
    interior MLE U_D = 0 and U_eff equals the plain alpha score.
    """
    d = 1 + D[:, None, None] * lam
    inv = 1 / d
    v = y * inv
    diag = np.diagonal(Hw, axis1=1, axis2=2)  # (2, m)
    quad_form = np.einsum("rak,akl,ral->r", v, Hw, v)
    U_a = .5 * D * (quad_form - np.sum(diag * inv, axis=(1, 2)))
    U_D = .5 * np.sum(lam * (v**2 - inv), axis=(1, 2))
    I_aa = .5 * D**2 * np.einsum("rak,akl,ral->r", inv, Hw**2, inv)
    I_DD = .5 * np.sum(lam**2 * inv**2, axis=(1, 2))
    I_aD = .5 * D * np.sum(diag * lam * inv**2, axis=(1, 2))
    return U_a - I_aD / I_DD * U_D, I_aa - I_aD**2 / I_DD, I_DD


def brownian_log_likelihood(track: Track, acquisition: Acquisition, D_um2_s: float) -> float:
    """Exact Gaussian log-likelihood of both axes' displacements at D >= 0."""
    w = _whiten(_prepared(track, acquisition), acquisition)
    return float(_loglik(np.array([float(D_um2_s)]), w["lam"], w["y"][None], w["const"])[0])


def nonbrownian_score(track: Track, acquisition: Acquisition, D_um2_s: float) -> dict[str, float]:
    """Asymptotic efficient-score quantities for alpha at alpha = 1, given D > 0.

    U is the efficient score (the D direction projected out); I_eff its variance.

    alpha_1step = 1 + U/I_eff is one Newton step from the Brownian fit toward
    the fBm MLE; z_asymptotic = U/sqrt(I_eff) is uncalibrated at short lengths.
    """
    if not D_um2_s > 0:
        raise ValueError("The alpha score is identically zero at D = 0")
    w = _whiten(_prepared(track, acquisition), acquisition)
    U, I_eff, _ = _score(np.array([float(D_um2_s)]), w["lam"], w["y"][None], w["Hw"])
    U, I_eff = float(U[0]), float(I_eff[0])
    return {"U": U, "I_eff": I_eff, "z_asymptotic": float(U / np.sqrt(I_eff)),
            "alpha_1step": 1 + U / I_eff, "alpha_1step_se": float(1 / np.sqrt(I_eff))}


def _upper_limit(w, D_hat, ll_hat, level):
    """Profile-likelihood one-sided upper limit, using the boundary-aware threshold."""
    threshold = chi2.ppf(2 * level - 1, 1)
    f = lambda D: 2 * (ll_hat - _loglik(np.array([D]), w["lam"], w["y"][None], w["const"])[0]) - threshold  # noqa: E731
    hi = max(D_hat, 1 / w["lam"].max())
    while f(hi) < 0:
        hi *= 2
        if hi > 1e12 * (D_hat + 1 / w["lam"].min()):
            return None
    return float(brentq(f, D_hat, hi, xtol=1e-14, rtol=1e-12))


def _empty(status: str, message: str) -> BrownianMLE:
    return BrownianMLE(dict.fromkeys(BrownianMLE.PARAMETERS), status, message)


def fit_brownian_mle(track: Track, acquisition: Acquisition,
                     options: MLEOptions = MLEOptions()) -> BrownianMLE:
    """Maximum-likelihood D >= 0 and a calibrated non-Brownian score for one track.

    Status 'unresolved': D_hat = 0, so localization noise explains the
    motion; the score is then undefined. Status 'ok' with z/p null means the
    bootstrap was disabled (n_boot=0) or too few replicates resolved motion.
    """
    validate_mle_options(options)
    track = _prepared(track, acquisition)
    w = _whiten(track, acquisition)
    lam, const = w["lam"], w["const"]
    D, at_upper = _maximize(lam, w["y"][None], const)
    if at_upper[0] or not np.isfinite(D[0]):
        return _empty("failed", "Likelihood maximum not bracketed")
    D_hat = float(D[0])
    ll_hat = float(_loglik(D, lam, w["y"][None], const)[0])
    ll_zero = float(_loglik(np.zeros(1), lam, w["y"][None], const)[0])
    lr = max(2 * (ll_hat - ll_zero), 0.)
    parameters = dict.fromkeys(BrownianMLE.PARAMETERS)
    parameters.update(D_um2_s=D_hat, D_upper_um2_s=_upper_limit(w, D_hat, ll_hat, options.upper_level),
                      log_likelihood=ll_hat, lr_motion=lr,
                      p_motion=float(.5 * chi2.sf(lr, 1)) if lr > 0 else 1.)
    if D_hat == 0:
        return BrownianMLE(parameters, "unresolved",
                           "D_hat = 0: localization noise explains the displacements")
    U, I_eff, _ = _score(D, lam, w["y"][None], w["Hw"])
    U, I_eff = float(U[0]), float(I_eff[0])
    if not I_eff > 1e-12:
        return BrownianMLE(parameters, "unidentified", "No information about alpha at this D and length")
    z = U / np.sqrt(I_eff)
    parameters.update(z_nonbrownian_asymptotic=float(z), alpha_1step=1 + U / I_eff,
                      alpha_1step_se=float(1 / np.sqrt(I_eff)))
    if options.n_boot == 0:
        return BrownianMLE(parameters, "ok", "Bootstrap calibration disabled")
    rng = np.random.default_rng([options.seed, track.track_id % 2**63])
    y_star = np.sqrt(1 + D_hat * lam) * rng.standard_normal((options.n_boot, *lam.shape))
    D_star, upper_star = _maximize(lam, y_star, const)
    valid = (D_star > 0) & ~upper_star
    U_s, I_s, _ = _score(D_star[valid], lam, y_star[valid], w["Hw"])
    ok = I_s > 1e-12
    z_star = U_s[ok] / np.sqrt(I_s[ok])
    n_valid = int(z_star.size)
    if n_valid < min(100, options.n_boot):
        return BrownianMLE(parameters, "ok", f"Only {n_valid} bootstrap replicates resolved motion",
                           options.n_boot, n_valid)
    # Mid-rank p-value, conditional on replicates that also resolve motion.
    p = (np.sum(z_star < z) + .5 * np.sum(z_star == z) + .5) / (n_valid + 1)
    parameters.update(z_nonbrownian=float(norm.ppf(p)), p_nonbrownian=float(2 * min(p, 1 - p)))
    return BrownianMLE(parameters, "ok", "", options.n_boot, n_valid)
