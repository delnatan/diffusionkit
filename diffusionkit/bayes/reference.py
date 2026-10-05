"""Blurred fractional Brownian motion covariance in numpy: the reference `bayes.likelihood`'s jax version is
checked against (tests/test_bayes_blur.py).

Per axis the motion has structure function E[(z(t+s) - z(t))^2] = 2 K |s|^alpha, and each frame records its average
over a box exposure (0 <= exposure_s <= dt_s). At alpha = 1 this is `gridpost.likelihood.motion_covariance`, the
Berglund box-shutter average, at any exposure.
"""
import numpy as np

# Below this exposure/lag ratio `_blurred_abs_power` uses its series (through
# (te/x)^12, truncation ~1e-16 here) rather than the exact second difference,
# which cancels: its relative error is ~ eps (x/te)^2, ~2e-14 at the switch.
_BLUR_SERIES_RATIO = .1
_BLUR_SERIES_TERMS = 6


def _blurred_abs_power(x: np.ndarray, alpha, exposure_s: float) -> np.ndarray:
    """E|x + u - v|^alpha for independent u, v ~ Uniform(0, exposure_s); x and alpha broadcast.

    w = u - v has the triangular density on [-te, te], so the average is
    (H(x+te) - 2 H(x) + H(x-te)) / te^2 with H'' = |x|^alpha, i.e.
    H(x) = |x|^(alpha+2) / ((alpha+1)(alpha+2)). Where te/|x| is small that
    difference cancels, so there the binomial series of |x|^alpha (1 + w/x)^alpha
    is used instead, with the even moments E[w^2k] = 2 te^2k / ((2k+1)(2k+2)).
    """
    x = np.abs(np.asarray(x, dtype=float))
    alpha = np.asarray(alpha, dtype=float)
    if exposure_s == 0:
        return x**alpha
    te = exposure_s
    series = x * _BLUR_SERIES_RATIO > te
    x_series = np.where(series, x, 1.)
    r2 = np.where(series, (te / x_series) ** 2, 0.)
    total, binom, power = 1., 1., 1.
    for k in range(1, _BLUR_SERIES_TERMS + 1):
        n = 2 * k
        binom = binom * (alpha - n + 2) * (alpha - n + 1) / ((n - 1) * n)  # C(alpha, n)
        power = power * r2
        total = total + binom * (2 / ((n + 1) * (n + 2))) * power
    exact = (np.abs(x + te) ** (alpha + 2) - 2 * x ** (alpha + 2) + np.abs(x - te) ** (alpha + 2)) / (
        (alpha + 1) * (alpha + 2) * te**2)
    return np.where(series, x_series**alpha * total, exact)


def fgn_autocovariance(n_disp: int, dt_s: float, alpha, exposure_s: float = 0.) -> np.ndarray:
    """gamma(k), k = 0 .. n_disp-1: `fgn_motion_covariance`'s first row, per unit K.

    `alpha` may be an array: the result has shape `np.shape(alpha) + (n_disp,)`.
    """
    alpha = np.asarray(alpha, dtype=float)[..., None]
    G = _blurred_abs_power(np.arange(-1, n_disp + 1) * dt_s, alpha, exposure_s)  # lags -1 .. n_disp
    return G[..., 2:] - 2 * G[..., 1:-1] + G[..., :-2]


def fgn_motion_covariance(n_disp: int, dt_s: float, alpha, exposure_s: float = 0.) -> np.ndarray:
    """(n_disp, n_disp) fBm displacement covariance per unit K, box-shutter blurred.

    The true motion has structure function E[(z(t+s) - z(t))^2] = 2 K |s|^alpha
    per axis, and each frame records its average over the exposure
    (0 <= exposure_s <= dt_s). With G(x) = E|x + u - v|^alpha over the two
    frames' exposure offsets (`_blurred_abs_power`),

        gamma(k) = G((k+1) dt) - 2 G(k dt) + G((k-1) dt)   (lag k).

    exposure_s=0 gives plain fGn, dt^alpha (|k+1|^alpha - 2|k|^alpha +
    |k-1|^alpha). alpha=1 gives `gridpost.likelihood.motion_covariance` (the Berglund box-shutter
    average) at any exposure, so this is its generalization, not a second
    blur model. Blur correlates adjacent displacements positively, which an
    unblurred model reads as a higher alpha. An array `alpha` gives a stack of
    matrices, shape `np.shape(alpha) + (n_disp, n_disp)`.
    """
    gamma = fgn_autocovariance(n_disp, dt_s, alpha, exposure_s)
    i = np.arange(n_disp)
    return gamma[..., np.abs(i[:, None] - i[None, :])]
