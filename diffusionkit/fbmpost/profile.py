"""Per-track alpha by profile likelihood: the amplitude K maximized at each alpha, not integrated.

With known noise the likelihood stays finite as K -> 0 (the track is "all noise"), so integrating K
under any fixed prior lets the prior's range below detection, and the width of the likelihood in K,
weigh each alpha; summed over tracks that tilt adds up (alpha note §8.3). The profile

    l_p(alpha) = max over K >= 0 of l(alpha, K)

has no prior in it. Its likelihood-ratio interval {alpha : l_p >= max l_p - chi2_1(level) / 2}
covered 95-98 % at nominal 95 % in every tested cell (alpha note §8.2), and summing profiles over
tracks pools alpha without bias (`population.shared_alpha`).

K is profiled per (track, alpha) on the whitened data (`likelihood.whiten_batch`): a scan of ln K in
e-fold steps, in units of the track's own noise, finds the peak, and safeguarded Newton refines it at
O(m) per step. D_ref's interval is the projection of the same likelihood region,
{(alpha, D_ref) : l >= max l - chi2_1(level) / 2}, onto D_ref: that is the profile-likelihood
interval of D_ref with alpha profiled out.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.stats import chi2

from .likelihood import Whitened, _CHUNK_FLOATS, loglik_points

# The scan of ln SNR = ln(K ||lam||), one point per e-fold. A search range, not a prior: Newton may leave it.
_SCAN = np.arange(np.log(1e-4), np.log(1e8) + .5, 1.)
_NEWTON_STEPS = 40
_MAX_STEP = 1.  # in ln K, the scan's spacing
_ROOT_STEPS = 40  # bisections of a bracket at most 64 wide in ln K: 6e-11
# A track whose best likelihood is within this of its K = 0 value has no motion above the noise.
_IDENTIFIED_NATS = 1e-6


class Profile(NamedTuple):
    """The profile over K of T tracks at n_a alphas."""
    loglik: np.ndarray  # (T, n_a) l_p(alpha)
    ln_K: np.ndarray    # (T, n_a) its argmax, -inf where K = 0 is best
    zero: np.ndarray    # (T,) l(K = 0), the noise-only value, the same at every alpha
    scan_u: np.ndarray  # (T, n_a, n_scan) ln K of the scan
    scan_ll: np.ndarray  # (T, n_a, n_scan) log-likelihood there


def _derivatives(lam, y2, const, u):
    """Value, gradient and Hessian in ln K, and the Fisher information, for rows (R, 2m) at u (R,)."""
    kl = np.exp(u)[:, None] * lam
    v = 1 + kl
    w = kl / v
    r = y2 / v
    f = const - .5 * (np.log(v) + r).sum(-1)
    g = -.5 * (w * (1 - r)).sum(-1)
    h = -.5 * (w * (1 - w) - r * w * (1 - 2 * w)).sum(-1)
    return f, g, h, .5 * (w * w).sum(-1)


def _value(lam, y2, const, u):
    v = 1 + np.exp(u)[:, None] * lam
    return const - .5 * (np.log(v) + y2 / v).sum(-1)


def _newton(lam, y2, const, u, f):
    """Maximize over u = ln K from (u, f) for rows (R, 2m): Newton where the Hessian is negative, Fisher scoring
    elsewhere, steps capped at `_MAX_STEP` and halved until the value does not fall."""
    for _ in range(_NEWTON_STEPS):
        f, g, h, info = _derivatives(lam, y2, const, u)
        step = np.where(h < 0, -g / np.minimum(h, -1e-300), g / np.maximum(info, 1e-300))
        step = np.clip(step, -_MAX_STEP, _MAX_STEP)
        if not (np.abs(step) > 1e-10).any():
            break
        t = np.ones_like(u)
        for _ in range(40):
            new = _value(lam, y2, const, u + t * step)
            worse = new < f - 1e-12 * (1 + np.abs(f))
            if not worse.any():
                break
            t = np.where(worse, t / 2, t)
        keep = ~worse
        u, f = np.where(keep, u + t * step, u), np.where(keep, new, f)
    return u, f


def profile_K(w: Whitened) -> Profile:
    """K profiled out at each of the whitened alphas, for every track in `w`."""
    T, n_a, m2 = w.lam.shape
    scale = .5 * np.log((w.lam ** 2).sum(-1))  # ln ||lam||: the scan is in units of the track's noise
    scan_u = _SCAN[None, None, :] - scale[..., None]
    scan_ll = loglik_points(w, scan_u)
    j = scan_ll.argmax(-1)[..., None]
    u0 = np.take_along_axis(scan_u, j, -1)[..., 0].ravel()
    f0 = np.take_along_axis(scan_ll, j, -1)[..., 0].ravel()
    lam, y2 = w.lam.reshape(T * n_a, m2), w.y2.reshape(T * n_a, m2)
    const = np.repeat(w.const, n_a)
    u, f = np.empty_like(u0), np.empty_like(f0)
    rows = max(1, _CHUNK_FLOATS // (8 * m2))  # `_derivatives` holds a few (rows, 2m) arrays at once
    for r in range(0, len(u0), rows):
        s = slice(r, r + rows)
        u[s], f[s] = _newton(lam[s], y2[s], const[s], u0[s], f0[s])
    zero = w.const - .5 * w.y2[:, 0].sum(-1)  # sum y^2 = ||L^-1 delta||^2 at every alpha
    u, f = u.reshape(T, n_a), f.reshape(T, n_a)
    at_zero = f <= zero[:, None] + 1e-12 * (1 + np.abs(zero[:, None]))
    return Profile(np.where(at_zero, zero[:, None], f), np.where(at_zero, -np.inf, u), zero, scan_u, scan_ll)


def _bisect(lam, y2, const, inside, outside, thr):
    """u between `inside` (l >= thr) and `outside` (l < thr) where l(u) = thr, for rows (R, 2m)."""
    for _ in range(_ROOT_STEPS):
        mid = (inside + outside) / 2
        up = _value(lam, y2, const, mid) >= thr
        inside, outside = np.where(up, mid, inside), np.where(up, outside, mid)
    return (inside + outside) / 2


def _bracket(lam, y2, const, start, thr, sign):
    """A point at `start + sign * d` (d = 1, 2, 4, ...) with l < thr, for rows whose l(start) >= thr."""
    d = np.ones_like(start)
    for _ in range(12):
        out = start + sign * d
        below = _value(lam, y2, const, out) < thr
        if below.all():
            return out
        d = np.where(below, d, 2 * d)
    return start + sign * d


def ln_K_interval(w: Whitened, p: Profile, thr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per (track, alpha) with l_p >= thr (T,): the ln K range where l >= thr, (T, n_a) each, NaN elsewhere.

    The lower end is -inf where the noise-only value l(K = 0) is itself above thr.
    """
    T, n_a, m2 = w.lam.shape
    lo, hi = np.full((T, n_a), np.nan), np.full((T, n_a), np.nan)
    t, a = np.nonzero(p.loglik >= thr[:, None])
    if not len(t):
        return lo, hi
    lam, y2, const, level = w.lam[t, a], w.y2[t, a], w.const[t], thr[t]
    # Where K = 0 is best, start just above the scan, where l is the noise-only value to ~1e-8.
    start = np.where(np.isfinite(p.ln_K[t, a]), p.ln_K[t, a], p.scan_u[t, a, 0] - 10)
    up = _bracket(lam, y2, const, start, level, 1)
    hi[t, a] = _bisect(lam, y2, const, start, up, level)
    has_lo = p.zero[t] < level
    if has_lo.any():
        k = np.flatnonzero(has_lo)
        down = _bracket(lam[k], y2[k], const[k], start[k], level[k], -1)
        lo[t[k], a[k]] = _bisect(lam[k], y2[k], const[k], start[k], down, level[k])
    lo[t[~has_lo], a[~has_lo]] = -np.inf
    return lo, hi


class TrackProfiles(NamedTuple):
    """Per-track summaries of T profiles on the alpha grid. `alpha_hat` is NaN where alpha is not identified."""
    curve: np.ndarray      # (T, n_grid) l_p(alpha) - max l_p
    loglik_max: np.ndarray  # (T,) the maximum, refined between grid points
    alpha_hat: np.ndarray
    alpha_lo: np.ndarray
    alpha_hi: np.ndarray
    alpha_edge: list       # "low", "high", "both" or None: which grid end the interval reaches
    dll_brownian: np.ndarray  # max l_p - l_p(1) >= 0
    p_brownian: np.ndarray    # chi2_1 tail of 2 dll_brownian
    ln_K_hat: np.ndarray      # at alpha_hat; -inf when K = 0
    log10_D_ref_hat: np.ndarray
    log10_D_ref_lo: np.ndarray  # -inf: the track only bounds D_ref from above
    log10_D_ref_hi: np.ndarray


def _crossing(a0, a1, l0, l1, thr):
    """The alpha between grid points a0 (l0 < thr) and a1 (l1 >= thr) where the linear interpolant reaches thr."""
    with np.errstate(divide="ignore", invalid="ignore"):  # unused where l1 == l0 (the interval reaches the edge)
        return a0 + (a1 - a0) * (thr - l0) / (l1 - l0)


def summarize(w: Whitened, alphas: np.ndarray, n_grid: int, i_one: int, level: float,
              tau_ref_s: float) -> TrackProfiles:
    """Profile summaries for the tracks of `w`, whitened on `alphas`: the alpha grid is `alphas[:n_grid]`, and
    `alphas[i_one]` is 1 (on the grid or appended after it)."""
    p = profile_K(w)
    ll, grid = p.loglik[:, :n_grid], alphas[:n_grid]
    T = len(ll)
    c = chi2.ppf(level, 1) / 2
    rows = np.arange(T)
    i = ll.argmax(1)
    top = ll[rows, i]
    # A parabola through the peak and its neighbours refines alpha_hat and the maximum between grid points.
    inner = (i > 0) & (i < n_grid - 1)
    j = np.clip(i, 1, n_grid - 2)
    lm, l0, lp = ll[rows, j - 1], ll[rows, j], ll[rows, j + 1]
    curv = lm - 2 * l0 + lp
    ok = inner & (curv < 0)
    delta = np.where(ok, .5 * (lm - lp) / np.where(ok, curv, -1), 0.)
    h = grid[1] - grid[0]
    alpha_hat = grid[i] + delta * h
    ll_max = np.where(ok, top - .25 * (lm - lp) * delta, top)
    identified = ll_max - p.zero > _IDENTIFIED_NATS
    thr = ll_max - c

    inside = ll >= thr[:, None]
    first = inside.argmax(1)
    last = n_grid - 1 - inside[:, ::-1].argmax(1)
    lo_prev = np.maximum(first - 1, 0)
    hi_next = np.minimum(last + 1, n_grid - 1)
    alpha_lo = np.where(first == 0, grid[0], _crossing(grid[lo_prev], grid[first], ll[rows, lo_prev],
                                                        ll[rows, first], thr))
    alpha_hi = np.where(last == n_grid - 1, grid[-1], _crossing(grid[hi_next], grid[last], ll[rows, hi_next],
                                                                 ll[rows, last], thr))
    edge = [{(True, True): "both", (True, False): "low", (False, True): "high"}.get((bool(a), bool(b)))
            for a, b in zip(first == 0, last == n_grid - 1)]

    dll = np.maximum(ll_max - p.loglik[:, i_one], 0.)
    shift = (alphas[None, :] - 1) * np.log(tau_ref_s)  # ln D_ref = ln K + (alpha - 1) ln tau_ref
    x = (p.ln_K + shift) / np.log(10)
    # log10 D_ref at alpha_hat: linear between the grid rows around it (the peak row where K = 0 at a neighbour).
    nb = np.clip(np.where(delta >= 0, i + 1, i - 1), 0, n_grid - 1)
    xi, xn = x[rows, i], x[rows, nb]
    frac = np.abs(delta)
    with np.errstate(invalid="ignore"):
        x_hat = np.where(np.isfinite(xn) & (nb != i), xi + frac * (xn - xi), xi)
    k_lo, k_hi = ln_K_interval(Whitened(w.lam[:, :n_grid], w.y2[:, :n_grid], w.const), p._replace(
        loglik=ll, ln_K=p.ln_K[:, :n_grid], scan_u=p.scan_u[:, :n_grid]), thr)
    with np.errstate(invalid="ignore"):
        x_lo = np.nanmin((k_lo + shift[:, :n_grid]) / np.log(10), axis=1)
        x_hi = np.nanmax((k_hi + shift[:, :n_grid]) / np.log(10), axis=1)
    ln_K_hat = x_hat * np.log(10) - (alpha_hat - 1) * np.log(tau_ref_s)
    nan = np.full(T, np.nan)
    return TrackProfiles(
        curve=ll - ll_max[:, None], loglik_max=ll_max,
        alpha_hat=np.where(identified, alpha_hat, nan), alpha_lo=alpha_lo, alpha_hi=alpha_hi, alpha_edge=edge,
        dll_brownian=dll, p_brownian=chi2.sf(2 * dll, 1),
        ln_K_hat=np.where(identified, ln_K_hat, -np.inf),
        log10_D_ref_hat=np.where(identified, x_hat, -np.inf), log10_D_ref_lo=x_lo, log10_D_ref_hi=x_hi)


def alphas_with_one(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """The alphas to whiten on: the grid, then 1 if the grid misses it; and the index of 1."""
    near = np.flatnonzero(np.isclose(grid, 1., rtol=0, atol=1e-12))
    if len(near):
        return grid, int(near[0])
    return np.append(grid, 1.), len(grid)
