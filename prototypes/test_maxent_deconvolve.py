"""Checks for maxent_deconvolve. Run: uv run --with pytest pytest prototypes"""
import numpy as np
from scipy.optimize import check_grad

import maxent_deconvolve as M
import posterior_1d as P

DT, ON_TIME = 0.035, 0.020


def _tracks_ll(D_values, rng, frames=(5, 13), u=P.U):
    """Per-track log-likelihoods on `u` for one simulated track per entry of D_values."""
    out = []
    for D in D_values:
        sd = rng.uniform(.03, .045, rng.integers(*frames))
        out.append(P.track_loglik(P.simulate(D, sd, DT, ON_TIME, rng), sd, DT, ON_TIME, u))
    return np.array(out)


def _restricted(lls, log_prior):
    """(L, m, support) exactly as maxent_deconvolve builds them internally."""
    support = np.isfinite(log_prior)
    L = np.exp(lls[:, support] - lls[:, support].max(axis=1, keepdims=True))
    return L, M.default_measure(int(support.sum())), support


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(10)
    prior = P.log_uniform(1e-3, 1.)
    lls = _tracks_ll(np.exp(rng.uniform(np.log(.01), np.log(.5), 20)), rng)
    L, m, _ = _restricted(lls, prior)

    def f(theta, L, m, alpha):
        return M._neg_objective_and_grad(theta, L, m, alpha)[0]

    def grad(theta, L, m, alpha):
        return M._neg_objective_and_grad(theta, L, m, alpha)[1]

    for alpha in (1e-2, 1., 100.):
        theta0 = np.log(m) + 0.05 * rng.standard_normal(len(m))
        err = check_grad(f, grad, theta0, L, m, alpha)
        assert err < 1e-4


def test_fit_maxent_is_a_local_maximum():
    rng = np.random.default_rng(11)
    prior = P.log_uniform(1e-3, 1.)
    lls = _tracks_ll(np.exp(rng.uniform(np.log(.01), np.log(.5), 30)), rng)
    L, m, _ = _restricted(lls, prior)
    alpha = 1.0
    w, _ = M.fit_maxent(L, m, alpha)
    F0 = M.mixture_loglik(L, w) + alpha * M.entropy(w, m)

    K = len(w)
    for _ in range(20):
        v = rng.standard_normal(K)
        v -= v.mean()  # project onto the sum-zero tangent direction
        for eps in (1e-3, 3e-3):
            w_pert = w + eps * v
            if np.any(w_pert <= 0):
                continue
            w_pert = w_pert / w_pert.sum()
            F1 = M.mixture_loglik(L, w_pert) + alpha * M.entropy(w_pert, m)
            assert F1 < F0


def test_alpha_to_zero_matches_unregularized_deconvolve():
    rng = np.random.default_rng(12)
    prior = P.log_uniform(1e-3, 1.)
    D = np.exp(np.log(rng.choice([.02, .2], 150)) + .15 * rng.standard_normal(150))
    lls = _tracks_ll(D, rng, frames=(8, 21))
    L, m, _ = _restricted(lls, prior)

    lambda_max0 = M._generalized_eigs(L, m)[-1]
    alpha = lambda_max0 * 1e-6
    w, _ = M.fit_maxent(L, m, alpha, maxiter=2000)
    logL_maxent = M.mixture_loglik(L, w)

    g = P.deconvolve(lls, prior, smooth=0, iters=3000)
    support = np.isfinite(prior)
    logL_em = M.mixture_loglik(L, g[support])

    assert logL_maxent > logL_em - 1e-2 * abs(logL_em)


def test_alpha_to_infinity_collapses_to_the_measure():
    rng = np.random.default_rng(13)
    prior = P.log_uniform(1e-3, 1.)
    lls = _tracks_ll(np.exp(rng.uniform(np.log(.01), np.log(.5), 20)), rng)
    L, m, _ = _restricted(lls, prior)

    lambda_max0 = M._generalized_eigs(L, m)[-1]
    w, _ = M.fit_maxent(L, m, lambda_max0 * 1e6)
    assert np.abs(w - m).max() < 1e-3


def _two_mode_dataset(rng, n=250):
    D = np.exp(np.log(rng.choice([.02, .2], n)) + .15 * rng.standard_normal(n))
    prior = P.log_uniform(1e-3, 1.)
    lls = _tracks_ll(D, rng, frames=(8, 21))
    return D, lls, prior


def test_evidence_peaks_in_the_interior_not_at_a_boundary():
    rng = np.random.default_rng(14)
    _, lls, prior = _two_mode_dataset(rng)
    fit = M.maxent_deconvolve(lls, prior)
    best = int(np.argmax(fit.evidence_curve))
    n = len(fit.alpha_grid)
    assert 1 <= best <= n - 2


def test_evidence_is_a_local_max_on_the_grid():
    rng = np.random.default_rng(15)
    _, lls, prior = _two_mode_dataset(rng)
    fit = M.maxent_deconvolve(lls, prior)
    best = int(np.argmax(fit.evidence_curve))
    n = len(fit.alpha_grid)
    if best > 0:
        assert fit.evidence_curve[best - 1] < fit.evidence_curve[best]
    if best < n - 1:
        assert fit.evidence_curve[best + 1] < fit.evidence_curve[best]


def test_eigenvalues_are_nonnegative_and_rank_bounded():
    rng = np.random.default_rng(16)
    n_tracks = 60
    _, lls, prior = _two_mode_dataset(rng, n=n_tracks)
    L, m, _ = _restricted(lls, prior)
    w, _ = M.fit_maxent(L, m, 1.0)
    eigvals = M._generalized_eigs(L, w)
    assert np.all(eigvals >= -1e-6)
    assert np.sum(eigvals > 1e-8 * eigvals.max()) <= n_tracks


def test_maxent_recovers_the_mass_of_each_mode():
    rng = np.random.default_rng(17)
    D, lls, prior = _two_mode_dataset(rng, n=300)
    fit = M.maxent_deconvolve(lls, prior)
    mid = np.log(np.sqrt(.02 * .2))
    recovered = fit.w[P.U > mid].sum()
    truth = np.mean(D > np.sqrt(.02 * .2))
    assert abs(recovered - truth) < 0.08


def test_maxent_peak_width_is_less_grid_resolution_sensitive_than_smoothed_em():
    rng = np.random.default_rng(18)
    D = np.exp(np.log(.2) + .15 * rng.standard_normal(300))  # single mode, so "the peak" is unambiguous
    prior = P.log_uniform(1e-3, 1.)

    def fwhm(u_grid):
        lls = _tracks_ll(D, np.random.default_rng(18), frames=(8, 21), u=u_grid)
        p = P.deconvolve(lls, P.log_uniform(1e-3, 1., u_grid), smooth=0.5)
        du = u_grid[1] - u_grid[0]
        dens = p / du
        half = dens.max() / 2
        above = u_grid[dens >= half]
        em_width = above[-1] - above[0]

        fit = M.maxent_deconvolve(lls, P.log_uniform(1e-3, 1., u_grid), u=u_grid)
        dens_m = fit.w / du
        half_m = dens_m.max() / 2
        above_m = u_grid[dens_m >= half_m]
        maxent_width = above_m[-1] - above_m[0]
        return em_width, maxent_width

    u_coarse = np.linspace(np.log(1e-4), np.log(10.), 251)
    u_fine = np.linspace(np.log(1e-4), np.log(10.), 501)
    em_coarse, maxent_coarse = fwhm(u_coarse)
    em_fine, maxent_fine = fwhm(u_fine)

    em_change = abs(em_fine - em_coarse) / em_coarse
    maxent_change = abs(maxent_fine - maxent_coarse) / maxent_coarse
    assert maxent_change < em_change


def test_n_good_is_between_zero_and_rank_and_larger_for_the_heterogeneous_dataset():
    rng = np.random.default_rng(19)
    homog_D = np.full(200, .05)
    lls_homog = _tracks_ll(homog_D, rng, frames=(8, 21))
    prior = P.log_uniform(1e-3, 1.)
    fit_homog = M.maxent_deconvolve(lls_homog, prior)

    rng2 = np.random.default_rng(20)
    _, lls_mixed, _ = _two_mode_dataset(rng2, n=200)
    fit_mixed = M.maxent_deconvolve(lls_mixed, prior)

    assert 0 <= fit_homog.n_good <= 200
    assert 0 <= fit_mixed.n_good <= 200
    assert fit_mixed.n_good > fit_homog.n_good
