"""Checks for posterior_alpha. Run: uv run --with pytest pytest prototypes"""
import numpy as np
from scipy.stats import multivariate_normal

import posterior_1d as PD
import posterior_alpha as PA

DT = 0.033


def dense_fgn_ll(disp, sd, dt, K, alpha):
    """Independent oracle: dense fGn + localization covariance, SciPy density."""
    m = len(disp)
    k = np.abs(np.subtract.outer(np.arange(m), np.arange(m)))
    G = K * dt**alpha * (np.abs(k + 1) ** alpha + np.abs(k - 1) ** alpha - 2 * k**alpha)
    B = PA.localization_cov(sd)
    return multivariate_normal(np.zeros(m), G + B).logpdf(disp)


def test_whitened_loglik_matches_direct_gaussian():
    rng = np.random.default_rng(1)
    sd = rng.uniform(0.02, 0.05, 7)
    disp = rng.normal(0, 0.1, 6)
    for alpha in (0.4, 1.0, 1.6):
        w = PA.whiten(disp, sd, DT, alpha)
        for K in (0.001, 0.05, 0.3):
            direct = dense_fgn_ll(disp, sd, DT, K, alpha)
            got = PA.loglik([w], np.log([K]))[0]
            assert np.isclose(got, direct, rtol=1e-8)


def test_motion_cov_alpha_one_matches_brownian():
    """alpha=1, no blur reduces exactly to posterior_1d's Brownian motion_cov."""
    m = 6
    np.testing.assert_allclose(PA.motion_cov(m, DT, 1.0), PD.motion_cov(m, DT, on_time=0.0))


def test_motion_cov_matches_fine_step_simulation_subdiffusive():
    """Covariance of simulated fGn displacements (no localization noise) is K * A(alpha)."""
    rng = np.random.default_rng(2)
    K, alpha, n, sims = 0.5, 0.6, 5, 20000
    disp = np.array([
        np.diff(PA.simulate(K, alpha, np.full(n, 1e-12), DT, rng, axes=1)[:, 0]) for _ in range(sims)
    ])
    emp, model = np.cov(disp.T), K * PA.motion_cov(n - 1, DT, alpha)
    assert np.abs(emp - model).max() < 0.06 * model.max()


def test_credible_intervals_are_calibrated():
    """Truth drawn from flat priors, data from the independent simulator: 90% intervals cover 90%."""
    K_true = 0.05
    K_prior = PA.log_uniform_K(1e-3, 1.0)
    rng = np.random.default_rng(3)
    for n_frames in (8, 15):
        hits, n_tracks = 0, 250
        for _ in range(n_tracks):
            alpha_true = rng.uniform(PA.ALPHA[0], PA.ALPHA[-1])
            sd = rng.uniform(0.025, 0.045, n_frames)
            x = PA.simulate(K_true, alpha_true, sd, DT, rng)
            p = PA.track_alpha_posterior(x, sd, DT, K_prior)
            s = PA.summary(p, level=0.9)
            hits += s["lo"] <= alpha_true <= s["hi"]
        assert abs(hits / n_tracks - 0.9) < 0.08


def test_short_track_posterior_is_wide():
    """A 5-frame track's alpha posterior stays wide (honest), not falsely confident."""
    rng = np.random.default_rng(4)
    n = 5
    sd = rng.uniform(0.03, 0.045, n)
    x = PA.simulate(0.05, 1.0, sd, DT, rng)
    p = PA.track_alpha_posterior(x, sd, DT, PA.log_uniform_K(1e-3, 1.0))
    s = PA.summary(p)
    assert (s["hi"] - s["lo"]) > 0.6 * (PA.ALPHA[-1] - PA.ALPHA[0])
