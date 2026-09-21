"""Checks for posterior_1d. Run: uv run --with pytest pytest prototypes"""
import numpy as np
from scipy.stats import multivariate_normal

import posterior_1d as P

DT, ON_TIME = 0.035, 0.020


def test_whitened_loglik_matches_direct_gaussian():
    rng = np.random.default_rng(1)
    sd = rng.uniform(.02, .05, 7)
    disp = rng.normal(0, .1, 6)
    w = P.whiten(disp, sd, DT, ON_TIME)
    for D in (0., .01, .3, 2.):
        Sigma = D * P.motion_cov(6, DT, ON_TIME) + P.localization_cov(sd)
        direct = multivariate_normal(np.zeros(6), Sigma).logpdf(disp)
        assert np.isclose(P.loglik([w], np.log([max(D, 1e-300)]))[0], direct, rtol=1e-9)


def test_motion_kernel_matches_fine_step_simulation():
    """Covariance of simulated displacements (no localization noise) is D * A."""
    rng = np.random.default_rng(2)
    D, n, sims = .5, 4, 20000
    disp = np.array([np.diff(P.simulate(D, np.full(n, 1e-12), DT, ON_TIME, rng, axes=1)[:, 0]) for _ in range(sims)])
    emp, model = np.cov(disp.T), D * P.motion_cov(n - 1, DT, ON_TIME)
    assert np.abs(emp - model).max() < .04 * model.max()
    assert model[0, 1] > 0  # blur, not localization noise, correlates neighbours positively


def test_sequence_is_causal():
    """Row k depends only on the first k + 1 frames."""
    rng = np.random.default_rng(3)
    sd = np.full(9, .03)
    x = P.simulate(.1, sd, DT, ON_TIME, rng)
    prior = P.log_uniform(1e-3, 1.)
    full = P.sequence(x, sd, DT, prior, ON_TIME)
    x2 = x.copy()
    x2[5:] += 1.
    changed = P.sequence(x2, sd, DT, prior, ON_TIME)
    assert np.allclose(full[:4], changed[:4])
    assert not np.allclose(full[4:], changed[4:])


def _coverage(draw_u, log_prior, n_frames, n_tracks=400, level=.9, seed=4):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_tracks):
        u_true = draw_u(rng)
        sd = rng.uniform(.025, .045, n_frames)
        x = P.simulate(np.exp(u_true), sd, DT, ON_TIME, rng)
        s = P.summary(P.posterior(P.track_loglik(x, sd, DT, ON_TIME), log_prior), level=level)
        hits += s["lo"] <= np.exp(u_true) <= s["hi"]
    return hits / n_tracks


def test_credible_intervals_are_calibrated():
    """Truth drawn from the prior, data from the independent simulator: 90% intervals cover 90%."""
    lo, hi = 1e-3, 1.
    flat = P.log_uniform(lo, hi)
    soft = P.log_normal(lo, hi)
    mu, sd = .5 * np.log(lo * hi), np.log(hi / lo) / (2 * 1.96)
    for n_frames in (5, 12):
        assert abs(_coverage(lambda r: r.uniform(np.log(lo), np.log(hi)), flat, n_frames) - .9) < .05
        assert abs(_coverage(lambda r: r.normal(mu, sd), soft, n_frames) - .9) < .05


def _tracks_ll(D_values, rng, frames=(5, 13)):
    """Per-track log-likelihoods on P.U for one simulated track per entry of D_values."""
    out = []
    for D in D_values:
        sd = rng.uniform(.03, .045, rng.integers(*frames))
        out.append(P.track_loglik(P.simulate(D, sd, DT, ON_TIME, rng), sd, DT, ON_TIME))
    return np.array(out)


def test_deconvolve_first_step_is_sum_of_posteriors():
    rng = np.random.default_rng(5)
    lls = _tracks_ll(np.exp(rng.uniform(np.log(.01), np.log(.5), 40)), rng)
    prior = P.log_uniform(1e-3, 1.)
    one_step = P.deconvolve(lls, prior, iters=1, smooth=0)
    assert np.allclose(one_step, np.mean([P.posterior(ll, prior) for ll in lls], axis=0))


def test_em_never_decreases_the_likelihood():
    rng = np.random.default_rng(6)
    lls = _tracks_ll(np.exp(rng.choice(np.log([.02, .2]), 150)), rng)
    prior = P.log_uniform(1e-3, 1.)
    L = np.exp(lls - lls.max(axis=1, keepdims=True))
    objective = [np.sum(np.log(L @ P.deconvolve(lls, prior, iters=k, smooth=0))) for k in (1, 2, 5, 20, 80)]
    assert np.all(np.diff(objective) > 0)


def test_pooled_posterior_is_calibrated_when_tracks_share_D():
    """Truth drawn from the prior, one D per dataset of 10 tracks: 90% intervals cover 90%."""
    rng = np.random.default_rng(7)
    prior = P.log_uniform(1e-3, 1.)
    hits, n_sets = 0, 300
    for _ in range(n_sets):
        D = np.exp(rng.uniform(np.log(1e-3), np.log(1.)))
        s = P.summary(P.pooled(_tracks_ll(np.full(10, D), rng), prior))
        hits += s["lo"] <= D <= s["hi"]
    assert abs(hits / n_sets - .9) < .06


def test_deconvolution_recovers_the_mass_of_each_mode():
    rng = np.random.default_rng(8)
    D = np.exp(np.log(rng.choice([.02, .2], 400)) + .15 * rng.standard_normal(400))
    g = P.deconvolve(_tracks_ll(D, rng, frames=(8, 21)), P.log_uniform(1e-3, 1.))
    assert abs(g[P.U > np.log(np.sqrt(.02 * .2))].sum() - np.mean(D > np.sqrt(.02 * .2))) < .05
