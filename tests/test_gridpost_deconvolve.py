"""Distribution of D across tracks (diffusionkit.gridpost.deconvolve) against simulation."""
import unittest
import warnings

import numpy as np
import polars as pl
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.stats import multivariate_t

from diffusionkit import Acquisition
from diffusionkit.gridpost import deconvolve as D
from diffusionkit.gridpost import posterior as P
from diffusionkit.gridpost.data import GridPostOptions
from diffusionkit.gridpost.likelihood import fgn_motion_covariance
from diffusionkit.gridpost.workflow import analyze_tracks

DT = .033
U = GridPostOptions().u_D()


def track_table(track_id, frames, positions, sd):
    return pl.DataFrame({
        "track_id": [track_id] * len(frames), "frame": frames,
        "x_um": positions[:, 0], "y_um": positions[:, 1],
        "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1],
    })


def simulate(D_true, sd, dt, rng, exposure=0., n_frames=None):
    n = len(sd) if n_frames is None else n_frames
    sub = 100
    h = dt / sub
    n_on = round(exposure / h)
    path = np.concatenate([np.zeros((1, 2)),
                           np.cumsum(rng.normal(0, np.sqrt(2 * D_true * h), ((n - 1) * sub + n_on, 2)), axis=0)])
    w = np.ones(n_on + 1)
    w[[0, -1]] = .5
    pos = np.array([w @ path[i * sub:i * sub + n_on + 1] / max(n_on, 1) if n_on else path[i * sub]
                    for i in range(n)])
    return pos + sd[:, None] * rng.standard_normal((n, 2))


def track_lls(D_values, rng, u=U, frames=(5, 21)):
    """Per-track log-likelihood rows on u, one track per entry of D_values."""
    table = simulated_table(D_values, rng, frames)
    return np.array([P.track_loglik(t, Acquisition(DT), u)
                     for t in table.sort("track_id", "frame").partition_by("track_id", maintain_order=True)])


def simulated_table(D_values, rng, frames=(5, 13), track_id0=0):
    """One track per entry of D_values, each with a distinct track_id."""
    tables = []
    for i, d in enumerate(D_values):
        n = rng.integers(*frames)
        sd = rng.uniform(.03, .045, (n, 2))
        pos = simulate(d, sd[:, 0], DT, rng, n_frames=n)
        tables.append(track_table(track_id0 + i, np.arange(n), pos, sd))
    return pl.concat(tables)


def joint_bumps(rng, alphas, u, n):
    """(n, len(alphas), len(u)) Gaussian log-likelihood bumps around two populations' centres."""
    centres = np.where(rng.random(n)[:, None] < .4, [.4, -3.], [1.1, -1.])
    centres = centres + rng.normal(0, [.15, .4], (n, 2))
    width = rng.uniform(.2, .6, (n, 2))
    return -.5 * (((alphas[None, :, None] - centres[:, 0, None, None]) / width[:, 0, None, None]) ** 2
                  + ((u[None, None, :] - centres[:, 1, None, None]) / width[:, 1, None, None]) ** 2)


def fbm_table(track_id, n, alpha, D_app, acquisition, rng):
    """One blurred fBm track whose apparent D (Brownian D with the same blurred step variance) is D_app."""
    dt, te = acquisition.dt_s, acquisition.exposure_s
    K = D_app * 2 * (dt - te / 3) / fgn_motion_covariance(1, dt, alpha, te)[0, 0]
    L = np.linalg.cholesky(K * fgn_motion_covariance(n - 1, dt, alpha, te))
    sd = rng.uniform(.02, .035, (n, 2))
    pos = np.vstack([np.zeros(2), np.cumsum(L @ rng.standard_normal((n - 1, 2)), axis=0)])
    return track_table(track_id, np.arange(n), pos + sd * rng.standard_normal((n, 2)), sd)


class DeconvolveTests(unittest.TestCase):
    def test_short_and_invalid_tracks_are_excluded_not_fatal(self):
        rng = np.random.default_rng(1)
        good = simulated_table([.05, .05, .05], rng, frames=(8, 15), track_id0=0)
        n = 5
        sd = rng.uniform(.03, .045, (n, 2))
        short = track_table(100, np.arange(2), simulate(.05, sd[:2, 0], DT, rng, n_frames=2), sd[:2])
        zero_sd = track_table(101, np.arange(n), simulate(.05, sd[:, 0], DT, rng, n_frames=n),
                              np.zeros_like(sd))
        table = pl.concat([good, short, zero_sd])
        result = D.deconvolve_tracks(table, Acquisition(DT))
        self.assertEqual(result.n_tracks, 3)
        self.assertEqual(result.n_excluded, 2)
        self.assertAlmostEqual(result.weights.sum(), 1., places=8)
        self.assertEqual(len(result.weights), len(result.u))

    def test_empty_after_exclusion_raises(self):
        rng = np.random.default_rng(2)
        n = 2
        sd = rng.uniform(.03, .045, (n, 2))
        short = track_table(0, np.arange(n), simulate(.05, sd[:, 0], DT, rng, n_frames=n), sd)
        with self.assertRaises(ValueError):
            D.deconvolve_tracks(short, Acquisition(DT))

    def test_min_frames_option_is_respected(self):
        rng = np.random.default_rng(3)
        table = simulated_table([.05] * 3, rng, frames=(5, 6), track_id0=0)  # all 5-frame tracks
        with self.assertRaises(ValueError):  # all excluded by the raised min_frames
            D.deconvolve_tracks(table, Acquisition(DT), options=GridPostOptions(min_frames=6))
        table2 = pl.concat([table, simulated_table([.05], rng, frames=(9, 10), track_id0=3)])
        result = D.deconvolve_tracks(table2, Acquisition(DT), options=GridPostOptions(min_frames=6))
        self.assertEqual(result.n_tracks, 1)
        self.assertEqual(result.n_excluded, 3)

    def test_default_log_prior_is_flat(self):
        rng = np.random.default_rng(4)
        table = simulated_table([.05] * 5, rng, track_id0=0)
        result = D.deconvolve_tracks(table, Acquisition(DT))
        np.testing.assert_array_equal(result.u, U)

    def test_options_grid_is_the_one_used(self):
        rng = np.random.default_rng(7)
        table = simulated_table([.05] * 5, rng, track_id0=0)
        options = GridPostOptions(D_min_um2_s=1e-3, D_max_um2_s=1., n_D=101)
        result = D.deconvolve_tracks(table, Acquisition(DT), options=options)
        np.testing.assert_array_equal(result.u, options.u_D())
        self.assertEqual(len(result.weights), 101)

    def test_tangent_gradient_and_hessian_match_finite_differences(self):
        rng = np.random.default_rng(5)
        lls = track_lls(np.exp(rng.normal(np.log(.05), .8, 40)), rng)[:, ::25]
        m = D._LogG(np.exp(lls - lls.max(axis=1, keepdims=True)), 25 * (U[1] - U[0]))
        Q = .3 * m.Omega + m.tilt
        x = .1 * rng.standard_normal(m.K)
        F, gt, H = m.local(x, Q)
        self.assertAlmostEqual(F, m.value(x, Q))

        def f(z):
            return m.value(x + m.P @ z, Q)

        E = np.eye(m.K - 1)
        h = 1e-5
        g_fd = np.array([(f(h * e) - f(-h * e)) / (2 * h) for e in E])
        np.testing.assert_allclose(g_fd, gt, atol=1e-6 * np.abs(gt).max())
        h = 1e-4
        H_fd = -np.array([[(f(h * a + h * b) - f(h * a - h * b) - f(-h * a + h * b) + f(-h * a - h * b))
                           / (4 * h * h) for b in E] for a in E])
        np.testing.assert_allclose(H_fd, H, atol=1e-5 * np.abs(H).max())

    def test_laplace_evidence_matches_importance_sampling(self):
        # Absolute log evidence on a 5-cell grid: prior normalization and determinants included.
        rng = np.random.default_rng(2)
        u = np.linspace(np.log(1e-3), 0, 5)
        lls = track_lls(np.exp(rng.normal(np.log(.05), .8, 300)), rng, u=u)
        m = D._LogG(np.exp(lls - lls.max(axis=1, keepdims=True)), u[1] - u[0])
        for lam in (3., 30.):
            mode = m.fit(lam, np.zeros(m.K))
            Q = lam * m.Omega + m.tilt
            prop = multivariate_t(loc=np.zeros(m.K - 1), shape=1.5 * np.linalg.inv(mode.C @ mode.C.T), df=5)
            z = prop.rvs(100_000, random_state=rng)
            X = mode.x[:, None] + m.P @ z.T
            ll = np.sum(np.log(m.L @ D._softmax(X)), axis=0)
            Zc = m.P.T @ X
            Qt = m.P.T @ Q @ m.P
            log_prior = (-.5 * np.einsum("in,ij,jn->n", Zc, Qt, Zc) + .5 * np.linalg.slogdet(Qt)[1]
                         - (m.K - 1) / 2 * np.log(2 * np.pi))
            lw = ll + log_prior - prop.logpdf(z)
            self.assertAlmostEqual(mode.log_evidence, logsumexp(lw) - np.log(len(lw)), delta=.1)

    def test_hmc_matches_importance_sampling(self):
        # 5 cells, 30 tracks: weak enough that the posterior is far from Gaussian, small
        # enough that importance sampling from a heavy-tailed proposal is exact.
        rng = np.random.default_rng(8)
        u = np.linspace(np.log(1e-3), 0, 5)
        lls = track_lls(np.exp(rng.normal(np.log(.05), .8, 30)), rng, u=u)
        m = D._LogG(np.exp(lls - lls.max(axis=1, keepdims=True)), u[1] - u[0])
        mode = m.fit(1., np.zeros(m.K))
        G = m.hmc(mode, 50, 400, rng)
        np.testing.assert_allclose(G.sum(axis=1), 1, atol=1e-12)
        self.assertGreaterEqual(G.min(), 0)
        Q = mode.lam * m.Omega + m.tilt
        prop = multivariate_t(loc=np.zeros(m.K - 1), shape=2 * np.linalg.inv(mode.C @ mode.C.T), df=4)
        z = prop.rvs(200_000, random_state=rng)
        X = mode.x[:, None] + m.P @ z.T
        Gi = D._softmax(X)
        lw = np.sum(np.log(m.L @ Gi), axis=0) - .5 * np.einsum("in,ij,jn->n", X, Q, X) - prop.logpdf(z)
        w = np.exp(lw - lw.max())
        w /= w.sum()
        mean = Gi @ w
        sd = np.sqrt((Gi - mean[:, None]) ** 2 @ w)
        np.testing.assert_allclose(G.mean(axis=0), mean, atol=.15 * sd.max())
        np.testing.assert_allclose(G.std(axis=0), sd, rtol=.15, atol=.01 * sd.max())

    def test_bands_stay_off_grid_edges_no_track_reaches(self):
        # Default 1e-4..10 grid, all tracks at D = 0.05: the posterior puts no mass past the
        # data. Laplace draws here pile the mass into an empty edge cell (module docstring).
        rng = np.random.default_rng(11)
        result = D.deconvolve(track_lls(np.full(500, .05), rng), U, n_samples=500)
        outside = result.samples[:, (U < np.log(1e-3)) | (U > np.log(1.))].sum(axis=1)
        self.assertLess(np.quantile(outside, .99), .01)
        lo, hi = result.band(.95, cumulative=True)
        self.assertTrue(np.all(hi[U < np.log(.02)] < .02) and np.all(lo[U > np.log(.12)] > .98))

    def test_recovers_mass_of_each_mode_with_an_interior_evidence_maximum(self):
        rng = np.random.default_rng(6)
        D_values = np.exp(np.log(rng.choice([.02, .2], 300)) + .15 * rng.standard_normal(300))
        table = simulated_table(D_values, rng, frames=(8, 21))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no edge or convergence warning
            result = D.deconvolve_tracks(table, Acquisition(DT), log_prior=P.log_uniform(1e-3, 1., U))
        mid = np.log(np.sqrt(.02 * .2))
        upper = result.u > mid
        truth = np.mean(D_values > np.sqrt(.02 * .2))
        self.assertLess(abs(result.weights[upper].sum() - truth), .06)
        lo, hi = np.quantile(result.samples[:, upper].sum(axis=1), [.0005, .9995])
        self.assertTrue(lo < truth < hi)
        self.assertTrue(np.all(result.samples[:, ~np.isfinite(P.log_uniform(1e-3, 1., U))] == 0))
        lo, hi = result.band(.9, cumulative=True)
        self.assertTrue(np.all(lo <= hi))
        self.assertAlmostEqual(hi[-1], 1.)

    def test_evidence_smooths_a_broad_truth_more_than_a_narrow_one(self):
        rng = np.random.default_rng(10)
        narrow = np.exp(np.log(rng.choice([.02, .2], 1000)) + .15 * rng.standard_normal(1000))
        broad = np.exp(rng.normal(np.log(.05), .8, 1000))
        prior = P.log_uniform(1e-3, 1., U)
        lam = [D.deconvolve(track_lls(d, rng), U, prior, n_samples=10).lam for d in (narrow, broad)]
        self.assertGreater(lam[1], 30 * lam[0])

    def test_flat_likelihoods_leave_the_alpha_distribution_unchanged(self):
        # On the alpha grid, as the per-track posteriors are used: a track with no
        # information has a flat likelihood and must not add mass at the prior's median.
        alphas = GridPostOptions().alphas()
        rng = np.random.default_rng(9)
        centers = rng.choice([.3, 1.4], 200) + .1 * rng.standard_normal(200)
        informative = -.5 * ((alphas[None] - centers[:, None]) / .15) ** 2
        flat_rows = np.zeros((400, len(alphas)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            alone = D.deconvolve(informative, alphas)
            mixed = D.deconvolve(np.vstack([informative, flat_rows]), alphas)
        np.testing.assert_allclose(mixed.log_evidence, alone.log_evidence, atol=1e-8)
        np.testing.assert_allclose(mixed.weights, alone.weights, atol=1e-8)
        for cumulative in (False, True):
            np.testing.assert_allclose(mixed.band(.9, cumulative), alone.band(.9, cumulative), atol=.05)
        self.assertLess(mixed.weights[(alphas > .7) & (alphas < 1.1)].sum(), .02)

    def test_rejects_uneven_grids_and_tiny_supports(self):
        with self.assertRaises(ValueError):
            D.deconvolve(np.zeros((3, 4)), np.array([0., 1., 3., 4.]))
        with self.assertRaises(ValueError):
            D.deconvolve(np.zeros((3, 4)), np.arange(4.), np.array([0., 0., -np.inf, -np.inf]))

    # ------------------------------------------------------------------
    # The joint (alpha, ln D) distribution: deconvolve_joint
    # ------------------------------------------------------------------

    def test_joint_tangent_matches_dense_projection(self):
        rng = np.random.default_rng(20)
        m = D._LogG2(rng.uniform(.1, 1, (6, 20)), 4, .5, .2)
        np.testing.assert_allclose(m.P.T @ m.P, np.eye(19), atol=1e-12)
        np.testing.assert_allclose(m.P.T @ np.ones(20), 0, atol=1e-12)
        M = rng.standard_normal((20, 20))
        M = M + M.T
        np.testing.assert_allclose(m._tangent(M), m.P.T @ M @ m.P, atol=1e-12)

    def test_joint_gradient_and_hessian_match_finite_differences(self):
        rng = np.random.default_rng(21)
        m = D._LogG2(rng.uniform(.01, 1, (40, 20)), 4, .5, .2)
        Q = m.Q((.3, .05))
        x = .1 * rng.standard_normal(m.K)
        F, gt, H = m.local(x, Q)
        self.assertAlmostEqual(F, m.value(x, Q))

        def f(z):
            return m.value(x + m.P @ z, Q)

        E = np.eye(m.K - 1)
        g_fd = np.array([(f(1e-5 * e) - f(-1e-5 * e)) / 2e-5 for e in E])
        np.testing.assert_allclose(g_fd, gt, atol=1e-6 * np.abs(gt).max())
        h = 1e-4
        H_fd = -np.array([[(f(h * a + h * b) - f(h * a - h * b) - f(-h * a + h * b) + f(-h * a - h * b))
                           / (4 * h * h) for b in E] for a in E])
        np.testing.assert_allclose(H_fd, H, atol=1e-5 * np.abs(H).max())

    def test_joint_prior_normalization_is_the_tangent_pseudo_determinant(self):
        """The Kronecker-sum eigenvalue formula against a dense log-determinant on the tangent space."""
        m = D._LogG2(np.ones((2, 30)), 5, .4, .3)
        for lam in ((1., 1.), (1e-3, 10.), (50., 1e-4)):
            Qt = m.P.T @ m.Q(lam) @ m.P
            self.assertAlmostEqual(m.log_prior_norm(lam), .5 * np.linalg.slogdet(Qt)[1], places=6)

    def test_joint_laplace_evidence_matches_importance_sampling(self):
        # 3 x 3 cells: prior normalization and determinants included, as in the 1D test. The
        # gap is Laplace's own error (up to 0.13 nats here, varying with lam; two importance
        # proposals agree to 0.01), not a missing constant, which would show at every lam.
        rng = np.random.default_rng(22)
        a, d = np.meshgrid(np.arange(3.), np.arange(3.), indexing="ij")
        centres = rng.normal([1, 1], .7, (300, 2))
        lls = -.5 * ((a[None] - centres[:, 0, None, None]) ** 2 + (d[None] - centres[:, 1, None, None]) ** 2)
        F = lls.reshape(300, -1)
        m = D._LogG2(np.exp(F - F.max(axis=1, keepdims=True)), 3, 1., 1.)
        for lam in ((1., 1.), (10., .1)):
            mode = m.fit(lam, np.zeros(m.K))
            Q = m.Q(lam)
            prop = multivariate_t(loc=np.zeros(m.K - 1), shape=1.5 * np.linalg.inv(mode.C @ mode.C.T), df=5)
            z = prop.rvs(100_000, random_state=rng)
            X = mode.x[:, None] + m.P @ z.T
            ll = np.sum(np.log(m.L @ D._softmax(X)), axis=0)
            Zc = m.P.T @ X
            Qt = m.P.T @ Q @ m.P
            log_prior = (-.5 * np.einsum("in,ij,jn->n", Zc, Qt, Zc) + .5 * np.linalg.slogdet(Qt)[1]
                         - (m.K - 1) / 2 * np.log(2 * np.pi))
            lw = ll + log_prior - prop.logpdf(z)
            self.assertAlmostEqual(mode.log_evidence, logsumexp(lw) - np.log(len(lw)), delta=.2)

    def test_joint_search_is_at_least_as_good_as_a_grid(self):
        rng = np.random.default_rng(23)
        alphas, u = np.linspace(.1, 1.7, 9), np.linspace(-6, 1, 12)
        lls = joint_bumps(rng, alphas, u, 300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = D.deconvolve_joint(lls, alphas, u, n_samples=50)
        F = lls.reshape(len(lls), -1)
        m = D._LogG2(np.exp(F - F.max(axis=1, keepdims=True)), len(alphas), u[1] - u[0], alphas[1] - alphas[0])
        grid = [m.fit((10. ** a, 10. ** b), np.zeros(m.K)).log_evidence
                for a in np.arange(3, -7, -2.) for b in np.arange(3, -7, -2.)]
        self.assertGreaterEqual(fit.log_evidence[:, 2].max(), max(grid) - .05)

    def test_joint_flat_rows_leave_it_unchanged(self):
        """A track with no information about (alpha, D) must not move the distribution."""
        rng = np.random.default_rng(24)
        alphas, u = np.linspace(.1, 1.7, 9), np.linspace(-6, 1, 12)
        informative = joint_bumps(rng, alphas, u, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alone = D.deconvolve_joint(informative, alphas, u, n_samples=100)
            mixed = D.deconvolve_joint(np.vstack([informative, np.zeros((300, 9, 12))]), alphas, u, n_samples=100)
        np.testing.assert_allclose(mixed.weights, alone.weights, atol=1e-8)
        np.testing.assert_allclose(mixed.lam, alone.lam, rtol=1e-6)
        np.testing.assert_allclose(mixed.samples.mean(axis=0), alone.samples.mean(axis=0), atol=.02)

    def test_joint_recovers_two_populations_from_simulated_tracks(self):
        """Caged (alpha 0.3, D 0.02) and Brownian (alpha 1, D 0.3) tracks, exposure blur modelled."""
        rng = np.random.default_rng(25)
        acquisition = Acquisition(DT, .02)
        options = GridPostOptions(n_D=101, joint_D_bin=5, n_alpha=9, alpha_min=.1, alpha_max=1.7)
        caged = rng.random(300) < .35
        table = pl.concat([fbm_table(i, rng.integers(10, 31), *((.3, .02) if c else (1., .3)), acquisition, rng)
                           for i, c in enumerate(caged)])
        post = analyze_tracks(table, acquisition, options, keep_posteriors=True).posteriors
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no bound or convergence warning
            fit = D.deconvolve_joint(post.log_post_joint, options.alphas(), options.u_joint_D(), n_samples=500)
        low = options.alphas() < .65
        mass = fit.samples[:, low, :].sum(axis=(1, 2))
        self.assertLess(abs(fit.weights[low].sum() - caged.mean()), .07)
        lo, hi = np.quantile(mass, [.0005, .9995])
        self.assertTrue(lo < caged.mean() < hi)
        np.testing.assert_allclose(fit.samples.sum(axis=(1, 2)), 1, atol=1e-12)

    def test_joint_rejects_bad_shapes_and_uneven_grids(self):
        alphas, u = np.linspace(.1, 1.7, 9), np.linspace(-6, 1, 12)
        with self.assertRaisesRegex(ValueError, "log_post"):
            D.deconvolve_joint(np.zeros((3, 12, 9)), alphas, u)
        with self.assertRaisesRegex(ValueError, "evenly"):
            D.deconvolve_joint(np.zeros((3, 9, 12)), alphas, np.r_[u[:-1], 5.])

    def test_import_does_not_load_bayes_or_plotting(self):
        import subprocess
        import sys
        code = ("import diffusionkit.gridpost, sys; "
                "assert not any(x in sys.modules for x in ('jax','numpyro','matplotlib'))")
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
