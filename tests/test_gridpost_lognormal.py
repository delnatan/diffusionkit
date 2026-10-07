"""Complete pooling (shared D) and log-normal partial pooling (diffusionkit.gridpost.lognormal)."""
import unittest

import numpy as np
import polars as pl
from scipy.special import logsumexp

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions, analyze_tracks, by_track_length, cdf_distance
from diffusionkit.gridpost import lognormal as LN
from diffusionkit.gridpost import posterior as P

DT = .033
U = GridPostOptions().u_D()
GRID = np.linspace(np.log(1e-4), np.log(10), 241)


def simulated_table(D_values, rng, frames=(5, 13)):
    """One Brownian track per entry of D_values, instantaneous positions with per-frame SDs."""
    tables = []
    for i, d in enumerate(D_values):
        n = int(rng.integers(*frames))
        sd = rng.uniform(.03, .045, (n, 2))
        pos = np.cumsum(rng.normal(0, np.sqrt(2 * d * DT), (n, 2)), axis=0) + sd * rng.standard_normal((n, 2))
        tables.append(pl.DataFrame({"track_id": i, "frame": np.arange(n), "x_um": pos[:, 0], "y_um": pos[:, 1],
                                    "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1]}))
    return pl.concat(tables)


def track_lls(D_values, rng, frames=(5, 13)):
    """Per-track log-likelihood rows on U, one track per entry of D_values."""
    table = simulated_table(D_values, rng, frames)
    return np.array([P.track_loglik(t, Acquisition(DT), U) for t in table.partition_by("track_id", maintain_order=True)])


def gaussian_rows(centers, widths, u=GRID):
    """Log-likelihood rows that are exactly Gaussian in ln D: the marginal under N(mu, sigma) is known."""
    return -.5 * ((u[None] - centers[:, None]) / widths[:, None]) ** 2


def exact_log_post(centers, widths, mu, sigma):
    """The (mu, sigma) log posterior for Gaussian rows, flat priors, no truncation."""
    M, S = np.meshgrid(mu, sigma, indexing="ij")
    var = S[..., None] ** 2 + widths ** 2
    lp = (-.5 * (centers - M[..., None]) ** 2 / var - .5 * np.log(var)).sum(-1)
    return lp - logsumexp(lp)


class HatWeightTests(unittest.TestCase):
    def test_exact_against_quadrature_of_the_linear_interpolant(self):
        L = np.exp(-.5 * ((GRID - np.log(.05)) / .3) ** 2)
        x = np.linspace(GRID[0], GRID[-1], 1_000_001)  # dense trapezoid: the interpolant is piecewise linear
        for mu, s in [(np.log(.05), .5), (np.log(.2), .02), (np.log(1e-3), 2.), (GRID[2], .3)]:
            g = LN.hat_weights(GRID, mu, s)[:, 0]
            self.assertAlmostEqual(g.sum(), 1., places=12)
            w = np.exp(-.5 * ((x - mu) / s) ** 2)
            exact = np.trapezoid(np.interp(x, GRID, L) * w, x) / np.trapezoid(w, x)
            self.assertAlmostEqual(L @ g / exact, 1., places=8)

    def test_zero_sigma_is_linear_interpolation_and_the_limit_of_small_sigma(self):
        L = np.exp(-.5 * ((GRID - np.log(.05)) / .3) ** 2)
        mu = np.log(.07)
        at_zero = L @ LN.hat_weights(GRID, mu, 0.)[:, 0]
        self.assertAlmostEqual(at_zero, np.interp(mu, GRID, L), places=12)
        self.assertAlmostEqual(L @ LN.hat_weights(GRID, mu, 1e-7)[:, 0], at_zero, places=9)
        np.testing.assert_allclose(LN.hat_weights(GRID, GRID[-1], 0.)[:, 0], np.eye(len(GRID))[-1], atol=1e-10)

    def test_a_far_track_gets_its_tiny_likelihood_not_a_rounding_floor(self):
        L = np.exp(-.5 * ((GRID - np.log(1e-3)) / .1) ** 2)
        m = [L @ LN.hat_weights(GRID, np.log(.1), s)[:, 0] for s in (.4, .3, .25)]
        self.assertLess(m[0], 1e-20)  # 11 sigma away
        self.assertTrue(m[0] > m[1] > m[2] > 0)

    def test_negative_sigma_raises(self):
        with self.assertRaises(ValueError):
            LN.hat_weights(GRID, 0., -.1)


class SharedDTests(unittest.TestCase):
    def test_matches_the_product_of_gaussian_rows_below_the_grid_step(self):
        rng = np.random.default_rng(0)
        centers, widths = np.log(.05) + .4 * rng.standard_normal(3000), rng.uniform(.2, 1., 3000)
        shared = LN.fit_shared_D(gaussian_rows(centers, widths), GRID)
        prec = (1 / widths ** 2).sum()
        mean, sd = (centers / widths ** 2).sum() / prec, prec ** -.5
        self.assertLess(sd, (GRID[1] - GRID[0]) / 4)  # narrower than a grid cell
        s = shared.summary(.9)
        self.assertAlmostEqual(np.log(s["median"]), mean, delta=.05 * sd)
        self.assertAlmostEqual(np.log(s["hi"]) - np.log(s["lo"]), 2 * 1.6449 * sd, delta=.05 * sd)
        self.assertFalse(shared.at_grid_edge)
        self.assertEqual(shared.n_tracks, 3000)

    def test_peak_at_the_grid_end_warns(self):
        rows = gaussian_rows(np.full(5, GRID[-1] + 1.), np.full(5, .5))
        with self.assertWarnsRegex(UserWarning, "grid end"):
            self.assertTrue(LN.fit_shared_D(rows, GRID).at_grid_edge)


class LogNormalTests(unittest.TestCase):
    def test_matches_the_exact_posterior_for_gaussian_rows(self):
        rng = np.random.default_rng(1)
        n = 400
        centers = np.log(.05) + .6 * rng.standard_normal(n)
        widths = rng.uniform(.15, .8, n)
        centers = centers + widths * rng.standard_normal(n)
        fit = LN.fit_lognormal(gaussian_rows(centers, widths), GRID, n_samples=200)
        self.assertEqual(fit.problem, "")
        exact = exact_log_post(centers, widths, fit.mu, fit.sigma)
        tv = .5 * np.abs(np.exp(fit.log_post) - np.exp(exact)).sum()
        self.assertLess(tv, .01)
        # the grid zoomed in: each axis resolves the posterior with many points
        for name, axis in (("mu", fit.mu), ("sigma", fit.sigma)):
            p = np.exp(fit.marginal(name))
            self.assertGreater((p > 1e-3 * p.max()).sum(), len(axis) / 3)

    def test_recovers_a_simulated_log_normal_population(self):
        rng = np.random.default_rng(2)
        mu, sigma = np.log(.05), .7
        lls = track_lls(np.exp(mu + sigma * rng.standard_normal(500)), rng, frames=(5, 30))
        s = LN.fit_lognormal(lls, U, n_samples=300).summary(.95)
        self.assertLess(s["D_median_um2_s"]["lo"], np.exp(mu))
        self.assertGreater(s["D_median_um2_s"]["hi"], np.exp(mu))
        self.assertLess(s["sigma_ln_D"]["lo"], sigma)
        self.assertGreater(s["sigma_ln_D"]["hi"], sigma)
        self.assertLess(s["D_mean_um2_s"]["lo"], np.exp(mu + sigma ** 2 / 2) * 1.15)

    def test_one_shared_D_gives_sigma_near_zero_and_agrees_with_the_shared_fit(self):
        rng = np.random.default_rng(3)
        lls = track_lls(np.full(600, .1), rng, frames=(5, 30))
        fit = LN.fit_lognormal(lls, U, n_samples=300)
        s = fit.summary(.9)
        self.assertEqual(fit.sigma[0], 0.)  # the grid reaches sigma = 0: complete pooling is in the model
        self.assertLess(s["sigma_ln_D"]["hi"], .2)
        shared = LN.fit_shared_D(lls, U).summary(.9)
        self.assertLess(shared["lo"], .1)
        self.assertGreater(shared["hi"], .1)
        self.assertAlmostEqual(np.log(s["D_median_um2_s"]["median"]), np.log(shared["median"]), delta=.03)

    def test_flat_rows_leave_the_posterior_unchanged(self):
        rng = np.random.default_rng(4)
        centers, widths = np.log(.05) + .5 * rng.standard_normal(200), np.full(200, .3)
        rows = gaussian_rows(centers, widths)
        alone = LN.fit_lognormal(rows, GRID, n_samples=50)
        mixed = LN.fit_lognormal(np.vstack([rows, np.zeros((300, len(GRID)))]), GRID, n_samples=50)
        np.testing.assert_allclose(mixed.mu, alone.mu)
        np.testing.assert_allclose(mixed.sigma, alone.sigma)
        np.testing.assert_allclose(mixed.log_post, alone.log_post, atol=1e-9)

    def test_draws_and_samples_agree_and_read_like_a_deconvolution(self):
        rng = np.random.default_rng(5)
        rows = gaussian_rows(np.log(.05) + .5 * rng.standard_normal(200), np.full(200, .3))
        fit = LN.fit_lognormal(rows, GRID, n_samples=64, rng=np.random.default_rng(1))
        self.assertEqual(fit.samples.shape, (64, len(GRID)))
        np.testing.assert_allclose(fit.samples.sum(1), 1.)
        np.testing.assert_allclose(fit.samples[7], LN.hat_weights(GRID, *fit.draws[7])[:, 0])
        self.assertAlmostEqual(fit.weights.sum(), 1.)
        lo, hi = fit.band(.9, cumulative=True)
        self.assertTrue(np.all(lo <= hi + 1e-12))
        self.assertEqual(fit.mass(0, .05).shape, (64,))
        self.assertEqual(cdf_distance(fit, fit).max(), 0.)

    def test_a_population_wider_than_the_sigma_bound_says_so(self):
        rng = np.random.default_rng(6)
        rows = gaussian_rows(np.log(.05) + 1. * rng.standard_normal(300), np.full(300, .2))
        with self.assertWarnsRegex(UserWarning, "sigma_max"):
            fit = LN.fit_lognormal(rows, GRID, sigma_max=.3, n_samples=20)
        self.assertIn("sigma_max", fit.problem)
        self.assertEqual(fit.sigma[-1], .3)

    def test_bad_inputs_raise(self):
        for lls, u in ((np.zeros((3, 5)), GRID), (np.zeros((0, len(GRID))), GRID),
                       (np.zeros((3, 4)), np.array([0., 1., 3., 4.]))):
            with self.assertRaises(ValueError):
                LN.fit_lognormal(lls, u)
            with self.assertRaises(ValueError):
                LN.fit_shared_D(lls, u)
        with self.assertRaises(ValueError):
            LN.fit_lognormal(np.zeros((3, len(GRID))), GRID, sigma_max=0.)


class FromAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        table = simulated_table(np.exp(np.log(.05) + .5 * rng.standard_normal(150)), rng)
        short = table.filter(pl.col("track_id") == 0).head(2).with_columns(pl.lit(999).alias("track_id"))
        cls.analysis = analyze_tracks(pl.concat([table, short]), Acquisition(DT), GridPostOptions(),
                                      keep_likelihoods=True)

    def test_wrappers_count_tracks_and_match_the_row_functions(self):
        pop = LN.lognormal_tracks(self.analysis, n_samples=50)
        self.assertEqual((pop.n_tracks, pop.n_excluded), (150, 1))
        self.assertEqual(pop.acquisition, Acquisition(DT))
        direct = LN.fit_lognormal(self.analysis.likelihoods.loglik_D, U, n_samples=50)
        np.testing.assert_array_equal(pop.log_post, direct.log_post)
        self.assertEqual(LN.shared_D_tracks(self.analysis).summary(), LN.fit_shared_D(
            self.analysis.likelihoods.loglik_D, U).summary())

    def test_by_track_length_takes_the_log_normal_as_its_population(self):
        pop = LN.lognormal_tracks(self.analysis, n_samples=40)
        comp = by_track_length(self.analysis.likelihoods, U, pop, n_draws=10)
        self.assertEqual(comp.partially_pooled.shape[0], 10)
        np.testing.assert_allclose(comp.partially_pooled.sum(axis=(1, 2)), 1., atol=1e-10)

    def test_needs_kept_likelihoods(self):
        bare = analyze_tracks(simulated_table([.05, .1], np.random.default_rng(0)), Acquisition(DT))
        for f in (LN.lognormal_tracks, LN.shared_D_tracks):
            with self.assertRaisesRegex(ValueError, "keep_likelihoods"):
                f(bare)


if __name__ == "__main__":
    unittest.main()
