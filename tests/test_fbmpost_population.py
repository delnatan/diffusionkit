"""The population of (alpha, D_ref) across tracks: the 2D deconvolution and the fbmpost cells it is fitted to."""
import unittest

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.fbmpost import (AlphaDPopulation, FbmPostOptions, JointGrid, analyze_tracks, deconvolve_tracks,
                                  simulate_tracks)
from diffusionkit.fbmpost import likelihood as L
from diffusionkit.fbmpost.population import informative_range
from diffusionkit.gridpost.deconvolve import deconvolve_2d

ACQ = Acquisition(.044, .02)


def gaussian_tracks(rng, n, u, v, sd, corr):
    """Per-track Gaussian log-likelihoods on u x v around (x, y) drawn from a correlated normal population."""
    xy = rng.multivariate_normal([0, 0], [[1, corr], [corr, 1]], n) * .5
    obs = xy + sd * rng.standard_normal(xy.shape)
    return -.5 * (((u[None, :, None] - obs[:, 0, None, None]) ** 2 + (v[None, None, :] - obs[:, 1, None, None]) ** 2)
                  / sd ** 2), xy


class Deconvolve2DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        cls.u = cls.v = np.linspace(-2.5, 2.5, 21)
        cls.lls, cls.truth = gaussian_tracks(rng, 400, cls.u, cls.v, .3, .7)
        cls.fit = deconvolve_2d(cls.lls, cls.u, cls.v, n_samples=200)

    def test_recovers_a_correlated_population(self):
        s, u, v = self.fit.samples, self.u[None, :, None], self.v[None, None, :]
        mu, mv = (s * u).sum((1, 2)), (s * v).sum((1, 2))
        su, sv = np.sqrt((s * u ** 2).sum((1, 2)) - mu ** 2), np.sqrt((s * v ** 2).sum((1, 2)) - mv ** 2)
        corr = ((s * u * v).sum((1, 2)) - mu * mv) / (su * sv)
        self.assertLess(abs(np.median(mu)), .1)
        self.assertLess(abs(np.median(su) - .5), .1)  # the population's SD, not the observations' (0.58)
        self.assertLess(abs(np.median(corr) - .7), .15)

    def test_weights_and_draws_are_distributions(self):
        self.assertAlmostEqual(self.fit.weights.sum(), 1, places=10)
        np.testing.assert_allclose(self.fit.samples.sum(axis=(1, 2)), 1, atol=1e-10)
        self.assertTrue((self.fit.samples >= 0).all())

    def test_a_track_with_no_information_changes_nothing(self):
        flat = np.zeros((5, len(self.u), len(self.v)))
        fit = deconvolve_2d(np.concatenate([self.lls, flat]), self.u, self.v, n_samples=50)
        self.assertEqual(fit.lam, self.fit.lam)
        np.testing.assert_allclose(fit.weights, self.fit.weights, atol=1e-6)

    def test_rejects_an_uneven_grid(self):
        with self.assertRaises(ValueError):
            deconvolve_2d(self.lls, self.u ** 3, self.v)


class JointLikelihoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2)
        cls.options = FbmPostOptions()
        short = pl.DataFrame({"track_id": [99, 99], "frame": [0, 1], "x_um": [0., .01], "y_um": [0., 0.],
                              "sigma_x_um": [.013] * 2, "sigma_y_um": [.013] * 2})
        cls.table = pl.concat([simulate_tracks(40, 40, .6, .01, .013, ACQ, rng=rng), short])
        cls.result = analyze_tracks(cls.table, ACQ, cls.options, keep_likelihoods=True)

    def test_one_slab_per_fitted_track_on_the_joint_grid(self):
        lik, grid = self.result.likelihoods, JointGrid()
        self.assertEqual(list(lik.track_ids), list(range(40)))  # the 2-frame track is excluded
        self.assertEqual(lik.loglik.shape, (40, len(grid.alphas(self.options)), len(grid.log10_D_ref())))
        np.testing.assert_allclose(lik.loglik.max(axis=(1, 2)), 0)
        np.testing.assert_allclose(lik.D_ref_floor_um2_s, .013 ** 2 / self.options.tau_ref_s)
        self.assertIsNone(analyze_tracks(self.table, ACQ, self.options).likelihoods)

    def test_alpha_cells_have_one_centred_on_one(self):
        alphas = JointGrid().alphas(self.options)
        self.assertTrue(np.isclose(alphas, 1).any())
        np.testing.assert_allclose(np.diff(alphas), .06)

    def test_each_track_peaks_at_its_fit(self):
        lik, fits = self.result.likelihoods, self.result.fits.filter(pl.col("status") == "ok")
        flat = lik.loglik.reshape(len(lik.track_ids), -1).argmax(axis=1)
        a, d = np.unravel_index(flat, lik.loglik.shape[1:])
        # The cells hold D at the frame interval; D_ref at tau_ref is D (tau_ref / tau_s)^(alpha - 1).
        D_ref = lik.log10_D_ref[d] + (lik.alphas[a] - 1) * np.log10(lik.tau_ref_s / lik.tau_s)
        self.assertLess(np.median(np.abs(D_ref - fits["log10_D_ref_hat"].to_numpy())), .15)
        self.assertLess(np.median(np.abs(lik.alphas[a] - fits["alpha_hat"].to_numpy())), .06)

    def test_cells_average_the_likelihood_over_each_cell(self):
        """A cell is the mean likelihood over its alpha rows and its log10 D_ref span, against a dense midpoint sum.
        Cells are stored relative to the track's best cell, so two cells' difference is compared."""
        track = self.table.filter(pl.col("track_id") == 3)
        w = L.whiten_track(track, ACQ, self.options.alphas())
        for points in (8, 4):  # fewer points per cell: cheaper, still within a few hundredths of a nat here
            grid = JointGrid(cell_points=points)
            cells = L.cells(w, self.options, grid, self.options.tau_ref_s)[0]
            a, d = 5, int(np.argmax(cells[5]))
            rows = slice(a * grid.alpha_group, (a + 1) * grid.alpha_group)
            step = grid.d_log10_D_ref

            def dense_cell(k):
                x = grid.log10_D_ref()[k] + step * ((np.arange(400) + .5) / 400 - .5)
                ll = L.log_likelihood_surface(track, ACQ, self.options, x)[rows]
                return np.log(np.exp(ll - 100).mean()) + 100

            for k in (d - 1, d + 1):  # near the peak, where a mixture can tell cells apart
                self.assertAlmostEqual(cells[a, d] - cells[a, k], dense_cell(d) - dense_cell(k), delta=.02)
        with self.assertRaises(ValueError):
            JointGrid(cell_points=0)

    def test_the_range_drops_cells_no_track_reaches(self):
        keep = informative_range(self.result.likelihoods)
        D = self.result.likelihoods.log10_D_ref[keep]
        self.assertGreater(D[-1], -2)  # the tracks sit at log10 D_ref = -2
        self.assertLess(D[-1], 1)

    def test_population_of_identical_tracks_is_narrow_at_their_alpha(self):
        pop = deconvolve_tracks(self.result.likelihoods, n_samples=200)
        self.assertEqual(pop.n_tracks, 40)
        self.assertLess(abs(np.median(pop.alpha_mean()) - .6), .1)
        self.assertLess(np.median(pop.alpha_sd()), .15)
        self.assertGreater(np.median(pop.p_alpha_below(1.)), .9)
        self.assertAlmostEqual(pop.log10_floor, np.log10(.013 ** 2 / self.options.tau_ref_s))
        self.assertLess(np.median(pop.mass_below_floor()), .2)


class BrownianPopulationTests(unittest.TestCase):
    """One D and alpha = 1 at a 23 ms frame, D_ref = 0.1 um^2/s sitting on an edge between 0.2-decade cells at
    tau_ref = 0.1 s: there the old grid split alpha into 0.82 and 1.06 with a population SD near 0.1."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        acq = Acquisition(.023, .01)
        tables = [simulate_tracks(60, n, 1., .1, .005, acq, rng=rng, first_track_id=60 * k)
                  for k, n in enumerate((4, 6, 10, 20, 40))]
        cls.result = analyze_tracks(pl.concat(tables), acq, FbmPostOptions(), keep_likelihoods=True)

    def test_cells_hold_D_at_the_frame_interval(self):
        lik = self.result.likelihoods
        self.assertAlmostEqual(lik.tau_s, .023)
        self.assertAlmostEqual(lik.tau_ref_s, .1)

    def test_one_alpha_cell_at_one(self):
        pop = deconvolve_tracks(self.result.likelihoods, n_samples=200)
        self.assertAlmostEqual(pop.alphas[np.argmax(pop.alpha_weights())], 1.)
        self.assertGreater(pop.alpha_weights().max(), .5)  # the old grid: 0.48 at 0.88, 0.24 at 1.12
        self.assertLess(np.median(pop.alpha_sd()), .06)  # the old grid: 0.11
        # Reported at tau_ref: the D_ref margin peaks at 0.1.
        self.assertLess(abs(pop.log10_D_ref[np.argmax(pop.weights.sum(axis=0))] + 1), .1)


class AlphaDPopulationTests(unittest.TestCase):
    def setUp(self):
        alphas, D = np.array([.4, .7, 1., 1.3]), np.array([-4., -3., -2.])
        w = np.zeros((4, 3))
        w[0, 0], w[2, 2] = .5, .5  # half at (0.4, below the floor), half at (1.0, above it)
        self.pop = AlphaDPopulation(alphas, D, w, w[None], (1., 1.), 2, -2.5)

    def test_cut_splits_a_cell_at_its_centre(self):
        self.assertAlmostEqual(self.pop.p_alpha_below(1.)[0], .75)
        self.assertAlmostEqual(self.pop.p_alpha_below(.85)[0], .5)

    def test_above_floor_reads_drop_the_mass_below_it(self):
        self.assertAlmostEqual(self.pop.mass_below_floor()[0], .5)
        self.assertAlmostEqual(self.pop.alpha_mean(above_floor=True)[0], 1.)
        np.testing.assert_allclose(self.pop.alpha_weights(above_floor=True), [0, 0, 1, 0])
        self.assertAlmostEqual(self.pop.alpha_mean()[0], .7)
        self.assertAlmostEqual(self.pop.corr()[0], 1.)


if __name__ == "__main__":
    unittest.main()
