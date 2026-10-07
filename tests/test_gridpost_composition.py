"""The unpooled and partially pooled distribution of D split by track length (gridpost.composition)."""
import importlib.util
import unittest

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions, analyze_tracks, by_track_length, deconvolve_tracks

ACQ = Acquisition(dt_s=.02, exposure_s=.02)
OPT = GridPostOptions(D_min_um2_s=1e-4, D_max_um2_s=10., n_D=101)
U = OPT.u_D()


def two_populations(rng, n_fast=300, n_slow=150):
    """Fast tracks (D 0.3) are short, slow ones (D 0.003) long: the GEM pattern."""
    rows = []
    for i in range(n_fast + n_slow):
        fast = i < n_fast
        n = int(rng.integers(3, 7)) if fast else int(rng.integers(20, 40))
        D = .3 if fast else .003
        sd = rng.uniform(.015, .025, (n, 2))
        pos = np.cumsum(rng.normal(0, np.sqrt(2 * D * ACQ.dt_s), (n, 2)), 0) + rng.normal(0, sd)
        rows.append(pl.DataFrame({"track_id": i, "frame": np.arange(n), "x_um": pos[:, 0], "y_um": pos[:, 1],
                                  "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1]}))
    return pl.concat(rows)


class ByTrackLengthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = analyze_tracks(two_populations(np.random.default_rng(0)), ACQ, OPT, keep_likelihoods=True)
        cls.population = deconvolve_tracks(cls.analysis, n_samples=200)

    def test_unpooled_groups_stack_to_the_mean_flat_prior_posterior(self):
        post = self.analysis.likelihoods
        comp = by_track_length(post, U)
        self.assertEqual(comp.n_tracks.sum(), len(post.track_ids))
        self.assertEqual(comp.n_detections.sum(), post.n_frames.sum())
        self.assertIsNone(comp.partially_pooled)
        p = np.exp(post.loglik_D)
        np.testing.assert_allclose(comp.unpooled.sum(0), p.mean(0) / p.mean(0).sum(), atol=1e-12)
        np.testing.assert_allclose(comp.unpooled.sum(1), comp.share(), atol=1e-12)
        self.assertEqual(comp.labels(), ["3", "4", "5-6", "7-9", "10-14", "15-24", "25+"])  # longest is 39
        self.assertEqual(by_track_length(post, U, edges=(3, 50)).n_tracks.tolist(), [len(post.track_ids), 0])

    def test_detections_weight_each_track_by_its_frames(self):
        post = self.analysis.likelihoods
        comp = by_track_length(post, U, weight="detections")
        n = post.n_frames
        np.testing.assert_allclose(comp.share(), comp.n_detections / n.sum())
        p = np.exp(post.loglik_D)
        np.testing.assert_allclose(comp.unpooled.sum(0), n @ p / n.sum(), atol=1e-12)

    def test_partially_pooled_groups_stack_to_about_the_population(self):
        comp = by_track_length(self.analysis.likelihoods, U, self.population, n_draws=50)
        self.assertEqual(comp.partially_pooled.shape, (50, 7, len(U)))
        np.testing.assert_allclose(comp.partially_pooled.sum(axis=(1, 2)), 1., atol=1e-10)
        total = comp.partially_pooled.sum(1).mean(0)
        self.assertLess(.5 * np.abs(total - self.population.samples.mean(0)).sum(), .05)
        lo, hi = comp.band(.9)
        self.assertTrue(np.all(lo <= hi))

    def test_short_tracks_sit_at_the_fast_mode_and_long_ones_at_the_slow(self):
        comp = by_track_length(self.analysis.likelihoods, U, self.population, edges=(3, 20))
        mean = comp.partially_pooled.mean(0)
        split = np.log(np.sqrt(.3 * .003))
        short, long = mean / mean.sum(1, keepdims=True)
        self.assertGreater(short[U > split].sum(), .95)
        self.assertGreater(long[U < split].sum(), .95)

    def test_bad_inputs_raise(self):
        post = self.analysis.likelihoods
        for kw in ({"edges": (5, 10)}, {"edges": (3, 3, 10)}, {"edges": ()}, {"weight": "frames"}):
            with self.assertRaises(ValueError):
                by_track_length(post, U, **kw)
        with self.assertRaises(ValueError):
            by_track_length(post, U[:-1])
        with self.assertRaises(ValueError):
            by_track_length(post, U, self.population.__class__(**{**self.population.__dict__, "u": U + 1.}))
        with self.assertRaises(ValueError):
            by_track_length(post, U).band()

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "needs the plots extra")
    def test_plot_draws_both_columns(self):
        from diffusionkit.gridpost.viz import plot_by_track_length
        comp = by_track_length(self.analysis.likelihoods, U, self.population, n_draws=20)
        floor = self.analysis.fits["D_floor_um2_s"].drop_nulls().to_numpy()
        fig = plot_by_track_length(comp, floor)
        self.assertEqual(len(fig.axes), 4)


if __name__ == "__main__":
    unittest.main()
