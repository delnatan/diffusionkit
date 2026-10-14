"""The population of D split by track length (gridpost.composition)."""
import importlib.util
import unittest

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions, analyze_tracks, by_track_length

ACQ = Acquisition(dt_s=.02, exposure_s=.02)
OPT = GridPostOptions(D_min_um2_s=1e-4, D_max_um2_s=10., n_D=101)
U = OPT.u_D()
D_FAST, D_SLOW = .3, .003
# (lengths, fraction fast) per group: the fast share falls with track length, the GEM pattern.
GROUPS = (((3, 5), .8), ((5, 10), .5), ((10, 20), .25), ((20, 40), .1))


def mixed_populations(rng, n_per_group=200):
    """Fast (D 0.3) and slow (D 0.003) tracks in every length group, mixed as `GROUPS` says.
    Returns the tracks and each group's realized fraction fast."""
    rows, frac, i = [], [], 0
    for (lo, hi), p_fast in GROUPS:
        fast = rng.random(n_per_group) < p_fast
        frac.append(fast.mean())
        for f in fast:
            n = int(rng.integers(lo, hi))
            sd = rng.uniform(.015, .025, (n, 2))
            step = np.sqrt(2 * (D_FAST if f else D_SLOW) * ACQ.dt_s)
            pos = np.cumsum(rng.normal(0, step, (n, 2)), 0) + rng.normal(0, sd)
            rows.append(pl.DataFrame({"track_id": i, "frame": np.arange(n), "x_um": pos[:, 0], "y_um": pos[:, 1],
                                      "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1]}))
            i += 1
    return pl.concat(rows), np.array(frac)


class ByTrackLengthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tracks, cls.frac_fast = mixed_populations(np.random.default_rng(0))
        cls.analysis = analyze_tracks(tracks, ACQ, OPT, keep_likelihoods=True)
        cls.comp = by_track_length(cls.analysis.likelihoods, U)

    def test_groups_stack_to_one_by_share(self):
        post, comp = self.analysis.likelihoods, self.comp
        self.assertEqual(comp.labels(), ["3-4", "5-9", "10-19", "20+"])
        self.assertEqual(comp.n_tracks.sum(), len(post.track_ids))
        self.assertEqual(comp.n_detections.sum(), post.n_frames.sum())
        np.testing.assert_allclose(comp.deconvolved.sum(1), comp.share(), atol=1e-12)
        self.assertAlmostEqual(comp.deconvolved.sum(), 1., places=12)
        lo, hi = comp.band(.9)
        self.assertTrue(np.all(lo <= hi) and lo.shape == hi.shape == comp.deconvolved.shape)

    def test_each_group_recovers_its_fast_fraction(self):
        split = np.sqrt(D_FAST * D_SLOW)
        for pop, truth in zip(self.comp.populations, self.frac_fast):
            self.assertAlmostEqual(float(pop.weights[U >= np.log(split)].sum()), truth, delta=.08)
            lo, hi = np.quantile(pop.mass(lo=split), [.005, .995])
            self.assertTrue(lo <= truth <= hi, (lo, truth, hi))

    def test_detections_change_only_the_shares(self):
        post = self.analysis.likelihoods
        comp = by_track_length(post, U, weight="detections", n_samples=200)
        np.testing.assert_allclose(comp.share(), comp.n_detections / post.n_frames.sum())
        for j, pop in enumerate(comp.populations):
            np.testing.assert_allclose(comp.deconvolved[j], comp.share()[j] * pop.weights)
            np.testing.assert_allclose(pop.weights, self.comp.populations[j].weights)
        switched = self.comp.with_weight("detections")
        np.testing.assert_allclose(switched.deconvolved, comp.deconvolved)
        self.assertIs(switched.populations, self.comp.populations)
        with self.assertRaises(ValueError):
            self.comp.with_weight("frames")

    def test_an_empty_group_has_no_population(self):
        comp = by_track_length(self.analysis.likelihoods, U, edges=(3, 50), n_samples=200)
        self.assertEqual(comp.n_tracks.tolist(), [len(self.analysis.likelihoods.track_ids), 0])
        self.assertIsNone(comp.populations[1])
        self.assertFalse(comp.deconvolved[1].any())

    def test_bad_inputs_raise(self):
        post = self.analysis.likelihoods
        for kw in ({"edges": (5, 10)}, {"edges": (3, 3, 10)}, {"edges": ()}, {"weight": "frames"}):
            with self.assertRaises(ValueError):
                by_track_length(post, U, **kw)
        with self.assertRaises(ValueError):
            by_track_length(post, U[:-1])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "needs the plots extra")
    def test_plot_draws_one_column(self):
        from diffusionkit.gridpost.viz import plot_by_track_length
        floor = self.analysis.fits["D_floor_um2_s"].drop_nulls().to_numpy()
        fig = plot_by_track_length(self.comp, floor)
        self.assertEqual(len(fig.axes), 2)


if __name__ == "__main__":
    unittest.main()
