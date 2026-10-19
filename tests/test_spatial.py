"""Spatial classes on a simulated cell: per-track Gaussian likelihoods in ln D, a slow class inside a disk."""
import unittest

import numpy as np
import polars as pl

from diffusionkit.gridpost import GridLikelihoods
import json

from diffusionkit.spatial import SpatialOptions, analyze_likelihoods, centroids, potts, restore

U = np.linspace(np.log(1e-4), np.log(10.), 201)
D_SLOW, D_FAST = .05, .4


def cell(rng, n=600, patterned=True, sd=.9):
    """Tracks uniform in a 30 um square; inside a 7 um disk 80% are slow, outside 5%. Each track's likelihood is
    a normal in ln D of width `sd` around a noisy reading of its true ln D."""
    xy = rng.uniform(0, 30, (n, 2))
    inside = np.linalg.norm(xy - 15, axis=1) < 7
    slow = rng.random(n) < np.where(inside, .8, .05)
    if not patterned:
        xy = xy[rng.permutation(n)]
    true = np.log(np.where(slow, D_SLOW, D_FAST)) + .15 * rng.standard_normal(n)
    obs = true + sd * rng.standard_normal(n)
    ll = -.5 * ((U[None, :] - obs[:, None]) / sd) ** 2
    ids = np.arange(100, 100 + n)
    lik = GridLikelihoods(ids, np.full(n, 5), ll - ll.max(axis=1, keepdims=True))
    pos = pl.DataFrame({"track_id": ids, "x_um": xy[:, 0], "y_um": xy[:, 1]})
    return lik, pos, slow


OPTS = SpatialOptions(max_classes=3, n_shuffles=19)


class PatternedCellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lik, cls.pos, cls.slow = cell(np.random.default_rng(3))
        cls.s = analyze_likelihoods(cls.lik, U, cls.pos, OPTS)

    def test_two_classes_suggested_and_found_in_order_of_D(self):
        self.assertEqual(self.s.n_classes, 2)
        np.testing.assert_allclose(np.exp(self.s.fit.centre), [D_SLOW, D_FAST], rtol=.5)
        self.assertTrue(self.s.selection.filter(pl.col("n_classes") == 2)["suggested"].item())

    def test_neighbour_gain_beats_the_shuffles(self):
        self.assertTrue(self.s.patterned)
        self.assertGreater(self.s.null.gain, self.s.null.shuffled_gain.max())
        self.assertAlmostEqual(self.s.null.p, 1 / 20)

    def test_space_classifies_better_than_the_tracks_alone(self):
        alone = potts.fit(np.exp(self.lik.loglik_D), U, self.s.neighbours, 2, beta=0.)
        acc = lambda q: np.mean((q.argmax(axis=1) == 0) == self.slow)
        self.assertGreater(acc(self.s.fit.q), acc(alone.q) + .01)
        self.assertGreater(acc(self.s.fit.q), .9)

    def test_neighbour_groups_partition_the_tracks(self):
        groups = self.s.groups()
        self.assertEqual(sorted(groups), [1, 2])
        ids = np.concatenate(list(groups.values()))
        self.assertEqual(sorted(ids.tolist()), sorted(self.lik.track_ids.tolist()))
        rep = self.s.report()
        self.assertEqual(rep["n_tracks"].sum(), len(ids))
        self.assertEqual(rep["n_detections"].sum(), 5 * len(ids))

    def test_columns(self):
        cols = self.s.columns()
        self.assertEqual(cols.height, len(self.lik.track_ids))
        self.assertTrue(cols["spatial_class"].is_in([1, 2]).all())
        self.assertTrue(((cols["spatial_p"] >= .5) & (cols["spatial_p"] <= 1)).all())
        self.assertTrue((cols["D_spatial_lo_um2_s"] <= cols["D_spatial_mean_um2_s"]).all())
        self.assertTrue((cols["D_spatial_mean_um2_s"] <= cols["D_spatial_hi_um2_s"]).all())

    def test_field_inside_the_disk_is_class_1(self):
        p, dist = self.s.field(np.array([[15., 15.], [2., 2.]]))
        self.assertGreater(p[0, 0], .5)
        self.assertLess(p[1, 0], .5)
        self.assertTrue((dist > 0).all())

    def test_record_and_probabilities_restore_the_run(self):
        record = json.loads(json.dumps(self.s.record()))
        back = restore(self.lik, U, record, self.s.probabilities())
        self.assertEqual(back.n_classes, 2)
        self.assertEqual(back.patterned, self.s.patterned)
        self.assertTrue(back.columns().equals(self.s.columns()))
        self.assertTrue(back.selection.equals(self.s.selection))
        np.testing.assert_allclose(back.field(np.array([[15., 15.]]))[0], self.s.field(np.array([[15., 15.]]))[0])


class UnpatternedCellTests(unittest.TestCase):
    def test_no_pattern_without_one(self):
        lik, pos, _ = cell(np.random.default_rng(4), patterned=False)
        s = analyze_likelihoods(lik, U, pos, SpatialOptions(max_classes=2, n_classes=2, n_shuffles=19))
        self.assertFalse(s.patterned)
        self.assertGreater(s.null.p, .05)


class InputTests(unittest.TestCase):
    def test_tracks_without_a_position_are_left_out(self):
        lik, pos, _ = cell(np.random.default_rng(5), n=120)
        s = analyze_likelihoods(lik, U, pos.head(100), SpatialOptions(max_classes=1, n_shuffles=0))
        self.assertEqual(len(s.track_ids), 100)
        self.assertIsNone(s.null)
        self.assertFalse(s.patterned)

    def test_centroids(self):
        t = pl.DataFrame({"track_id": [1, 1, 2], "x_um": [0., 2., 5.], "y_um": [1., 3., 7.]})
        np.testing.assert_allclose(centroids(t).select("x_um", "y_um").to_numpy(), [[1, 2], [5, 7]])

    def test_knn_never_lists_a_track_as_its_own_neighbour(self):
        xy = np.zeros((6, 2))  # coincident positions
        nb = potts.knn(xy, 3)
        self.assertFalse((nb == np.arange(6)[:, None]).any())

    def test_option_checks(self):
        with self.assertRaises(ValueError):
            SpatialOptions(n_classes=5, max_classes=4)
        with self.assertRaises(ValueError):
            potts.knn(np.zeros((3, 2)), 8)


if __name__ == "__main__":
    unittest.main()
