"""Spatial classes on simulated cells: per-track Gaussian likelihoods in ln D (slow inside a disk, or three regions),
and one movie of simulated tracks."""
import unittest

import json

import numpy as np
import polars as pl
from scipy.sparse import triu as sparse_triu

from diffusionkit.gridpost import GridLikelihoods
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
        alone = potts.fit(np.exp(self.lik.loglik_D), U, self.s.adjacency, 2, beta=0.)
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


    def test_beta_is_identified_and_below_the_frozen_phase(self):
        # beta per edge matches the prior's agreement to the fit's: inside the table, never at its top.
        self.assertGreater(self.s.fit.beta, 0.)
        self.assertLess(self.s.fit.beta, potts.BETA_GRID[-1])
        self.assertLess(self.s.fit.agreement, 1.)

    def test_mean_field_agreement_matches_gibbs(self):
        # beta's M-step reads the posterior's edge agreement off the mean field. Gibbs sampling of the same
        # posterior (the fit's classes and beta) gives the agreement without that approximation: the beta it
        # implies should be the fit's.
        f, A = self.s.fit, self.s.adjacency
        rng = np.random.default_rng(0)
        log_ev = np.log(f.evidence + 1e-300)
        z = f.q.argmax(axis=1)
        e = sparse_triu(A, 1).tocoo()
        agree = []
        for sweep in range(60):
            for i in rng.permutation(len(z)):
                nb = A.indices[A.indptr[i]:A.indptr[i + 1]]
                lp = log_ev[i] + f.beta * np.bincount(z[nb], minlength=f.n_classes)
                pr = np.exp(lp - lp.max())
                z[i] = rng.choice(f.n_classes, p=pr / pr.sum())
            if sweep >= 20:
                agree.append(np.mean(z[e.row] == z[e.col]))
        table = potts.agreement_table(A, f.n_classes)
        self.assertAlmostEqual(potts.beta_for(float(np.mean(agree)), table), f.beta, delta=.03)

    def test_a_record_from_the_old_model_is_refused(self):
        record = json.loads(json.dumps(self.s.record()))
        record.pop("model")
        with self.assertRaisesRegex(ValueError, "older model"):
            restore(self.lik, U, record, self.s.probabilities())


def regions3(rng, n=900, sd=.6, patterned=True):
    """Three regions: D 0.03 in one disk, 0.1 in another, 0.35 elsewhere, in a 30 um square."""
    xy = rng.uniform(0, 30, (n, 2))
    region = np.full(n, 2)
    region[np.linalg.norm(xy - 9, axis=1) < 6.5] = 0
    region[np.linalg.norm(xy - 21, axis=1) < 6.5] = 1
    if not patterned:
        xy = xy[rng.permutation(n)]
    obs = np.log(np.array([.03, .1, .35]))[region] + sd * rng.standard_normal(n)
    ll = -.5 * ((U[None, :] - obs[:, None]) / sd) ** 2
    ids = np.arange(n)
    lik = GridLikelihoods(ids, np.full(n, 5), ll - ll.max(axis=1, keepdims=True))
    return lik, pl.DataFrame({"track_id": ids, "x_um": xy[:, 0], "y_um": xy[:, 1]}), region


class ThreeRegionTests(unittest.TestCase):
    def test_three_classes_found_where_they_are(self):
        # The plain mixture's equal weights split the large fast region and merge the two small ones; the split
        # starts from the two-class fit find all three.
        lik, pos, region = regions3(np.random.default_rng(8))
        s = analyze_likelihoods(lik, U, pos, SpatialOptions(max_classes=4, n_shuffles=19))
        self.assertEqual(s.n_classes, 3)
        self.assertTrue(s.patterned)
        np.testing.assert_allclose(np.exp(s.fit.centre), [.03, .1, .35], rtol=.35)
        self.assertGreater(np.mean(s.fit.q.argmax(axis=1) == region), .9)


class AgreementTableTests(unittest.TestCase):
    def test_from_unrelated_labels_to_one_class(self):
        xy = np.random.default_rng(0).uniform(0, 20, (300, 2))
        A = potts.graph(xy, 8)
        for c in (2, 3):
            t = potts.agreement_table(A, c)
            self.assertAlmostEqual(t[0], 1 / c, delta=.03)
            self.assertGreater(t[-1], .97)
            self.assertTrue((np.diff(t) >= 0).all())
            self.assertEqual(potts.beta_for(t[0] - .01, t), 0.)
            mid = .5 * (t[0] + t[-1])
            b = potts.beta_for(mid, t)
            self.assertAlmostEqual(np.interp(b, potts.BETA_GRID, t), mid, delta=1e-6)

    def test_graph_is_symmetric_and_loop_free(self):
        A = potts.graph(np.random.default_rng(1).uniform(0, 10, (50, 2)), 4)
        self.assertEqual((A != A.T).nnz, 0)
        self.assertEqual(A.diagonal().sum(), 0)
        self.assertTrue((np.asarray(A.sum(axis=1)).ravel() >= 4).all())


class BetaRunawayTests(unittest.TestCase):
    """The movie where the old fit (class weights, beta by a surrogate score) ran beta to its cap with one class
    taking every track, and so picked 3 classes for a two-region cell: 800 simulated tracks of 3-8 frames, D 0.05
    inside a 6 um disk and 0.2 outside."""

    def test_two_classes_and_a_finite_beta(self):
        from diffusionkit import Acquisition
        from diffusionkit.fbmpost.simulate import simulate_tracks
        from diffusionkit.gridpost import analyze_tracks
        acq = Acquisition(dt_s=0.035, exposure_s=0.020)
        rng = np.random.default_rng(6)
        xy = rng.uniform(0, 20, (800, 2))
        D = np.where(np.hypot(*(xy - 10).T) < 6, .05, .2)
        tracks = pl.concat([simulate_tracks(1, int(rng.integers(3, 9)), 1., D[i], .03, acq, rng=rng, n_sub=10,
                                            first_track_id=i) for i in range(800)])
        d = analyze_tracks(tracks, acq, keep_likelihoods=True)
        pos = pl.DataFrame({"track_id": np.arange(800), "x_um": xy[:, 0], "y_um": xy[:, 1]})
        s = analyze_likelihoods(d.likelihoods, d.options.u_D(), pos, SpatialOptions(max_classes=3, n_shuffles=0))
        self.assertEqual(s.n_classes, 2)
        self.assertLess(s.fit.beta, 1.)
        inside = np.hypot(*(s.xy_um - 10).T) < 6
        self.assertGreater(np.mean((s.fit.q.argmax(axis=1) == 0) == inside), .95)


class WeakDataTests(unittest.TestCase):
    """Tracks that carry little each (likelihood width 1.3 in ln D): EM has a solution near beta = 0 and an ordered
    one, and a start at beta = 0 stays in the first even though the pattern is real."""

    def test_the_ordered_start_finds_the_pattern(self):
        lik, pos, slow = cell(np.random.default_rng(3), sd=1.3)
        s = analyze_likelihoods(lik, U, pos, SpatialOptions(max_classes=3, n_shuffles=19))
        self.assertEqual(s.n_classes, 2)
        self.assertTrue(s.patterned)
        self.assertGreater(np.mean((s.fit.q.argmax(axis=1) == 0) == slow), .85)
        # Without it, from the plain mixture, beta stays near 0.
        L, A = np.exp(lik.loglik_D), s.adjacency
        cold = potts.fit(L, U, A, 2, potts.agreement_table(A, 2), init=potts.fit(L, U, A, 2, beta=0.))
        self.assertLess(cold.beta, .1)

    def test_and_no_pattern_without_one(self):
        lik, pos, _ = cell(np.random.default_rng(3), sd=1.3, patterned=False)
        s = analyze_likelihoods(lik, U, pos, SpatialOptions(max_classes=2, n_classes=2, n_shuffles=19))
        self.assertFalse(s.patterned)


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
