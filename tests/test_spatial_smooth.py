"""The smooth map of ln D on simulated cells: per-track Gaussian likelihoods in ln D, slow inside a disk."""
import unittest

import numpy as np
import polars as pl

from diffusionkit.gridpost import GridLikelihoods
from diffusionkit.spatial import SmoothMapOptions, smooth, smooth_map_likelihoods

U = np.linspace(np.log(1e-4), np.log(10.), 201)
D_SLOW, D_FAST = .05, .2


def cell(rng, n=400, patterned=True, sd=.9):
    """Tracks uniform in a 20 um square; D_SLOW inside a 6 um disk, D_FAST outside. Each track's likelihood is
    a normal in ln D of width `sd` around a noisy reading of its true ln D."""
    xy = rng.uniform(0, 20, (n, 2))
    inside = np.linalg.norm(xy - 10, axis=1) < 6
    true = np.log(np.where(inside, D_SLOW, D_FAST))
    if not patterned:
        xy = xy[rng.permutation(n)]
    obs = true + sd * rng.standard_normal(n)
    ll = -.5 * ((U[None, :] - obs[:, None]) / sd) ** 2
    ids = np.arange(100, 100 + n)
    lik = GridLikelihoods(ids, np.full(n, 5), ll - ll.max(axis=1, keepdims=True))
    pos = pl.DataFrame({"track_id": ids, "x_um": xy[:, 0], "y_um": xy[:, 1]})
    return lik, pos, inside, obs


OPTS = SmoothMapOptions(n_shuffles=19)
CUT = np.log(np.sqrt(D_SLOW * D_FAST))


class LaplaceEvidenceTests(unittest.TestCase):
    """With Gaussian rows the posterior is Gaussian, so the Laplace evidence is exact: compare it with the
    closed form log N(y; 0, C + sd^2 I) + n/2 log(2 pi sd^2), C the prior covariance of ln D."""

    def test_matches_the_closed_form(self):
        rng = np.random.default_rng(1)
        lik, pos, _, obs = cell(rng, n=120, sd=.9)
        xy = pos.select("x_um", "y_um").to_numpy()
        n, sd, mean_sd = len(obs), .9, 10.
        L = smooth.laplacian(xy, 8)
        spatial, plain = smooth._models(lik.loglik_D, U, L, mean_sd)
        # The rows were shifted to peak at 0, which drops -1/2 log(2 pi sd^2) per track from the closed form.
        shift = .5 * n * np.log(2 * np.pi * sd ** 2)

        def exact(C):
            S = C + sd ** 2 * np.eye(n)
            _, logdet = np.linalg.slogdet(S)
            return -.5 * (obs @ np.linalg.solve(S, obs) + logdet + n * np.log(2 * np.pi)) + shift

        x0 = np.zeros(n)
        for ls, lv in [(0., 1.), (2., -1.), (-1., 6.)]:
            f = spatial.fit(ls, lv, x0, x0.copy(), variances=True)
            K = np.exp(ls) * L.toarray() + np.eye(n) / (n * mean_sd ** 2)
            C = np.linalg.inv(K) + np.eye(n) / np.exp(lv)
            self.assertAlmostEqual(f.evidence, exact(C), delta=1e-3 * abs(exact(C)))
            post = np.linalg.inv(np.linalg.inv(C) + np.eye(n) / sd ** 2)
            np.testing.assert_allclose(f.sd, np.sqrt(np.diag(post)), rtol=1e-3)
            np.testing.assert_allclose(f.x, post @ obs / sd ** 2, atol=1e-3)
        f = plain.fit(0., 1., x0, np.zeros(1))
        C = mean_sd ** 2 * np.ones((n, n)) + np.eye(n) / np.e
        self.assertAlmostEqual(f.evidence, exact(C), delta=1e-3 * abs(exact(C)))


class PatternedCellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lik, cls.pos, cls.inside, cls.obs = cell(np.random.default_rng(3))
        cls.m = smooth_map_likelihoods(cls.lik, U, cls.pos, OPTS)

    def test_map_beats_no_space_and_every_shuffle(self):
        self.assertGreater(self.m.gain, 20)
        self.assertGreater(self.m.gain, self.m.null.shuffled_gain.max())
        self.assertAlmostEqual(self.m.null.p, 1 / 20)
        self.assertTrue(self.m.patterned)

    def test_map_classifies_better_than_the_tracks_alone(self):
        acc = lambda x: np.mean((x < CUT) == self.inside)
        self.assertGreater(acc(self.m.ln_D), .9)
        self.assertGreater(acc(self.m.ln_D), acc(self.obs) + .1)

    def test_columns(self):
        cols = self.m.columns()
        self.assertEqual(cols.height, len(self.lik.track_ids))
        self.assertEqual(cols["track_id"].to_list(), self.lik.track_ids.tolist())
        self.assertTrue((cols["D_map_lo_um2_s"] < cols["D_map_um2_s"]).all())
        self.assertTrue((cols["D_map_um2_s"] < cols["D_map_hi_um2_s"]).all())
        self.assertTrue((cols["ln_D_map_sd"] < .9).all())  # neighbours narrow every track's own reading
        self.assertEqual(self.m.record()["n_tracks"], cols.height)


class NoPatternTests(unittest.TestCase):
    def test_shuffled_positions_show_no_map(self):
        lik, pos, _, _ = cell(np.random.default_rng(4), patterned=False)
        m = smooth_map_likelihoods(lik, U, pos, OPTS)
        self.assertLess(abs(m.gain), 3)
        self.assertGreater(m.null.p, .05)
        self.assertFalse(m.patterned)


class InputTests(unittest.TestCase):
    def test_tracks_without_a_position_are_left_out(self):
        lik, pos, _, _ = cell(np.random.default_rng(5), n=60)
        m = smooth_map_likelihoods(lik, U, pos.head(50), SmoothMapOptions(n_shuffles=0))
        self.assertEqual(len(m.track_ids), 50)
        self.assertIsNone(m.null)

    def test_bad_options(self):
        with self.assertRaises(ValueError):
            SmoothMapOptions(mean_sd=0.)
        with self.assertRaises(ValueError):
            SmoothMapOptions(n_shuffles=-1)


if __name__ == "__main__":
    unittest.main()
