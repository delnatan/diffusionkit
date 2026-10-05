"""Drift field estimation, subtraction and the shared-motion diagnostic."""
import unittest

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.drift import PolyBasis, estimate_drift, neighbour_correlation, subtract

ACQ = Acquisition(dt_s=.02, exposure_s=.02)
N_FRAMES = 30


def path(t):
    """A rigid path (um), 0 at t = 0: ~0.3 um in x over the movie, a wiggle in y."""
    return np.stack([-.01 * t, .02 * np.sin(t / 5.)], 1)


def simulate(rng, n_tracks=600, drift=path, field=None, first_frame=0, D_mobile=.3, flow=None):
    rows = []
    for i in range(n_tracks):
        still = rng.random() < .6
        n = N_FRAMES if still and rng.random() < .3 else int(rng.integers(3, 15))
        f = first_frame + int(rng.integers(0, N_FRAMES - n + 1)) + np.arange(n)
        D = 1e-4 if still else D_mobile
        start = rng.uniform(0, 40, 2)
        pos = start + np.vstack([np.zeros(2), np.cumsum(rng.normal(0, np.sqrt(2 * D * ACQ.dt_s), (n - 1, 2)), 0)])
        t = (f - first_frame).astype(float)
        pos = pos + drift(t) - drift(t[:1])
        if flow is not None:
            pos = pos + flow(start, t) - flow(start, t[:1])
        sd = rng.uniform(.012, .025, (n, 2))
        pos = pos + rng.normal(0, sd)
        rows.append(pl.DataFrame({"track_id": i, "frame": f, "x_um": pos[:, 0], "y_um": pos[:, 1],
                                  "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1]}))
    return pl.concat(rows)


class DriftTests(unittest.TestCase):
    def test_rigid_path_recovered_to_nanometres(self):
        rng = np.random.default_rng(1)
        tracks = simulate(rng)
        dr = estimate_drift(tracks, ACQ, degree=0)
        self.assertTrue(dr.converged)
        est = dr.at((20., 20.))
        err = est - path(np.arange(N_FRAMES, dtype=float))
        self.assertLess(np.sqrt(np.mean(err ** 2)), .004)
        # the reported SE is the right order: the error at the last frame is within a few SE
        se = np.sqrt(dr.path_cov((20., 20.))[:, -1, -1])
        self.assertTrue(np.all(np.abs(err[-1]) < 4 * se))

    def test_subtract_removes_the_path(self):
        rng = np.random.default_rng(2)
        tracks = simulate(rng)
        dr = estimate_drift(tracks, ACQ, degree=2)
        out = subtract(tracks, dr)
        self.assertEqual(out.columns, tracks.columns)
        self.assertEqual(out.height, tracks.height)
        whole = out.filter(pl.len().over("track_id") == N_FRAMES).sort("track_id", "frame")
        before = tracks.filter(pl.col("track_id").is_in(whole["track_id"].unique().implode())).sort("track_id", "frame")
        net = lambda t: (t.group_by("track_id").agg(pl.col("x_um").last() - pl.col("x_um").first())["x_um"].to_numpy())
        self.assertLess(np.abs(np.median(net(whole))), .01)
        self.assertGreater(np.abs(np.median(net(before))), .25)

    def test_no_drift_estimates_none(self):
        rng = np.random.default_rng(3)
        tracks = simulate(rng, drift=lambda t: np.zeros((len(t), 2)))
        dr = estimate_drift(tracks, ACQ, degree=2)
        se = np.sqrt(np.stack([np.diag(c) for c in dr.path_cov((20., 20.))], 1))[1:]
        z = dr.at((20., 20.))[1:] / se
        self.assertLess(np.max(np.abs(z)), 4.)
        self.assertLess(np.sqrt(np.mean(z ** 2)), 2.)

    def test_frames_need_not_start_at_zero(self):
        rng = np.random.default_rng(4)
        tracks = simulate(rng, first_frame=100)
        dr = estimate_drift(tracks, ACQ, degree=0)
        self.assertEqual(dr.frames[0], 100)
        self.assertLess(np.sqrt(np.mean((dr.at((0., 0.)) - path(np.arange(N_FRAMES, dtype=float))) ** 2)), .004)
        with self.assertRaises(ValueError):
            subtract(tracks.with_columns(pl.col("frame") - 1), dr)

    def test_gaps_raise(self):
        rng = np.random.default_rng(5)
        tracks = simulate(rng, n_tracks=50)
        gap = tracks.filter(~((pl.col("track_id") == tracks["track_id"][0]) & (pl.col("frame") == tracks["frame"][1])))
        if gap.filter(pl.col("track_id") == tracks["track_id"][0]).height >= 3:
            with self.assertRaises(ValueError):
                estimate_drift(gap, ACQ, degree=0)

    def test_poly_basis(self):
        b = PolyBasis(2, (0., 0.), (10., 10.))
        self.assertEqual(b.size, 6)
        self.assertEqual(b((5., 5.)).shape, (6,))
        self.assertEqual(b(np.zeros((4, 3, 2))).shape, (4, 3, 6))
        self.assertAlmostEqual(b((5., 5.))[0], 1.)


class NeighbourCorrelationTests(unittest.TestCase):
    def test_shared_local_motion_is_detected_and_own_motion_is_not(self):
        rng = np.random.default_rng(6)
        none = lambda t: np.zeros((len(t), 2))
        # patches 10 um across sliding past each other: 60 nm over the movie
        flow = lambda x, t: (.06 * np.sign(np.sin(2 * np.pi * x[0] / 20.)) * t / N_FRAMES)[:, None] * np.array([0., 1.])
        quiet = simulate(rng, n_tracks=1500, drift=none, D_mobile=.3)
        moving = simulate(np.random.default_rng(6), n_tracks=1500, drift=none, D_mobile=.3, flow=flow)
        kw = dict(radii_um=(4.,), min_frames=N_FRAMES, n_null=20)
        q = neighbour_correlation(quiet, **kw).row(0, named=True)
        m = neighbour_correlation(moving, **kw).row(0, named=True)
        self.assertLess(abs(q["corr"] - q["null_mean"]), 4 * q["null_sd"] + .05)
        self.assertGreater(m["corr"] - m["null_mean"], 4 * m["null_sd"])

    def test_too_few_tracks_gives_an_empty_table(self):
        rng = np.random.default_rng(7)
        out = neighbour_correlation(simulate(rng, n_tracks=5))
        self.assertEqual(out.height, 0)
        self.assertIn("corr", out.columns)


if __name__ == "__main__":
    unittest.main()
