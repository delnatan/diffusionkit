"""The table-to-arrays layer: which tracks the vectorized paths take, and that they agree with the per-track path."""
import unittest

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.tracks import TrackArrays, by_length, chunks


def table(n, track_id, seed=0, first_frame=0):
    rng = np.random.default_rng(seed + track_id)
    return pl.DataFrame({"track_id": [track_id] * n, "frame": np.arange(n) + first_frame,
                         "x_um": np.cumsum(rng.normal(0, .05, n)), "y_um": np.cumsum(rng.normal(0, .05, n)),
                         "sigma_x_um": rng.uniform(.01, .03, n), "sigma_y_um": rng.uniform(.01, .03, n)})


class TrackArraysTests(unittest.TestCase):
    def test_flags_match_the_per_track_checks(self):
        good = table(6, 1)
        gap = table(6, 2).filter(pl.col("frame") != 3)
        nan = table(6, 3).with_columns(pl.when(pl.col("frame") == 2).then(None).otherwise(pl.col("x_um")).alias("x_um"))
        zero = table(6, 4).with_columns(pl.lit(0.).alias("sigma_y_um"))
        short = table(2, 5)
        negative = table(5, 6, first_frame=-2)
        arrays = TrackArrays.from_table(pl.concat([zero, good, nan, gap, short, negative]).sample(fraction=1., seed=2,
                                                                                                    shuffle=True),
                                        Acquisition(.03))
        self.assertEqual(arrays.track_id.tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(arrays.valid.tolist(), [True, False, False, True, True, False])
        self.assertEqual(arrays.clean.tolist(), [True, False, False, False, False, False])
        np.testing.assert_array_equal(arrays.stack(np.array([0]))[0][0], good.select("x_um", "y_um").to_numpy())
        self.assertTrue(arrays.frame(0).equals(good))

    def test_time_and_length_columns_are_checked(self):
        t = table(5, 1).with_columns((pl.col("frame") * .03).alias("t_s"), pl.lit(5).alias("track_length"))
        self.assertTrue(TrackArrays.from_table(t, Acquisition(.03)).clean[0])
        self.assertFalse(TrackArrays.from_table(t, Acquisition(.04)).valid[0])
        self.assertFalse(TrackArrays.from_table(t.with_columns(pl.lit(4).alias("track_length")), Acquisition(.03)).valid[0])

    def test_empty_table(self):
        arrays = TrackArrays.from_table(table(3, 1).head(0), Acquisition(.03))
        self.assertEqual(len(arrays), 0)

    def test_chunks_and_lengths(self):
        self.assertEqual([list(c) for c in chunks(5, 2)], [[0, 1], [2, 3], [4]])
        groups = by_length(range(5), np.array([3, 4, 3, 5, 4]))
        self.assertEqual([g.tolist() for g in groups], [[0, 2], [1, 4], [3]])


class BatchedWorkflowTests(unittest.TestCase):
    def test_gridpost_table_matches_per_track_calls(self):
        from diffusionkit.gridpost import analyze_track, analyze_tracks
        tracks = pl.concat([table(n, i) for i, n in enumerate((5, 9, 5, 14, 9, 3))])
        acq = Acquisition(.03, .01)
        r = analyze_tracks(tracks, acq, keep_likelihoods=True)
        for i, g in enumerate(tracks.partition_by("track_id", maintain_order=True)):
            direct = analyze_track(g, acq)
            row = r.fits.row(i, named=True)
            for name, value in direct.posterior_D.parameters.items():
                self.assertAlmostEqual(row[name], value, delta=1e-12 * abs(value))
            np.testing.assert_allclose(r.likelihoods.loglik_D[i], direct.loglik_D, rtol=0, atol=1e-9)

    def test_fbmpost_table_matches_per_track_calls(self):
        from diffusionkit.fbmpost import FbmPostOptions, analyze_track, analyze_tracks
        tracks = pl.concat([table(n, i) for i, n in enumerate((6, 10, 6, 10))])
        acq, options = Acquisition(.03, .01), FbmPostOptions(n_alpha=21, n_snr=81)
        r = analyze_tracks(tracks, acq, options, keep_posteriors=True)
        for i, g in enumerate(tracks.partition_by("track_id", maintain_order=True)):
            direct = analyze_track(g, acq, options)
            row = r.fits.row(i, named=True)
            for name, value in direct.posterior.parameters.items():
                self.assertAlmostEqual(row[name], value, delta=1e-9 * max(abs(value), 1e-6))
            np.testing.assert_allclose(r.posteriors.log_post_alpha[i], direct.log_post_alpha, rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
