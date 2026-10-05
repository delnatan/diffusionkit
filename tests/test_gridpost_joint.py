"""The per-track joint posterior file (diffusionkit.gridpost.joint)."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions, analyze_tracks
from diffusionkit.gridpost import joint as J

OPTIONS = GridPostOptions(n_D=101, joint_D_bin=10, n_alpha=9, alpha_min=.1, alpha_max=1.7)


def tracks(lengths, rng):
    tables = []
    for i, n in enumerate(lengths):
        sd = rng.uniform(.02, .04, (n, 2))
        pos = np.cumsum(rng.normal(0, .08, (n, 2)), axis=0) + sd * rng.standard_normal((n, 2))
        tables.append(pl.DataFrame({"track_id": [i] * n, "frame": np.arange(n), "x_um": pos[:, 0],
                                    "y_um": pos[:, 1], "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1]}))
    return pl.concat(tables)


class JointFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        cls.acquisition = Acquisition(.03, .02)
        table = tracks([5, 12, 2, 30], rng)  # the 2-frame track is excluded
        cls.analysis = analyze_tracks(table, cls.acquisition, OPTIONS, keep_posteriors=True)
        cls.joint = J.joint_posteriors(cls.analysis, sample="wt")

    def test_collects_ok_tracks_with_their_lengths(self):
        self.assertEqual(self.joint.track_id.tolist(), [0, 1, 3])
        self.assertEqual(self.joint.n_frames.tolist(), [5, 12, 30])
        self.assertEqual(set(self.joint.sample), {"wt"})
        self.assertEqual(self.joint.log_post.shape, (3, 9, 10))
        np.testing.assert_array_equal(self.joint.u, OPTIONS.u_joint_D())
        with self.assertRaisesRegex(ValueError, "keep_posteriors"):
            J.joint_posteriors(analyze_tracks(tracks([5], np.random.default_rng(2)), self.acquisition, OPTIONS))

    def test_round_trip_within_half_a_quantum(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "joint.parquet"
            J.write_joint_posteriors(path, self.joint)
            back = J.read_joint_posteriors(path)
        self.assertEqual(back.options, OPTIONS)
        self.assertEqual(back.acquisition, self.acquisition)
        np.testing.assert_array_equal(back.track_id, self.joint.track_id)
        np.testing.assert_array_equal(back.n_frames, self.joint.n_frames)
        self.assertEqual(back.sample.tolist(), ["wt"] * 3)
        lp = self.joint.log_post
        kept = lp > lp.max(axis=(1, 2), keepdims=True) - 60
        np.testing.assert_allclose(back.log_post[kept], lp[kept], atol=1e-3)
        np.testing.assert_allclose(np.exp(back.log_post).sum(axis=(1, 2)), 1, atol=1e-12)

    def test_reading_a_foreign_file_fails_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "other.parquet"
            pl.DataFrame({"a": [1]}).write_parquet(path)
            with self.assertRaisesRegex(ValueError, "not a joint posterior file"):
                J.read_joint_posteriors(path)
            pl.DataFrame({"a": [1]}).write_parquet(path, metadata={J.METADATA_KEY: '{"format": "old/0"}'})
            with self.assertRaisesRegex(ValueError, "format"):
                J.read_joint_posteriors(path)

    def test_pooling_checks_cells_acquisition_and_labels(self):
        ko = replace(self.joint, sample=np.full(3, "ko", dtype=object),
                     acquisition=Acquisition(.03 * (1 + 1e-4), .02))
        pooled = J.pool_joint_posteriors([self.joint, ko])
        self.assertEqual(pooled.log_post.shape, (6, 9, 10))
        self.assertEqual(pooled.sample.tolist(), ["wt"] * 3 + ["ko"] * 3)
        with self.assertRaisesRegex(ValueError, "repeats"):
            J.pool_joint_posteriors([self.joint, self.joint])
        with self.assertRaisesRegex(ValueError, "dt_s"):
            J.pool_joint_posteriors([self.joint, replace(ko, acquisition=Acquisition(.033, .02))])
        with self.assertRaisesRegex(ValueError, "cells differ"):
            J.pool_joint_posteriors([self.joint, replace(ko, options=replace(OPTIONS, n_alpha=17))])
        with self.assertRaises(ValueError):
            J.pool_joint_posteriors([])


if __name__ == "__main__":
    unittest.main()
