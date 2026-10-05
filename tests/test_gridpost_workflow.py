"""Table/per-track workflow contracts for the D grid posterior."""
import subprocess
import sys
import unittest

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions, analyze_track, analyze_tracks
from diffusionkit.gridpost import posterior as P
from diffusionkit.gridpost.workflow import FIT_SCHEMA


def table(n=5, track_id=7):
    rng = np.random.default_rng(12)
    positions = rng.normal(size=(n, 2)) * .05
    sd = np.linspace(.005, .04, 2*n).reshape(n, 2)
    return pl.DataFrame({
        "track_id": [track_id] * n, "frame": np.arange(n),
        "x_um": positions[:, 0], "y_um": positions[:, 1],
        "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1],
    })


class WorkflowTests(unittest.TestCase):
    def test_short_track_is_excluded(self):
        out = analyze_track(table(2), Acquisition(.03))
        self.assertEqual(out.posterior_D.status, "excluded")
        self.assertIsNone(out.log_post_D)

    def test_exposure_blur_is_modelled(self):
        acquisition, options = Acquisition(.03, .02), GridPostOptions()
        out = analyze_track(table(), acquisition, options)
        self.assertEqual(out.posterior_D.status, "ok")
        direct = P.track_posterior(table(), acquisition, options=options)
        self.assertAlmostEqual(out.posterior_D.parameters["D_post_median_um2_s"], direct["median"], places=12)
        unblurred = analyze_track(table(), Acquisition(.03), options)
        self.assertNotAlmostEqual(out.posterior_D.parameters["D_post_median_um2_s"],
                                  unblurred.posterior_D.parameters["D_post_median_um2_s"], places=6)

    def test_zero_localization_sd_gives_invalid_input(self):
        zero = table().with_columns(pl.Series("sigma_x_um", np.zeros(5)))
        out = analyze_track(zero, Acquisition(.03))
        self.assertEqual(out.posterior_D.status, "invalid_input")

    def test_one_track_matches_direct_call(self):
        t = table()
        direct = analyze_track(t, Acquisition(.03))
        result = analyze_tracks(t, Acquisition(.03))
        post_D = result.fits.row(0, named=True)
        self.assertEqual(result.fits.height, 1)
        self.assertEqual(post_D["model"], "posterior_D")
        for name, value in direct.posterior_D.parameters.items():
            self.assertEqual(post_D[name], value)

    def test_info_bits_reported_from_the_log_posterior(self):
        out = analyze_track(table(8), Acquisition(.03))
        bits = out.posterior_D.parameters["D_post_info_bits"]
        self.assertAlmostEqual(bits, P.information_bits(out.log_post_D, P.flat(GridPostOptions().u_D())))
        self.assertGreater(bits, 0.)

    def test_localization_floor_is_mean_sd_squared_over_blurred_step(self):
        acquisition = Acquisition(.03, .02)
        track = table(8)
        sd = track.select("sigma_x_um", "sigma_y_um").to_numpy()
        expected = np.mean(sd ** 2) / (.03 - .02 / 3)
        out = analyze_track(track, acquisition)
        self.assertAlmostEqual(out.posterior_D.parameters["D_floor_um2_s"], expected, places=15)
        # At the floor, motion's diagonal in the displacement covariance equals the noise's (constant SD).
        from diffusionkit.gridpost.likelihood import localization_covariance, motion_covariance
        s = np.full(8, .02)
        floor = P.localization_floor(track.with_columns(sigma_x_um=s, sigma_y_um=s), acquisition)
        np.testing.assert_allclose(floor * np.diag(motion_covariance(7, .03, .02)),
                                   np.diag(localization_covariance(np.column_stack([s, s]))[0]), rtol=1e-12)

    def test_empty_output_retains_schema_and_reports_progress(self):
        calls = []
        result = analyze_tracks(table().head(0), Acquisition(.03), progress=lambda *x: calls.append(x))
        self.assertEqual(result.fits.schema, FIT_SCHEMA)
        self.assertEqual(result.fits.height, 0)
        self.assertEqual(calls, [(0, 0)])

    def test_invalid_and_short_tracks_do_not_drop_valid_tracks(self):
        good = table()
        gap = table(track_id=8).with_columns((pl.col("frame")*2).alias("frame"))
        short = table(2, track_id=9)
        calls = []
        result = analyze_tracks(pl.concat([good, gap, short]), Acquisition(.03), progress=lambda *x: calls.append(x))
        self.assertEqual(result.fits.height, 3)
        self.assertEqual(result.fits.filter(pl.col("track_id") == 8)["status"].to_list(), ["invalid_input"])
        self.assertEqual(result.fits.filter(pl.col("track_id") == 9)["status"].to_list(), ["excluded"])
        self.assertEqual(calls, [(0, 3), (1, 3), (2, 3), (3, 3)])

    def test_options_level_controls_interval_width(self):
        t = table()
        narrow = analyze_track(t, Acquisition(.03), GridPostOptions(level=.5))
        wide = analyze_track(t, Acquisition(.03), GridPostOptions(level=.99))
        d_narrow = narrow.posterior_D.parameters["D_post_hi_um2_s"] - narrow.posterior_D.parameters["D_post_lo_um2_s"]
        d_wide = wide.posterior_D.parameters["D_post_hi_um2_s"] - wide.posterior_D.parameters["D_post_lo_um2_s"]
        self.assertLess(d_narrow, d_wide)

    def test_options_grid_is_the_one_used(self):
        """A custom D range reaches every posterior: none falls back to a default grid."""
        t = table(12)
        options = GridPostOptions(D_min_um2_s=1e-3, D_max_um2_s=2., n_D=101)
        out = analyze_track(t, Acquisition(.03), options)
        self.assertEqual(out.log_post_D.shape, (101,))
        u = options.u_D()
        expected = P.summary(P.posterior(P.track_loglik(t, Acquisition(.03), u), P.flat(u)), u, options.level)
        self.assertAlmostEqual(out.posterior_D.parameters["D_post_median_um2_s"], expected["median"], places=12)
        self.assertAlmostEqual(np.exp(out.log_post_D).sum(), 1., places=12)
        self.assertEqual(P.track_posterior(t, Acquisition(.03), options=options), expected)

    def test_narrow_grid_moves_the_summary_and_says_so(self):
        """An upper edge below where the data put D cuts the posterior: the median sits
        at the edge and the message names it."""
        t = table(12)
        wide = analyze_track(t, Acquisition(.03))
        D_med = wide.posterior_D.parameters["D_post_median_um2_s"]
        cut = analyze_track(t, Acquisition(.03), GridPostOptions(D_max_um2_s=D_med / 3))
        self.assertEqual(cut.posterior_D.status, "ok")
        self.assertLess(cut.posterior_D.parameters["D_post_hi_um2_s"], D_med / 3 + 1e-12)
        self.assertIn("D_max_um2_s", cut.posterior_D.message)
        self.assertEqual(wide.posterior_D.message, "")

    def test_keep_posteriors_returns_ok_tracks_only(self):
        good = table()
        short = table(2, track_id=9)
        result = analyze_tracks(pl.concat([good, short]), Acquisition(.03), keep_posteriors=True)
        post = result.posteriors
        self.assertEqual(post.track_ids.tolist(), [7])
        self.assertEqual(post.log_post_D.shape, (1, result.options.n_D))
        np.testing.assert_allclose(post.log_post_D[0], analyze_track(good, Acquisition(.03)).log_post_D)
        self.assertIsNone(analyze_tracks(good, Acquisition(.03)).posteriors)

    def test_thread_pool_map_matches_serial(self):
        """A caller's executor map gives the serial result, rows and progress in track order."""
        from concurrent.futures import ThreadPoolExecutor
        tracks = pl.concat([table(n, track_id=i) for i, n in enumerate((12, 2, 6, 30, 5, 9))])
        zero = tracks.with_columns(pl.when(pl.col("track_id") == 4).then(0.).otherwise(pl.col("sigma_x_um"))
                                   .alias("sigma_x_um"))  # one invalid_input track
        acquisition = Acquisition(.03, .015)
        serial = analyze_tracks(zero, acquisition, keep_posteriors=True)
        calls = []
        with ThreadPoolExecutor(4) as pool:
            threaded = analyze_tracks(zero, acquisition, keep_posteriors=True, map_fn=pool.map,
                                      progress=lambda done, total: calls.append(done))
        self.assertTrue(serial.fits.equals(threaded.fits))
        self.assertIn("invalid_input", serial.fits["status"].to_list())
        np.testing.assert_array_equal(serial.posteriors.log_post_D, threaded.posteriors.log_post_D)
        np.testing.assert_array_equal(serial.posteriors.track_ids, threaded.posteriors.track_ids)
        self.assertEqual(calls, list(range(7)))

    def test_keep_posteriors_empty_keeps_grid_width(self):
        result = analyze_tracks(table(2), Acquisition(.03), GridPostOptions(n_D=11), keep_posteriors=True)
        self.assertEqual(result.posteriors.log_post_D.shape, (0, 11))

    def test_invalid_grid_options_raise(self):
        for bad in (dict(D_min_um2_s=0.), dict(D_min_um2_s=1., D_max_um2_s=.5), dict(D_max_um2_s=np.inf),
                    dict(n_D=1), dict(level=1.), dict(level=0.)):
            with self.subTest(**bad), self.assertRaises(ValueError):
                GridPostOptions(**bad)

    def test_import_does_not_load_bayes_or_plotting(self):
        code = "import diffusionkit.gridpost, sys; assert not any(x in sys.modules for x in ('jax','numpyro','matplotlib'))"
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
