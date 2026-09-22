"""Table/per-track workflow contracts for the D and alpha grid posteriors."""
import subprocess
import sys
import unittest

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions, analyze_track, analyze_tracks
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
        self.assertEqual(out.posterior_alpha.status, "excluded")

    def test_exposure_blur_excludes_alpha_not_D(self):
        out = analyze_track(table(), Acquisition(.03, .02))
        self.assertNotEqual(out.posterior_D.status, "excluded")
        self.assertEqual(out.posterior_alpha.status, "excluded")

    def test_zero_localization_sd_gives_invalid_input(self):
        zero = table().with_columns(pl.Series("sigma_x_um", np.zeros(5)))
        out = analyze_track(zero, Acquisition(.03))
        self.assertEqual(out.posterior_D.status, "invalid_input")
        self.assertEqual(out.posterior_alpha.status, "invalid_input")

    def test_one_track_matches_direct_call(self):
        t = table()
        direct = analyze_track(t, Acquisition(.03))
        result = analyze_tracks(t, Acquisition(.03))
        post_D = result.fits.filter(pl.col("model") == "posterior_D").row(0, named=True)
        post_alpha = result.fits.filter(pl.col("model") == "posterior_alpha").row(0, named=True)
        self.assertEqual(result.fits.height, 2)
        for name, value in direct.posterior_D.parameters.items():
            self.assertEqual(post_D[name], value)
        for name, value in direct.posterior_alpha.parameters.items():
            self.assertEqual(post_alpha[name], value)

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
        self.assertEqual(result.fits.height, 6)
        self.assertEqual(result.fits.filter(pl.col("track_id") == 8)["status"].to_list(), ["invalid_input"]*2)
        self.assertEqual(result.fits.filter(pl.col("track_id") == 9)["status"].to_list(), ["excluded"]*2)
        self.assertEqual(calls, [(0, 3), (1, 3), (2, 3), (3, 3)])

    def test_options_level_controls_interval_width(self):
        t = table()
        narrow = analyze_track(t, Acquisition(.03), GridPostOptions(level=.5))
        wide = analyze_track(t, Acquisition(.03), GridPostOptions(level=.99))
        d_narrow = narrow.posterior_D.parameters["D_post_hi_um2_s"] - narrow.posterior_D.parameters["D_post_lo_um2_s"]
        d_wide = wide.posterior_D.parameters["D_post_hi_um2_s"] - wide.posterior_D.parameters["D_post_lo_um2_s"]
        self.assertLess(d_narrow, d_wide)

    def test_import_does_not_load_bayes_or_plotting(self):
        code = "import diffusionkit.gridpost, sys; assert not any(x in sys.modules for x in ('jax','numpyro','matplotlib'))"
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
