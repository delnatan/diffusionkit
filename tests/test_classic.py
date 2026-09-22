"""Independent numerical checks and contracts for the rebuilt classical core."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np
import polars as pl

from diffusionkit import Acquisition, AcquisitionParams, Track, load_tracks, track_from_table
from diffusionkit.classic import (
    MSDCurve, MSDOptions, analyze_track, analyze_tracks, compute_msd,
    fit_anomalous_msd, fit_brownian_msd,
)
from diffusionkit.classic.workflow import FIT_SCHEMA, MSD_SCHEMA


def track(n=5):
    rng = np.random.default_rng(12)
    return Track(7, np.arange(n), rng.normal(size=(n, 2)) * .05,
                 np.linspace(.005, .04, 2*n).reshape(n, 2))


def table(t=None):
    t = track() if t is None else t
    data = {"track_id": [t.track_id] * len(t.frames), "frame": t.frames,
            "x_um": t.positions_um[:, 0], "y_um": t.positions_um[:, 1]}
    if t.localization_sd_um is not None:
        data.update(sigma_x_um=t.localization_sd_um[:, 0], sigma_y_um=t.localization_sd_um[:, 1])
    return pl.DataFrame(data)


def curve(y, offset=None, dt=.033):
    y = np.asarray(y, dtype=float)
    offset = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=float)
    lags = np.arange(1, len(y)+1)
    return MSDCurve(lags, lags*dt, 21-lags, y+offset, offset)


class InputTests(unittest.TestCase):
    def test_invalid_acquisition_and_options(self):
        for dt in (0., -1., np.nan, np.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                analyze_track(track(), Acquisition(dt))
        with self.assertRaisesRegex(ValueError, "Motion blur"):
            compute_msd(track(), Acquisition(.03, .02))
        with self.assertRaisesRegex(ValueError, "exceed"):
            analyze_track(track(), Acquisition(.03, .04))
        blurred = analyze_track(track(), Acquisition(.03, .02))
        self.assertEqual((blurred.brownian.status, blurred.anomalous.status), ("excluded", "excluded"))
        self.assertIsNone(blurred.msd)
        self.assertNotEqual(blurred.posterior_D.status, "excluded")
        for opts in (MSDOptions(max_lag=0), MSDOptions(max_lag=1.5),
                     MSDOptions(min_frames=1), MSDOptions(localization="unknown"),
                     MSDOptions(max_nfev=0)):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                analyze_track(track(), Acquisition(.03), opts)

    def test_gaps_duplicates_and_fractional_frames_rejected(self):
        for frames in (np.array([0, 1, 3, 4, 5]), np.array([0, 1, 1, 2, 3]),
                       np.arange(5)+.1, np.array([-1, 0, 1, 2, 3])):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                analyze_track(replace(track(), frames=frames), Acquisition(.03))

    def test_nonfinite_positions_and_errors_rejected(self):
        for field in ("positions_um", "localization_sd_um"):
            for bad in (np.nan, np.inf):
                values = getattr(track(), field).copy()
                values[0, 0] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    analyze_track(replace(track(), **{field: values}), Acquisition(.03))
        with self.assertRaises(ValueError):
            analyze_track(replace(track(), localization_sd_um=-np.ones((5, 2))), Acquisition(.03))

    def test_missing_errors_requires_explicit_opt_out(self):
        t = replace(track(), localization_sd_um=None)
        with self.assertRaisesRegex(ValueError, "Localization SDs"):
            analyze_track(t, Acquisition(.03))
        out = analyze_track(t, Acquisition(.03), MSDOptions(localization="ignore"))
        np.testing.assert_array_equal(out.msd.localization_offset_um2, 0.)
        np.testing.assert_array_equal(analyze_track(track(), Acquisition(.03),
                                                   MSDOptions(localization="ignore")).msd.localization_offset_um2, 0.)

    def test_sorting_copies_preserves_error_alignment(self):
        t = track()
        order = [2, 4, 0, 1, 3]
        shuffled = table(t)[order]
        restored = track_from_table(shuffled, Acquisition(.03))
        np.testing.assert_array_equal(restored.positions_um, t.positions_um)
        np.testing.assert_array_equal(restored.localization_sd_um, t.localization_sd_um)
        self.assertFalse(restored.frames.flags.writeable)
        np.testing.assert_allclose(compute_msd(restored, Acquisition(.03)).msd_um2,
                                   compute_msd(t, Acquisition(.03)).msd_um2)

    def test_single_table_requires_one_id_and_consistent_metadata(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            track_from_table(pl.concat([table(), table(replace(track(), track_id=8))]), Acquisition(.03))
        for bad in (table().with_columns(pl.lit(99).alias("track_length")),
                    table().with_columns((pl.col("frame")*.06).alias("t_s"))):
            with self.assertRaises(ValueError):
                track_from_table(bad, Acquisition(.03))

    def test_bad_schema_rejected(self):
        for bad in (table().drop("x_um"), table().drop("sigma_x_um"),
                    table().with_columns(pl.col("frame").cast(pl.Float64)),
                    table().with_columns(pl.lit(None, dtype=pl.Int64).alias("track_id"))):
            with self.subTest(schema=bad.schema), self.assertRaises(ValueError):
                analyze_tracks(bad, Acquisition(.03))

    def test_pixel_conversion_and_fractional_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"tracks.csv"
            pl.DataFrame({"track_id": [1, 1], "frame": [1, 0], "x": [2., 3.], "y": [4., 5.],
                          "sigma_x": [.1, .2], "sigma_y": [.3, .4]}).write_csv(path)
            converted = load_tracks(path, AcquisitionParams(.1, .03))
            np.testing.assert_allclose(converted["x_um"], [.3, .2])
            np.testing.assert_allclose(converted["sigma_y_um"], [.04, .03])
            self.assertEqual(converted["track_length"].to_list(), [2, 2])
            pl.read_csv(path).with_columns((pl.col("frame")+.5).alias("frame")).write_csv(path)
            with self.assertRaises(ValueError):
                load_tracks(path, AcquisitionParams(.1, .03))


class MSDTests(unittest.TestCase):
    def test_heteroscedastic_pair_sum_oracle(self):
        t = track(8)
        observed = compute_msd(t, Acquisition(.03), MSDOptions(max_lag=7))
        for j, lag in enumerate(observed.lag):
            pairs = [(i, i+lag) for i in range(8-lag)]
            expected_msd = np.mean([sum((t.positions_um[b, d]-t.positions_um[a, d])**2
                                        for d in (0, 1)) for a, b in pairs])
            expected_offset = np.mean([sum(t.localization_sd_um[a, d]**2+t.localization_sd_um[b, d]**2
                                           for d in (0, 1)) for a, b in pairs])
            self.assertAlmostEqual(observed.msd_um2[j], expected_msd)
            self.assertAlmostEqual(observed.localization_offset_um2[j], expected_offset)
        self.assertGreater(np.ptp(observed.localization_offset_um2), 0.)

    def test_constant_noise_reduces_to_four_sigma_squared(self):
        t = replace(track(), localization_sd_um=np.full((5, 2), .02))
        np.testing.assert_allclose(compute_msd(t, Acquisition(.03)).localization_offset_um2, 4*.02**2)

    def test_error_placement_matters(self):
        t = track()
        moved = replace(t, localization_sd_um=np.roll(t.localization_sd_um, 1, axis=0))
        a, b = (compute_msd(x, Acquisition(.03)) for x in (t, moved))
        self.assertFalse(np.allclose(a.localization_offset_um2, b.localization_offset_um2))
        np.testing.assert_array_equal(a.msd_um2, b.msd_um2)

    def test_units_translation_and_axis_permutation(self):
        t, acq = track(), Acquisition(.03)
        a = analyze_track(t, acq)
        b = analyze_track(replace(t, positions_um=t.positions_um+100), acq)
        self.assertAlmostEqual(a.brownian.parameters["D_um2_s"], b.brownian.parameters["D_um2_s"])
        swapped = replace(t, positions_um=t.positions_um[:, ::-1], localization_sd_um=t.localization_sd_um[:, ::-1])
        np.testing.assert_allclose(compute_msd(t, acq).msd_um2, compute_msd(swapped, acq).msd_um2)
        scaled = replace(t, positions_um=10*t.positions_um, localization_sd_um=10*t.localization_sd_um)
        c = analyze_track(scaled, acq)
        self.assertAlmostEqual(c.brownian.parameters["D_um2_s"], 100*a.brownian.parameters["D_um2_s"])

    def test_window_and_short_track(self):
        c = compute_msd(track(), Acquisition(.03), MSDOptions(max_lag=100))
        self.assertEqual(c.lag.tolist(), [1, 2, 3, 4])
        self.assertEqual(c.n_pairs.tolist(), [4, 3, 2, 1])
        out = analyze_track(track(3), Acquisition(.03))
        self.assertEqual(out.brownian.status, "excluded")
        self.assertIsNone(out.msd)


class EstimatorTests(unittest.TestCase):
    def test_exact_brownian_and_power_law_mean_curves(self):
        tau = np.arange(1, 5)*.033
        offset = np.array([.001, .003, .002, .004])
        fit = fit_brownian_msd(curve(4*.05*tau, offset))
        self.assertAlmostEqual(fit.parameters["D_um2_s"], .05)
        for alpha in (.2, .7, 1., 1.8):
            fit = fit_anomalous_msd(curve(4*.05*tau**alpha, offset))
            with self.subTest(alpha=alpha):
                self.assertEqual(fit.status, "ok")
                self.assertAlmostEqual(fit.parameters["alpha"], alpha, places=6)
                self.assertAlmostEqual(fit.parameters["K_um2_s_alpha"], .05, places=6)
                self.assertEqual(fit.uncertainty_method, "not_estimated")

    def test_negative_values_are_retained(self):
        fit = fit_brownian_msd(curve([-1., -2., -3.], [5., 5., 5.]))
        self.assertEqual(fit.status, "nonphysical")
        self.assertLess(fit.parameters["D_um2_s"], 0.)
        fit = fit_anomalous_msd(curve([-.01, .02, .03], [.02, .02, .02]))
        self.assertEqual(fit.n_lags, 3)
        self.assertIn(fit.status, ("ok", "boundary"))

    def test_boundary_and_unidentified_are_not_ok(self):
        tau = np.arange(1, 4)*.033
        self.assertEqual(fit_anomalous_msd(curve(4*.05*tau**2)).status, "boundary")
        for y in ([0., 0., 0.], [-1., -1., -1.]):
            fit = fit_anomalous_msd(curve(y, [2., 2., 2.]))
            self.assertEqual(fit.status, "unidentified")
            self.assertIsNone(fit.parameters["alpha"])

    def test_failure_and_insufficient_lags(self):
        failed = fit_anomalous_msd(curve([.01, .07, .09]), max_nfev=1)
        self.assertEqual(failed.status, "optimizer_failed")
        self.assertEqual(failed.optimizer_status, 0)
        self.assertEqual(fit_anomalous_msd(curve([.01, .02])).status, "insufficient_data")
        self.assertEqual(fit_brownian_msd(curve([])).status, "insufficient_data")

    def test_nonlinear_objective_against_independent_profile_grid(self):
        rng = np.random.default_rng(22)
        tau = np.arange(1, 4)*.033
        alpha_grid = np.linspace(0, 2, 10001)
        basis = tau[:, None]**alpha_grid
        for _ in range(20):
            y = rng.normal(.01, .015, 3)
            if np.all(y <= 0):
                continue
            c = curve(y, np.maximum(-y, 0.))
            fit = fit_anomalous_msd(c)
            amplitudes = np.maximum((y[:, None]*basis).sum(axis=0)/(basis**2).sum(axis=0), 0.)
            oracle_sse = np.min(((amplitudes*basis-y[:, None])**2).sum(axis=0))
            self.assertLessEqual(fit.residual_sum_squares_um4, oracle_sse+1e-10)

    def test_heteroscedastic_brownian_recovery(self):
        # Independent position simulator; mean fit is linear, so fitting the
        # sample mean curve equals averaging the individual OLS estimates.
        rng = np.random.default_rng(480)
        dt, D, count = .033, .05, 12000
        for n in (5, 10, 20):
            sd = np.linspace(.005, .07, 2*n).reshape(n, 2)
            steps = rng.normal(size=(count, n-1, 2))*np.sqrt(2*D*dt)
            pos = np.concatenate([np.zeros((count, 1, 2)), steps.cumsum(axis=1)], axis=1)
            pos += rng.normal(size=pos.shape)*sd
            means, offsets, per_track = [], [], []
            for lag in (1, 2, 3):
                values = ((pos[:, lag:]-pos[:, :-lag])**2).sum(axis=2).mean(axis=1)
                offset = (sd[lag:]**2+sd[:-lag]**2).sum(axis=1).mean()
                means.append(values.mean())
                offsets.append(offset)
                per_track.append(values-offset)
            tau = np.arange(1, 4)*dt
            ds = tau @ np.array(per_track)/(4*(tau @ tau))
            estimate = fit_brownian_msd(curve(np.array(means)-offsets, offsets)).parameters["D_um2_s"]
            with self.subTest(n=n):
                self.assertAlmostEqual(estimate, ds.mean())
                self.assertLess(abs(estimate-D), 5*ds.std(ddof=1)/np.sqrt(count))


class WorkflowTests(unittest.TestCase):
    def test_one_track_matches_direct_call(self):
        t = track()
        direct = analyze_track(t, Acquisition(.03))
        result = analyze_tracks(table(t), Acquisition(.03))
        normal = result.fits.filter(pl.col("model") == "brownian").row(0, named=True)
        post = result.fits.filter(pl.col("model") == "posterior_D").row(0, named=True)
        self.assertEqual(result.fits.height, 3)
        self.assertEqual(normal["D_um2_s"], direct.brownian.parameters["D_um2_s"])
        for name, value in direct.posterior_D.parameters.items():
            self.assertEqual(post[name], value)
        self.assertEqual(result.msd.height, 3)

    def test_empty_output_retains_schema_and_reports_progress(self):
        calls = []
        result = analyze_tracks(table().head(0), Acquisition(.03), progress=lambda *x: calls.append(x))
        self.assertEqual(result.fits.schema, FIT_SCHEMA)
        self.assertEqual(result.msd.schema, MSD_SCHEMA)
        self.assertEqual(result.fits.height, 0)
        self.assertEqual(calls, [(0, 0)])

    def test_invalid_and_short_tracks_do_not_drop_valid_tracks(self):
        good = table()
        gap = table(replace(track(), track_id=8, frames=np.arange(5)*2))
        short = table(replace(track(2), track_id=9))
        calls = []
        result = analyze_tracks(pl.concat([good, gap, short]), Acquisition(.03), progress=lambda *x: calls.append(x))
        self.assertEqual(result.fits.height, 9)
        self.assertEqual(result.fits.filter(pl.col("track_id") == 8)["status"].to_list(), ["invalid_input"]*3)
        self.assertEqual(result.fits.filter(pl.col("track_id") == 9)["status"].to_list(), ["excluded"]*3)
        self.assertEqual(result.msd["track_id"].unique().to_list(), [7])
        self.assertEqual(calls, [(0, 3), (1, 3), (2, 3), (3, 3)])

    def test_import_does_not_load_bayes_plotting_or_legacy(self):
        code = "import diffusionkit.classic, sys; assert not any(x in sys.modules for x in ('jax','numpyro','matplotlib','diffusionkit.legacy.classic'))"
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_legacy_paths_preserve_saved_implementation(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from diffusionkit.classic import fit_population
        self.assertIn("legacy.classic", fit_population.__module__)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        from diffusionkit.classic.fitting import fit_normal_diffusion
        from diffusionkit.legacy.classic.fitting import fit_normal_diffusion as original
        self.assertIs(fit_normal_diffusion, original)


if __name__ == "__main__":
    unittest.main()
