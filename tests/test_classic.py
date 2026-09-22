"""Independent numerical checks and contracts for the classical MSD core."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import polars as pl

from diffusionkit import Acquisition, AcquisitionParams, load_tracks, validated_track_frame
from diffusionkit.classic import (
    MSDCurve, MSDOptions, analyze_track, analyze_tracks, compute_msd,
    fit_anomalous_msd, fit_brownian_msd,
)
from diffusionkit.classic.workflow import FIT_SCHEMA, MSD_SCHEMA


def table(n=5, track_id=7):
    rng = np.random.default_rng(12)
    positions = rng.normal(size=(n, 2)) * .05
    sd = np.linspace(.005, .04, 2*n).reshape(n, 2)
    return pl.DataFrame({
        "track_id": [track_id] * n, "frame": np.arange(n),
        "x_um": positions[:, 0], "y_um": positions[:, 1],
        "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1],
    })


def _with_value(df, col, idx, value):
    arr = df[col].to_numpy().copy()
    arr[idx] = value
    return df.with_columns(pl.Series(col, arr))


def curve(y, offset=None, dt=.033):
    y = np.asarray(y, dtype=float)
    offset = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=float)
    lags = np.arange(1, len(y)+1)
    return MSDCurve(lags, lags*dt, 21-lags, y+offset, offset)


class InputTests(unittest.TestCase):
    def test_invalid_acquisition_and_options(self):
        for dt in (0., -1., np.nan, np.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                analyze_track(table(), Acquisition(dt))
        with self.assertRaisesRegex(ValueError, "Motion blur"):
            compute_msd(table(), Acquisition(.03, .02))
        with self.assertRaisesRegex(ValueError, "exceed"):
            analyze_track(table(), Acquisition(.03, .04))
        blurred = analyze_track(table(), Acquisition(.03, .02))
        self.assertEqual((blurred.brownian.status, blurred.anomalous.status), ("excluded", "excluded"))
        self.assertIsNone(blurred.msd)
        for opts in (MSDOptions(max_lag=0), MSDOptions(max_lag=1.5),
                     MSDOptions(min_frames=1), MSDOptions(localization="unknown"),
                     MSDOptions(max_nfev=0)):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                analyze_track(table(), Acquisition(.03), opts)

    def test_gaps_duplicates_and_fractional_frames_rejected(self):
        for frames in (np.array([0, 1, 3, 4, 5]), np.array([0, 1, 1, 2, 3]),
                       np.array([-1, 0, 1, 2, 3])):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                analyze_track(table().with_columns(pl.Series("frame", frames)), Acquisition(.03))
        with self.assertRaises(ValueError):
            analyze_track(table().with_columns((pl.col("frame")+.1).alias("frame")), Acquisition(.03))

    def test_nonfinite_positions_and_errors_rejected(self):
        for field in ("x_um", "sigma_x_um"):
            for bad in (np.nan, np.inf):
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    analyze_track(_with_value(table(), field, 0, bad), Acquisition(.03))
        negative = table().with_columns((-pl.col("sigma_x_um")).alias("sigma_x_um"),
                                        (-pl.col("sigma_y_um")).alias("sigma_y_um"))
        with self.assertRaises(ValueError):
            analyze_track(negative, Acquisition(.03))

    def test_missing_errors_requires_explicit_opt_out(self):
        t = table().drop("sigma_x_um", "sigma_y_um")
        with self.assertRaisesRegex(ValueError, "Localization SDs"):
            analyze_track(t, Acquisition(.03))
        out = analyze_track(t, Acquisition(.03), MSDOptions(localization="ignore"))
        np.testing.assert_array_equal(out.msd.localization_offset_um2, 0.)
        np.testing.assert_array_equal(analyze_track(table(), Acquisition(.03),
                                                   MSDOptions(localization="ignore")).msd.localization_offset_um2, 0.)

    def test_sorting_copies_preserves_error_alignment(self):
        t = table()
        order = [2, 4, 0, 1, 3]
        shuffled = t[order]
        restored = validated_track_frame(shuffled, Acquisition(.03))
        np.testing.assert_array_equal(restored.select("x_um", "y_um").to_numpy(),
                                      t.select("x_um", "y_um").to_numpy())
        np.testing.assert_array_equal(restored.select("sigma_x_um", "sigma_y_um").to_numpy(),
                                      t.select("sigma_x_um", "sigma_y_um").to_numpy())
        np.testing.assert_allclose(compute_msd(restored, Acquisition(.03)).msd_um2,
                                   compute_msd(t, Acquisition(.03)).msd_um2)

    def test_single_table_requires_one_id_and_consistent_metadata(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validated_track_frame(pl.concat([table(), table(track_id=8)]), Acquisition(.03))
        for bad in (table().with_columns(pl.lit(99).alias("track_length")),
                    table().with_columns((pl.col("frame")*.06).alias("t_s"))):
            with self.assertRaises(ValueError):
                validated_track_frame(bad, Acquisition(.03))

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
        t = table(8)
        positions = t.select("x_um", "y_um").to_numpy()
        sd = t.select("sigma_x_um", "sigma_y_um").to_numpy()
        observed = compute_msd(t, Acquisition(.03), MSDOptions(max_lag=7))
        for j, lag in enumerate(observed.lag):
            pairs = [(i, i+lag) for i in range(8-lag)]
            expected_msd = np.mean([sum((positions[b, d]-positions[a, d])**2
                                        for d in (0, 1)) for a, b in pairs])
            expected_offset = np.mean([sum(sd[a, d]**2+sd[b, d]**2
                                           for d in (0, 1)) for a, b in pairs])
            self.assertAlmostEqual(observed.msd_um2[j], expected_msd)
            self.assertAlmostEqual(observed.localization_offset_um2[j], expected_offset)
        self.assertGreater(np.ptp(observed.localization_offset_um2), 0.)

    def test_constant_noise_reduces_to_four_sigma_squared(self):
        t = table().with_columns(pl.lit(.02).alias("sigma_x_um"), pl.lit(.02).alias("sigma_y_um"))
        np.testing.assert_allclose(compute_msd(t, Acquisition(.03)).localization_offset_um2, 4*.02**2)

    def test_error_placement_matters(self):
        t = table()
        sd = t.select("sigma_x_um", "sigma_y_um").to_numpy()
        moved = t.with_columns(pl.Series("sigma_x_um", np.roll(sd[:, 0], 1)),
                               pl.Series("sigma_y_um", np.roll(sd[:, 1], 1)))
        a, b = (compute_msd(x, Acquisition(.03)) for x in (t, moved))
        self.assertFalse(np.allclose(a.localization_offset_um2, b.localization_offset_um2))
        np.testing.assert_array_equal(a.msd_um2, b.msd_um2)

    def test_units_translation_and_axis_permutation(self):
        t, acq = table(), Acquisition(.03)
        a = analyze_track(t, acq)
        b = analyze_track(t.with_columns((pl.col("x_um")+100).alias("x_um"),
                                         (pl.col("y_um")+100).alias("y_um")), acq)
        self.assertAlmostEqual(a.brownian.parameters["D_um2_s"], b.brownian.parameters["D_um2_s"])
        positions = t.select("x_um", "y_um").to_numpy()
        sd = t.select("sigma_x_um", "sigma_y_um").to_numpy()
        swapped = t.with_columns(pl.Series("x_um", positions[:, 1]), pl.Series("y_um", positions[:, 0]),
                                 pl.Series("sigma_x_um", sd[:, 1]), pl.Series("sigma_y_um", sd[:, 0]))
        np.testing.assert_allclose(compute_msd(t, acq).msd_um2, compute_msd(swapped, acq).msd_um2)
        scaled = t.with_columns((pl.col("x_um")*10).alias("x_um"), (pl.col("y_um")*10).alias("y_um"),
                                (pl.col("sigma_x_um")*10).alias("sigma_x_um"),
                                (pl.col("sigma_y_um")*10).alias("sigma_y_um"))
        c = analyze_track(scaled, acq)
        self.assertAlmostEqual(c.brownian.parameters["D_um2_s"], 100*a.brownian.parameters["D_um2_s"])

    def test_window_and_short_track(self):
        c = compute_msd(table(), Acquisition(.03), MSDOptions(max_lag=100))
        self.assertEqual(c.lag.tolist(), [1, 2, 3, 4])
        self.assertEqual(c.n_pairs.tolist(), [4, 3, 2, 1])
        out = analyze_track(table(3), Acquisition(.03))
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
        t = table()
        direct = analyze_track(t, Acquisition(.03))
        result = analyze_tracks(t, Acquisition(.03))
        normal = result.fits.filter(pl.col("model") == "brownian").row(0, named=True)
        self.assertEqual(result.fits.height, 2)
        self.assertEqual(normal["D_um2_s"], direct.brownian.parameters["D_um2_s"])
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
        gap = table(track_id=8).with_columns((pl.col("frame")*2).alias("frame"))
        short = table(2, track_id=9)
        calls = []
        result = analyze_tracks(pl.concat([good, gap, short]), Acquisition(.03), progress=lambda *x: calls.append(x))
        self.assertEqual(result.fits.height, 6)
        self.assertEqual(result.fits.filter(pl.col("track_id") == 8)["status"].to_list(), ["invalid_input"]*2)
        self.assertEqual(result.fits.filter(pl.col("track_id") == 9)["status"].to_list(), ["excluded"]*2)
        self.assertEqual(result.msd["track_id"].unique().to_list(), [7])
        self.assertEqual(calls, [(0, 3), (1, 3), (2, 3), (3, 3)])

    def test_import_does_not_load_bayes_or_plotting(self):
        code = "import diffusionkit.classic, sys; assert not any(x in sys.modules for x in ('jax','numpyro','matplotlib'))"
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
