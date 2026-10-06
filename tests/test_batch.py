"""Batch contracts: experiments keep their identity, batch == per-movie, ensemble MSD recovers D."""
import unittest

import numpy as np
import polars as pl

from diffusionkit import Acquisition, Experiment
from diffusionkit import classic, gridpost

ACQ = Acquisition(.03)


def movie(D, n_tracks=40, n=12, seed=0, sd=.01, first_id=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tracks):
        pos = np.cumsum(rng.normal(size=(n, 2)) * np.sqrt(2 * D * ACQ.dt_s), axis=0)
        pos = pos + rng.normal(size=(n, 2)) * sd
        rows.append(pl.DataFrame({"track_id": first_id + t, "frame": np.arange(n), "x_um": pos[:, 0],
                                  "y_um": pos[:, 1], "sigma_x_um": sd, "sigma_y_um": sd}))
    return pl.concat(rows)


def batch_inputs():
    # track ids deliberately collide across experiments
    return [Experiment("a1", movie(.05, seed=1), ACQ, "slow"), Experiment("a2", movie(.05, seed=2), ACQ, "slow"),
            Experiment("b1", movie(1., seed=3), Acquisition(.03, .01), "fast")]


class ExperimentTests(unittest.TestCase):
    def test_names_must_be_unique(self):
        e = batch_inputs()[0]
        with self.assertRaisesRegex(ValueError, "unique"):
            gridpost.analyze_experiments([e, e])
        with self.assertRaises(ValueError):
            gridpost.analyze_experiments([])


class GridPostBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exps = batch_inputs()
        cls.batch = gridpost.analyze_experiments(cls.exps, gridpost.GridPostOptions(n_D=121))

    def test_matches_single_movie(self):
        for e in self.exps:
            direct = gridpost.analyze_tracks(e.tracks, e.acquisition, self.batch.options, keep_posteriors=True)
            sel = self.batch.select(experiment=e.name)
            self.assertEqual(sel.fits.drop("sample", "experiment").to_dicts(), direct.fits.to_dicts())
            np.testing.assert_array_equal(sel.posteriors.log_post_D, direct.posteriors.log_post_D)
            self.assertEqual(sel.acquisition, e.acquisition)

    def test_labels_and_colliding_ids(self):
        self.assertEqual(self.batch.fits.height, 120)
        self.assertEqual(self.batch.fits.select("experiment", "track_id").n_unique(), 120)
        self.assertEqual(self.batch.select(sample="slow").fits.height, 80)
        self.assertIsNone(self.batch.select(sample=None).acquisition)  # mixed exposure
        with self.assertRaisesRegex(ValueError, "no sample"):
            self.batch.select(sample="nope")

    def test_progress_counts_tracks_over_batch(self):
        seen = []
        gridpost.analyze_experiments(self.exps, self.batch.options, progress=lambda d, t: seen.append((d, t)))
        self.assertEqual(seen[-1], (120, 120))
        self.assertEqual([d for d, _ in seen], sorted(d for d, _ in seen))

    def test_populations_separate_samples(self):
        pops = self.batch.populations("sample", n_samples=100)
        self.assertEqual(list(pops), ["slow", "fast"])
        self.assertEqual(pops["slow"].n_tracks, 80)
        slow_mass = pops["slow"].mass(0, .3)
        fast_mass = pops["fast"].mass(0, .3)
        self.assertGreater(slow_mass.mean(), .8)
        self.assertLess(fast_mass.mean(), .2)
        d = gridpost.cdf_distance(pops["slow"], pops["fast"])
        self.assertEqual(d.shape, (100,))
        self.assertGreater(d.mean(), 1.)  # ~ln(20) apart
        self.assertEqual(set(self.batch.populations("all", n_samples=50)), {None})

    def test_from_analyses_restored_pieces_match(self):
        analyses = {e.name: gridpost.analyze_tracks(e.tracks, e.acquisition, self.batch.options, keep_posteriors=True)
                    for e in self.exps}
        rebuilt = gridpost.GridPostBatch.from_analyses(analyses, {"a1": "slow", "a2": "slow", "b1": "fast"})
        self.assertEqual(rebuilt.fits.to_dicts(), self.batch.fits.to_dicts())
        np.testing.assert_array_equal(rebuilt.posteriors.log_post_D, self.batch.posteriors.log_post_D)
        other = gridpost.analyze_tracks(self.exps[0].tracks, ACQ, gridpost.GridPostOptions(n_D=61), keep_posteriors=True)
        with self.assertRaisesRegex(ValueError, "different GridPostOptions"):
            gridpost.GridPostBatch.from_analyses({"a1": analyses["a1"], "x": other})
        bare = gridpost.analyze_tracks(self.exps[0].tracks, ACQ, self.batch.options)
        with self.assertRaisesRegex(ValueError, "kept no posteriors"):
            gridpost.GridPostBatch.from_analyses({"a1": bare})

    def test_without_posteriors(self):
        batch = gridpost.analyze_experiments(self.exps[:1], keep_posteriors=False)
        with self.assertRaisesRegex(ValueError, "keep_posteriors"):
            batch.select()

    def test_by_track_length_takes_a_selection(self):
        sel = self.batch.select(sample="slow")
        comp = gridpost.by_track_length(sel.posteriors, self.batch.options.u_D())
        self.assertEqual(comp.n_tracks.sum(), 80)


class ClassicBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exps = [Experiment("a1", movie(.2, 150, seed=4), ACQ, "s"), Experiment("a2", movie(.2, 150, seed=5), ACQ, "s"),
                    Experiment("b", movie(.05, 150, seed=6), ACQ)]
        cls.batch = classic.analyze_experiments(cls.exps, classic.MSDOptions(max_lag=4))

    def test_ensemble_recovers_D_with_interval(self):
        ens = classic.ensemble_msd(self.batch, "sample", n_boot=100)
        fit = ens.fit(4, offset="provided")
        brown = fit.filter((pl.col("group") == "s") & (pl.col("model") == "brownian")).row(0, named=True)
        self.assertEqual(brown["status"], "ok")
        self.assertAlmostEqual(brown["D_um2_s"], .2, delta=.02)
        self.assertLess(brown["D_um2_s_lo"], brown["D_um2_s"])
        self.assertGreater(brown["D_um2_s_hi"], brown["D_um2_s"])
        self.assertEqual(brown["uncertainty_method"], "cluster_bootstrap_track")
        self.assertEqual(ens.curves.filter(pl.col("group") == "s")["n_units"].to_list(), [300] * 4)
        power = fit.filter((pl.col("group") == "s") & (pl.col("model") == "power_law")).row(0, named=True)
        self.assertAlmostEqual(power["alpha"], 1., delta=.25)

    def test_n_points_is_required_and_checked(self):
        ens = classic.ensemble_msd(self.batch, "all", n_boot=0)
        with self.assertRaises(TypeError):
            ens.fit()
        for bad in (2, 5, 3.5):
            with self.assertRaises(ValueError):
                ens.fit(bad)
        self.assertEqual(set(ens.fit(3)["n_points"]), {3})

    def test_refit_does_not_resample(self):
        ens = classic.ensemble_msd(self.batch, "sample", n_boot=20)
        a, b = ens.fit(3), ens.fit(4)
        self.assertEqual(a["n_points"].unique().to_list(), [3])
        self.assertEqual(b["n_points"].unique().to_list(), [4])
        self.assertEqual(ens.curve("s").lag.tolist(), [1, 2, 3, 4])

    def test_pooled_curve_is_pair_weighted_mean(self):
        ens = classic.ensemble_msd(self.batch, "all", n_boot=0)
        rows = self.batch.msd.filter(pl.col("lag") == 2)
        expected = (rows["n_pairs"] * rows["msd_um2"]).sum() / rows["n_pairs"].sum()
        self.assertAlmostEqual(ens.curves.filter(pl.col("lag") == 2)["msd_um2"][0], expected, places=12)
        fit = ens.fit(3, offset="provided")
        self.assertTrue(fit["D_um2_s_lo"].is_null().all())
        self.assertEqual(set(fit["uncertainty_method"]), {"not_estimated"})

    def test_experiment_resampling_and_track_weighting(self):
        ens = classic.ensemble_msd(self.batch, "sample", weight="tracks", resample="experiment", n_boot=50)
        s = ens.fit(3, offset="provided").filter((pl.col("group") == "s") & (pl.col("model") == "brownian")).row(0, named=True)
        self.assertEqual(s["n_units"], 2)
        self.assertEqual(s["uncertainty_method"], "cluster_bootstrap_experiment")

    def test_mixed_dt_is_refused(self):
        other = Experiment("c", movie(.2, 20, seed=7), Acquisition(.05), "s")
        batch = classic.analyze_experiments(self.exps[:1] + [other], classic.MSDOptions(max_lag=3))
        with self.assertRaisesRegex(ValueError, "frame intervals"):
            classic.ensemble_msd(batch, "sample", n_boot=0)

    def test_blurred_acquisition_has_no_ensemble(self):
        batch = classic.analyze_experiments([Experiment("x", movie(.2, 10), Acquisition(.03, .01))])
        with self.assertRaisesRegex(ValueError, "excluded"):
            classic.ensemble_msd(batch, "all")

    def test_bad_arguments(self):
        for kw in ({"by": "x"}, {"weight": "x"}, {"resample": "x"}):
            with self.assertRaises(ValueError):
                classic.ensemble_msd(self.batch, **kw)


class LagWindowTests(unittest.TestCase):
    def test_window_lags(self):
        self.assertEqual(classic.window_lags(40, .25), 10)
        self.assertEqual(classic.window_lags(4, .25), 3)   # at least three lags
        self.assertEqual(classic.window_lags(2, .25), 2)   # never more than exist
        with self.assertRaises(ValueError):
            classic.window_lags(10, 0)

    def test_fraction_gives_longer_tracks_more_lags(self):
        short, long_ = movie(.2, 1, 9, seed=1), movie(.2, 1, 41, seed=2)
        opts = classic.MSDOptions(max_lag=None, lag_fraction=.3)
        n = [len(classic.compute_msd(t, ACQ, opts).lag) for t in (short, long_)]
        self.assertEqual(n, [3, 12])

    def test_exactly_one_window_rule(self):
        for kw in ({"lag_fraction": .3}, {"max_lag": None}, {"lag_fraction": 1.5, "max_lag": None}):
            with self.assertRaises(ValueError):
                classic.analyze_tracks(movie(.2, 2), ACQ, classic.MSDOptions(**kw))


class TextbookFitTests(unittest.TestCase):
    """No SD columns anywhere: the offset is the intercept of the MSD-vs-lag line."""

    @classmethod
    def setUpClass(cls):
        cls.exps = [Experiment(f"m{i}", movie(.2, 200, 15, seed=20 + i, sd=.04).drop("sigma_x_um", "sigma_y_um"), ACQ, "s")
                    for i in range(2)]
        cls.batch = classic.analyze_experiments(cls.exps, classic.MSDOptions(max_lag=None, lag_fraction=.4, localization="ignore"))

    def test_ensemble_recovers_D_sigma_and_alpha(self):
        ens = classic.ensemble_msd(self.batch, "sample", n_boot=60)
        fit = ens.fit(5)
        lin = fit.filter(pl.col("model") == "linear").row(0, named=True)
        p = fit.filter(pl.col("model") == "power_law").row(0, named=True)
        self.assertAlmostEqual(lin["D_um2_s"], .2, delta=.03)
        self.assertAlmostEqual(lin["localization_sd_um"], .04, delta=.01)
        self.assertAlmostEqual(lin["offset_um2"], 4 * .04 ** 2, delta=.003)
        self.assertLess(lin["localization_sd_um_lo"], lin["localization_sd_um"])
        self.assertAlmostEqual(p["alpha"], 1., delta=.15)
        self.assertAlmostEqual(p["K_um2_s_alpha"], .2, delta=.04)

    def test_ensemble_fit_is_the_two_functions_composed(self):
        ens = classic.ensemble_msd(self.batch, "all", n_boot=0)
        curve = ens.curve("all")
        lin = classic.fit_linear_msd(curve, 5)
        log = classic.fit_loglog_msd(curve, lin.parameters["offset_um2"], n_points=5)
        row = ens.fit(5)
        self.assertAlmostEqual(row.filter(pl.col("model") == "linear")["D_um2_s"][0], lin.parameters["D_um2_s"], places=12)
        self.assertAlmostEqual(row.filter(pl.col("model") == "power_law")["alpha"][0], log.parameters["alpha"], places=12)

    def test_loglog_drops_nonpositive_points_and_says_so(self):
        curve = classic.MSDCurve(np.arange(1, 5), np.arange(1, 5) * .1, np.full(4, 9), np.array([.01, .5, .9, 1.2]),
                                 np.full(4, .05))
        fit = classic.fit_loglog_msd(curve)
        self.assertEqual(fit.n_lags, 3)
        self.assertIn("dropped", fit.message)
        thin = classic.MSDCurve(np.arange(1, 4), np.arange(1, 4) * .1, np.full(3, 9), np.array([.01, .02, .9]), np.full(3, .05))
        self.assertEqual(classic.fit_loglog_msd(thin).status, "insufficient_data")

    def test_negative_intercept_is_flagged_and_not_subtracted(self):
        tau = np.arange(1, 5) * .1
        curve = classic.MSDCurve(np.arange(1, 5), tau, np.full(4, 9), 4 * .2 * tau ** 1.5, np.zeros(4))
        lin = classic.fit_linear_msd(curve)
        self.assertEqual(lin.status, "nonphysical")
        self.assertIsNone(lin.parameters["localization_sd_um"])
        ens_like = classic.fit_loglog_msd(curve, 0.)  # what the ensemble fit does with a negative intercept
        self.assertAlmostEqual(ens_like.parameters["alpha"], 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
