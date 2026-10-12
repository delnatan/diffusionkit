"""The fBm (alpha, D) grid posterior: its covariance, likelihood, prior anchoring, calibration and workflow."""
import unittest

import numpy as np
import polars as pl
from scipy.stats import multivariate_normal

from diffusionkit import Acquisition
from diffusionkit.fbmpost import FbmPostOptions, analyze_track, analyze_tracks, simulate_tracks
from diffusionkit.fbmpost import covariance as R
from diffusionkit.fbmpost import posterior as P
from diffusionkit.fbmpost.workflow import FIT_SCHEMA
from diffusionkit.gridpost import brownian_log_likelihood
from diffusionkit.gridpost.likelihood import localization_covariance, motion_covariance

DT = .033
ACQ = Acquisition(.044, .02)


class CovarianceTests(unittest.TestCase):
    """The numpy blurred-fBm covariance (`fbmpost.covariance`)."""
    def test_blurred_fgn_reduces_to_berglund_at_alpha_one(self):
        """fBm's exposure average at alpha=1 is the D posterior's Berglund closed form, at any exposure."""
        for exposure in (0., .001, .02, DT):
            np.testing.assert_allclose(R.fgn_motion_covariance(6, DT, 1.0, exposure),
                                       motion_covariance(6, DT, exposure), rtol=1e-12, atol=1e-15)

    def test_blurred_fgn_without_exposure_is_plain_fgn(self):
        k = np.arange(5.)
        for alpha in (.3, 1.4):
            gamma = DT**alpha * (np.abs(k + 1)**alpha - 2*k**alpha + np.abs(k - 1)**alpha)
            np.testing.assert_allclose(R.fgn_motion_covariance(5, DT, alpha)[0], gamma, rtol=1e-12)

    def test_blurred_fgn_matches_quadrature(self):
        """The closed form against a midpoint double integral of |x + u - v|^alpha over both exposures."""
        exposure, m, n = .02, 5, 600
        u = (np.arange(n) + .5) / n * exposure
        w = (u[:, None] - u[None, :]).ravel()
        for alpha in (.3, .8, 1.5, 1.9):
            G = lambda x: np.mean(np.abs(x + w)**alpha)  # noqa: E731
            gamma = [G((k+1)*DT) - 2*G(k*DT) + G((k-1)*DT) for k in range(m)]
            np.testing.assert_allclose(R.fgn_motion_covariance(m, DT, alpha, exposure)[0], gamma,
                                       rtol=1e-3, atol=1e-6 * gamma[0])

    def test_blurred_fgn_is_continuous_across_its_series_switch(self):
        """E|x + u - v|^alpha switches to its series at te/x = R._BLUR_SERIES_RATIO; both sides must agree."""
        for alpha in (.05, .4, 1.3, 1.95):
            for te in (1e-4, .02):
                x = te / R._BLUR_SERIES_RATIO * np.array([1 - 1e-12, 1 + 1e-12])
                below, above = R._blurred_abs_power(x, alpha, te)
                self.assertLess(abs(below / above - 1), 1e-10)
        for alpha in (.4, 1.3):
            # Lag 0's blur fades like te^alpha, slowly at small alpha: 1e-30 s is invisible.
            np.testing.assert_allclose(R.fgn_motion_covariance(4, DT, alpha, 1e-30),
                                       R.fgn_motion_covariance(4, DT, alpha), rtol=1e-10)

    def test_blurred_fgn_matches_fine_step_simulation(self):
        """Covariance of box-averaged, exactly simulated fBm displacements (no localization noise)."""
        rng = np.random.default_rng(3)
        exposure, m, sub, sims = .02, 3, 60, 20000
        h = DT / sub
        n_on = round(exposure / h)
        t = np.arange(1, (m + 1) * sub + 1) * h  # fine time points after z(0) = 0
        for alpha in (.5, 1.5):
            # fBm with E[(z(t) - z(s))^2] = 2|t - s|^alpha: Cov(z(t), z(s)) = |t|^a + |s|^a - |t - s|^a.
            cov = t[:, None]**alpha + t[None, :]**alpha - np.abs(t[:, None] - t[None, :])**alpha
            path = np.hstack([np.zeros((sims, 1)), rng.standard_normal((sims, len(t))) @ np.linalg.cholesky(cov).T])
            w = np.ones(n_on + 1)
            w[[0, -1]] = .5
            pos = np.stack([path[:, i*sub:i*sub + n_on + 1] @ w / n_on for i in range(m + 1)], axis=1)
            emp, model = np.cov(np.diff(pos, axis=1).T), R.fgn_motion_covariance(m, DT, alpha, exposure)
            self.assertLess(np.abs(emp - model).max(), .04 * model.max())


class LikelihoodTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.track = simulate_tracks(1, 12, .6, .02, .015, ACQ, rng=rng).with_columns(
            sigma_x_um=pl.Series(rng.uniform(.01, .03, 12)))
        self.options = FbmPostOptions()
        self.grid = P.log_likelihood_grid(self.track, ACQ, self.options)
        self.ll = self.grid.loglik

    def test_matches_direct_gaussian(self):
        """The per-alpha whitening is the exact Gaussian density, per-frame SDs and blur included."""
        xy = self.track.select("x_um", "y_um").to_numpy()
        B = localization_covariance(self.track.select("sigma_x_um", "sigma_y_um").to_numpy())
        for ia, iu in [(3, 50), (49, 200), (90, 420)]:
            a = self.options.alphas()[ia]
            K = np.exp(self.grid.ln_K[ia, iu])
            C = R.fgn_motion_covariance(11, ACQ.dt_s, a, ACQ.exposure_s)
            direct = sum(multivariate_normal(np.zeros(11), K * C + B[k]).logpdf(np.diff(xy[:, k])) for k in range(2))
            self.assertAlmostEqual(self.ll[ia, iu], direct, places=8)

    def test_alpha_one_row_is_the_brownian_likelihood(self):
        """At alpha = 1, K is the Brownian D, and the row is gridpost's likelihood."""
        i = int(np.argmin(np.abs(self.options.alphas() - 1)))
        self.assertAlmostEqual(self.options.alphas()[i], 1.)
        for iu in (40, 250, 500):
            D = np.exp(self.grid.ln_K[i, iu])
            self.assertAlmostEqual(self.ll[i, iu], brownian_log_likelihood(self.track, ACQ, D), places=8)

    def test_chunking_does_not_change_the_grid(self):
        old = P._CHUNK_FLOATS
        try:
            P._CHUNK_FLOATS = 1
            np.testing.assert_allclose(P.log_likelihood_grid(self.track, ACQ, self.options).loglik, self.ll,
                                       rtol=0, atol=1e-9)
        finally:
            P._CHUNK_FLOATS = old


class PriorAndCalibrationTests(unittest.TestCase):
    def test_immobile_tracks_return_the_flat_alpha_prior(self):
        """With the amplitude in SNR units, a track that sees only noise teaches nothing about alpha."""
        tracks = simulate_tracks(40, 20, 1., 0., .015, ACQ, rng=np.random.default_rng(1))
        r = analyze_tracks(tracks, ACQ, keep_posteriors=True)
        p = np.exp(r.posteriors.log_post_alpha).mean(axis=0)
        self.assertLess(np.abs(p * len(p) - 1).max(), .1)
        self.assertLess(r.fits["alpha_post_info_bits"].max(), .05)
        self.assertTrue((r.fits["D_grid_edge"] == "low").all())

    def test_tau_ref_changes_D_ref_not_alpha(self):
        track = simulate_tracks(1, 30, .5, .02, .01, ACQ, rng=np.random.default_rng(2))
        a = analyze_track(track, ACQ, FbmPostOptions(tau_ref_s=.05)).posterior.parameters
        b = analyze_track(track, ACQ, FbmPostOptions(tau_ref_s=.5)).posterior.parameters
        for name in ("alpha_post_mean", "alpha_post_lo", "alpha_post_hi", "alpha_post_info_bits"):
            self.assertAlmostEqual(a[name], b[name], places=12)
        self.assertGreater(a["D_ref_post_mean_um2_s"], b["D_ref_post_mean_um2_s"])  # alpha < 1: D falls with lag

    def test_recovers_alpha_and_D_ref(self):
        rng = np.random.default_rng(3)
        for alpha in (.4, 1., 1.5):
            fits = analyze_tracks(simulate_tracks(30, 40, alpha, .03, .01, ACQ, rng=rng), ACQ).fits
            a_cover = ((fits["alpha_post_lo"] <= alpha) & (alpha <= fits["alpha_post_hi"])).mean()
            d_cover = ((fits["D_ref_post_lo_um2_s"] <= .03) & (.03 <= fits["D_ref_post_hi_um2_s"])).mean()
            self.assertGreater(a_cover, .75, alpha)
            self.assertGreater(d_cover, .75, alpha)
            self.assertLess(abs(fits["alpha_post_mean"].median() - alpha), .1, alpha)


class WorkflowTests(unittest.TestCase):
    def table(self, n=8):
        return simulate_tracks(1, n, .7, .02, .015, ACQ, rng=np.random.default_rng(4), first_track_id=7)

    def test_short_track_is_excluded(self):
        out = analyze_track(self.table(2), ACQ)
        self.assertEqual(out.posterior.status, "excluded")
        self.assertIsNone(out.log_post_alpha)

    def test_zero_localization_sd_gives_invalid_input(self):
        zero = self.table().with_columns(pl.Series("sigma_x_um", np.zeros(8)))
        r = analyze_tracks(zero, ACQ)
        self.assertEqual(r.fits["status"][0], "invalid_input")

    def test_table_matches_direct_call(self):
        t = self.table()
        direct = analyze_track(t, ACQ)
        r = analyze_tracks(t, ACQ, keep_posteriors=True)
        self.assertEqual(dict(r.fits.schema), FIT_SCHEMA)
        row = r.fits.row(0, named=True)
        self.assertEqual(row["model"], "posterior_alpha_D")
        for name, value in direct.posterior.parameters.items():
            self.assertEqual(row[name], value)
        np.testing.assert_array_equal(r.posteriors.log_post_alpha[0], direct.log_post_alpha)
        self.assertEqual(r.posteriors.track_ids.tolist(), [7])

    def test_ellipse_moments_are_consistent(self):
        p = analyze_track(self.table(30), ACQ).posterior.parameters
        self.assertTrue(-1 <= p["alpha_log10_D_ref_corr"] <= 1)
        self.assertTrue(p["alpha_post_lo"] <= p["alpha_post_mean"] <= p["alpha_post_hi"])
        self.assertTrue(p["D_ref_post_lo_um2_s"] <= p["D_ref_post_hi_um2_s"])


if __name__ == "__main__":
    unittest.main()
