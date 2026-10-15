"""The fBm (alpha, D_ref) profile fit: its covariance, likelihood, profile over K, intervals, pooling and workflow."""
import unittest

import numpy as np
import polars as pl
from scipy.stats import chi2, multivariate_normal

from diffusionkit import Acquisition
from diffusionkit.fbmpost import FbmPostOptions, analyze_track, analyze_tracks, shared_alpha, simulate_tracks
from diffusionkit.fbmpost import covariance as R
from diffusionkit.fbmpost import likelihood as L
from diffusionkit.fbmpost import profile as P
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


def exact_tracks(T, n, alpha, D_ref, sd, rng, tau_ref=.1):
    """Tracks drawn from the likelihood's own model: blurred fGn steps (closed form) plus localization noise."""
    m = n - 1
    C = D_ref * tau_ref ** (1 - alpha) * R.fgn_motion_covariance(m, ACQ.dt_s, alpha, ACQ.exposure_s)
    steps = np.einsum("ij,tjk->tik", np.linalg.cholesky(C), rng.standard_normal((T, m, 2)))
    xy = np.concatenate([np.zeros((T, 1, 2)), np.cumsum(steps, axis=1)], axis=1) + sd * rng.standard_normal((T, n, 2))
    return pl.DataFrame({"track_id": np.repeat(np.arange(T), n), "frame": np.tile(np.arange(n), T),
                         "x_um": xy[..., 0].ravel(), "y_um": xy[..., 1].ravel(),
                         "sigma_x_um": np.full(T * n, sd), "sigma_y_um": np.full(T * n, sd)})


class LikelihoodTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.track = simulate_tracks(1, 12, .6, .02, .015, ACQ, rng=rng).with_columns(
            sigma_x_um=pl.Series(rng.uniform(.01, .03, 12)))
        self.options = FbmPostOptions()
        self.x = np.linspace(-4, 0, 9)
        self.ll = L.log_likelihood_surface(self.track, ACQ, self.options, self.x)

    def test_matches_direct_gaussian(self):
        """The per-alpha whitening is the exact Gaussian density, per-frame SDs and blur included."""
        xy = self.track.select("x_um", "y_um").to_numpy()
        B = localization_covariance(self.track.select("sigma_x_um", "sigma_y_um").to_numpy())
        for ia, ix in [(3, 1), (49, 4), (90, 8)]:
            a = self.options.alphas()[ia]
            K = 10 ** self.x[ix] * self.options.tau_ref_s ** (1 - a)
            C = R.fgn_motion_covariance(11, ACQ.dt_s, a, ACQ.exposure_s)
            direct = sum(multivariate_normal(np.zeros(11), K * C + B[k]).logpdf(np.diff(xy[:, k])) for k in range(2))
            self.assertAlmostEqual(self.ll[ia, ix], direct, places=8)

    def test_alpha_one_row_is_the_brownian_likelihood(self):
        """At alpha = 1, K and D_ref are the Brownian D, and the row is gridpost's likelihood."""
        i = int(np.argmin(np.abs(self.options.alphas() - 1)))
        self.assertAlmostEqual(self.options.alphas()[i], 1.)
        for ix in (0, 4, 8):
            self.assertAlmostEqual(self.ll[i, ix], brownian_log_likelihood(self.track, ACQ, 10 ** self.x[ix]), places=8)

    def test_chunking_does_not_change_the_surface(self):
        old = L._CHUNK_FLOATS
        try:
            L._CHUNK_FLOATS = 1
            np.testing.assert_allclose(L.log_likelihood_surface(self.track, ACQ, self.options, self.x), self.ll,
                                       rtol=0, atol=1e-9)
        finally:
            L._CHUNK_FLOATS = old


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        cls.options = FbmPostOptions(n_alpha=25)
        cls.alphas = cls.options.alphas()
        tables = [exact_tracks(3, n, a, d, .015, rng) for n, a, d in [(8, .5, .003), (30, 1.3, .03), (15, .3, 3e-4)]]
        cls.w = [L.whiten_track(t.filter(pl.col("track_id") == i), ACQ, cls.alphas) for t in tables for i in range(3)]

    def dense(self, w, n=4001):
        """The log-likelihood on a fine ln K grid around the profile's range, (n_a, n)."""
        u = np.linspace(-30, 15, n)
        return L.loglik_points(w, np.broadcast_to(u, (1, len(self.alphas), n)))[0], u

    def test_profile_is_the_maximum_over_K(self):
        for w in self.w:
            p = P.profile_K(w)
            ll, _ = self.dense(w)
            best = np.maximum(ll.max(axis=1), p.zero[0])
            self.assertTrue((p.loglik[0] >= best - 1e-9).all())  # Newton beats the dense grid
            self.assertTrue((p.loglik[0] <= best + 1e-3).all())  # and only by the grid's resolution

    def test_D_ref_interval_is_the_projection_of_the_likelihood_region(self):
        level = .95
        for w in self.w:
            s = P.summarize(w, self.alphas, len(self.alphas), 12, level, .1)
            ll, u = self.dense(w)
            inside = ll >= s.loglik_max[0] - chi2.ppf(level, 1) / 2
            x = (u[None, :] + (self.alphas[:, None] - 1) * np.log(.1)) / np.log(10)
            self.assertAlmostEqual(x[inside].max(), s.log10_D_ref_hi[0], delta=.01)
            if np.isfinite(s.log10_D_ref_lo[0]):
                self.assertAlmostEqual(x[inside].min(), s.log10_D_ref_lo[0], delta=.01)
            else:
                self.assertTrue(inside[:, 0].any())  # the region reaches K -> 0

    def test_immobile_tracks_bound_D_ref_from_above_only(self):
        tracks = simulate_tracks(40, 20, 1., 0., .015, ACQ, rng=np.random.default_rng(1))
        fits = analyze_tracks(tracks, ACQ).fits
        self.assertTrue((fits["alpha_edge"] == "both").all())
        self.assertGreater(fits["alpha_hat"].null_count(), 10)
        self.assertTrue((fits["D_ref_lo_um2_s"] < 1e-4).all())
        self.assertTrue((fits["D_ref_hi_um2_s"] < .1).all())

    def test_tau_ref_changes_D_ref_not_alpha(self):
        track = simulate_tracks(1, 30, .5, .02, .01, ACQ, rng=np.random.default_rng(2))
        a = analyze_track(track, ACQ, FbmPostOptions(tau_ref_s=.05)).fit.parameters
        b = analyze_track(track, ACQ, FbmPostOptions(tau_ref_s=.5)).fit.parameters
        for name in ("alpha_hat", "alpha_lo", "alpha_hi", "alpha_dll_brownian", "K_hat", "loglik_max"):
            self.assertAlmostEqual(a[name], b[name], places=8)
        self.assertGreater(a["D_ref_hat_um2_s"], b["D_ref_hat_um2_s"])  # alpha < 1: D falls with lag

    def test_intervals_cover(self):
        rng = np.random.default_rng(3)
        for alpha in (.4, 1., 1.5):
            fits = analyze_tracks(exact_tracks(60, 40, alpha, .03, .01, rng), ACQ).fits
            a_cover = ((fits["alpha_lo"] <= alpha) & (alpha <= fits["alpha_hi"])).mean()
            d_cover = ((fits["D_ref_lo_um2_s"] <= .03) & (.03 <= fits["D_ref_hi_um2_s"])).mean()
            self.assertGreater(a_cover, .88, alpha)
            self.assertGreater(d_cover, .88, alpha)
            self.assertLess(abs(fits["alpha_hat"].median() - alpha), .06, alpha)

    def test_brownian_test(self):
        rng = np.random.default_rng(4)
        brownian = analyze_tracks(exact_tracks(200, 30, 1., .03, .01, rng), ACQ).fits
        self.assertLess((brownian["alpha_p_brownian"] < .05).mean(), .09)
        sub = analyze_tracks(exact_tracks(40, 30, .4, .03, .01, rng), ACQ).fits
        self.assertGreater((sub["alpha_p_brownian"] < .05).mean(), .6)  # 0.73 at this seed
        np.testing.assert_allclose(brownian["alpha_p_brownian"], chi2.sf(2 * brownian["alpha_dll_brownian"], 1))


class SharedAlphaTests(unittest.TestCase):
    def test_pooled_alpha_is_unbiased_and_covers(self):
        rng = np.random.default_rng(6)
        for alpha in (.4, 1.):
            r = analyze_tracks(exact_tracks(150, 20, alpha, .01, .015, rng), ACQ, keep_profiles=True)
            s = shared_alpha(r.profiles)
            self.assertLess(s.alpha_lo, alpha)
            self.assertGreater(s.alpha_hi, alpha)
            self.assertLess(s.alpha_hi - s.alpha_lo, .15)
            self.assertEqual(s.n_tracks, 150)
            self.assertFalse(s.at_grid_edge)
            if alpha == .4:
                self.assertLess(s.p_brownian, 1e-6)

    def test_one_track_reads_its_own_interval(self):
        r = analyze_tracks(exact_tracks(1, 40, .7, .03, .01, np.random.default_rng(7)), ACQ, keep_profiles=True)
        s, row = shared_alpha(r.profiles), r.fits.row(0, named=True)
        self.assertAlmostEqual(s.alpha_hat, row["alpha_hat"], delta=.01)
        self.assertAlmostEqual(s.alpha_lo, row["alpha_lo"], delta=.01)
        self.assertAlmostEqual(s.alpha_hi, row["alpha_hi"], delta=.01)


class WorkflowTests(unittest.TestCase):
    def table(self, n=8):
        return simulate_tracks(1, n, .7, .02, .015, ACQ, rng=np.random.default_rng(4), first_track_id=7)

    def test_short_track_is_excluded(self):
        out = analyze_track(self.table(2), ACQ)
        self.assertEqual(out.fit.status, "excluded")
        self.assertIsNone(out.profile)

    def test_zero_localization_sd_gives_invalid_input(self):
        zero = self.table().with_columns(pl.Series("sigma_x_um", np.zeros(8)))
        r = analyze_tracks(zero, ACQ)
        self.assertEqual(r.fits["status"][0], "invalid_input")

    def test_table_matches_direct_call(self):
        t = self.table()
        direct = analyze_track(t, ACQ)
        r = analyze_tracks(t, ACQ, keep_profiles=True)
        self.assertEqual(dict(r.fits.schema), FIT_SCHEMA)
        row = r.fits.row(0, named=True)
        self.assertEqual(row["model"], "fbm_alpha_D")
        for name, value in direct.fit.parameters.items():
            self.assertEqual(row[name], value)
        np.testing.assert_array_equal(r.profiles.loglik[0], direct.profile)
        self.assertEqual(r.profiles.track_ids.tolist(), [7])

    def test_estimates_sit_inside_their_intervals(self):
        p = analyze_track(self.table(30), ACQ).fit.parameters
        self.assertTrue(p["alpha_lo"] <= p["alpha_hat"] <= p["alpha_hi"])
        self.assertTrue(p["D_ref_lo_um2_s"] <= p["D_ref_hat_um2_s"] <= p["D_ref_hi_um2_s"])
        self.assertAlmostEqual(np.log10(p["D_ref_hat_um2_s"]), p["log10_D_ref_hat"])

    def test_alpha_off_the_grid_still_gets_a_brownian_test(self):
        options = FbmPostOptions(n_alpha=20)  # 1 is not a grid point
        self.assertFalse(np.isclose(options.alphas(), 1).any())
        p = analyze_track(self.table(30), ACQ, options).fit.parameters
        q = analyze_track(self.table(30), ACQ).fit.parameters
        self.assertAlmostEqual(p["alpha_dll_brownian"], q["alpha_dll_brownian"], delta=.02)


if __name__ == "__main__":
    unittest.main()
