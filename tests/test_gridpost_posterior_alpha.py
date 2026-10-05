"""Grid posterior over alpha (diffusionkit.gridpost.posterior_alpha) against simulation."""
import unittest

import numpy as np
import polars as pl
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions
from diffusionkit.gridpost import posterior as PD
from diffusionkit.gridpost import posterior_alpha as PA
from diffusionkit.gridpost.likelihood import fgn_motion_covariance, localization_covariance

DT = .033
OPTIONS = GridPostOptions()
ALPHA, U_D = OPTIONS.alphas(), OPTIONS.u_D()


def track_table(track_id, frames, positions, sd):
    return pl.DataFrame({
        "track_id": [track_id] * len(frames), "frame": frames,
        "x_um": positions[:, 0], "y_um": positions[:, 1],
        "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1],
    })


def simulate(K, alpha, sd, dt, rng, n_frames=None, exposure=0.):
    """(n, 2) measured positions: exact (exposure-averaged) fGn increments (Cholesky) + localization noise.

    The blurred covariance is checked against a fine-step fBm simulation in
    test_gridpost_likelihood, independently of the posterior code."""
    n = len(sd) if n_frames is None else n_frames
    L = np.linalg.cholesky(K * fgn_motion_covariance(n - 1, dt, alpha, exposure))
    disp = np.stack([L @ rng.standard_normal(n - 1) for _ in range(2)], axis=1)
    true = np.vstack([np.zeros(2), np.cumsum(disp, axis=0)])
    return true + sd[:, None] * rng.standard_normal((n, 2))


def K_per_D(alpha, exposure=0.):
    """K per unit apparent D: Berglund's blurred Brownian step variance, 2 (dt - te/3) per axis,
    over the fBm one per unit K. dt^(1-alpha) without blur."""
    return 2 * (DT - exposure / 3) / fgn_motion_covariance(1, DT, alpha, exposure)[0, 0]


def dense_loglik(track, alpha, D, exposure=0.):
    """Independent oracle: dense per-axis fGn + localization covariance, SciPy density.

    D is the apparent diffusivity, the Brownian D with the same blurred step variance."""
    delta = np.diff(track.select("x_um", "y_um").to_numpy(), axis=0)
    sd = track.select("sigma_x_um", "sigma_y_um").to_numpy()
    m = delta.shape[0]
    A = D * K_per_D(alpha, exposure) * fgn_motion_covariance(m, DT, alpha, exposure)
    return sum(multivariate_normal(np.zeros(m), A + B).logpdf(delta[:, axis])
              for axis, B in enumerate(localization_covariance(sd)))


class PosteriorAlphaTests(unittest.TestCase):
    def test_matches_direct_gaussian(self):
        rng = np.random.default_rng(1)
        n = 7
        sd = rng.uniform(.02, .05, (n, 2))
        for alpha in (0.4, 1.0, 1.6):
            pos = simulate(.05, alpha, sd[:, 0], DT, rng, n_frames=n)
            t = track_table(1, np.arange(n), pos, sd)
            ll = PA.track_loglik_given_alpha(t, Acquisition(DT), alpha, U_D[::5])
            direct = np.array([dense_loglik(t, alpha, D) for D in np.exp(U_D[::5])])
            np.testing.assert_allclose(ll, direct, atol=1e-8)

    def test_alpha_one_reduces_to_the_D_posterior(self):
        """No exposure blur, alpha=1: the fGn model is exactly the Brownian D-posterior model."""
        rng = np.random.default_rng(2)
        n = 6
        sd = rng.uniform(.02, .05, (n, 2))
        pos = simulate(.05, 1.0, sd[:, 0], DT, rng, n_frames=n)
        t = track_table(1, np.arange(n), pos, sd)
        np.testing.assert_allclose(
            PA.track_loglik_given_alpha(t, Acquisition(DT), 1.0, U_D),
            PD.track_loglik(t, Acquisition(DT), U_D), atol=1e-10)

    def test_credible_intervals_are_calibrated(self):
        """Truth drawn from flat priors, data simulated independently: 90% intervals cover 90%."""
        K_true = .05
        D_prior = PD.flat(U_D)
        rng = np.random.default_rng(3)
        for n_frames in (8, 15):
            hits, n_tracks = 0, 250
            for _ in range(n_tracks):
                alpha_true = rng.uniform(ALPHA[0], ALPHA[-1])
                sd = rng.uniform(.025, .045, (n_frames, 2))
                pos = simulate(K_true, alpha_true, sd[:, 0], DT, rng, n_frames=n_frames)
                t = track_table(1, np.arange(n_frames), pos, sd)
                s = PA.track_alpha_posterior(t, Acquisition(DT), D_prior, GridPostOptions(level=.9))
                hits += s["lo"] <= alpha_true <= s["hi"]
            self.assertLess(abs(hits/n_tracks - .9), .08)

    def test_short_track_posterior_is_wide(self):
        """A 5-frame track's alpha posterior stays wide (honest), not falsely confident."""
        rng = np.random.default_rng(4)
        n = 5
        sd = rng.uniform(.03, .045, (n, 2))
        pos = simulate(.05, 1.0, sd[:, 0], DT, rng, n_frames=n)
        t = track_table(1, np.arange(n), pos, sd)
        s = PA.track_alpha_posterior(t, Acquisition(DT), PD.flat(U_D))
        self.assertGreater(s["hi"] - s["lo"], 0.6 * (ALPHA[-1] - ALPHA[0]))

    def test_alpha_posterior_does_not_depend_on_the_unit_of_time(self):
        """The same track, grid and prior in seconds or in milliseconds: the same alpha posterior.

        A nuisance grid over fBm's K (um^2/s^alpha) fails this, because what a
        fixed K range means depends on alpha and on the unit; D at dt does not.
        Slow, noise-dominated motion puts the scale near the grid floor, where
        the difference shows."""
        rng = np.random.default_rng(14)
        n, exposure = 8, .02
        sd = np.full((n, 2), .015)
        t = track_table(1, np.arange(n), simulate(.001, .5, sd[:, 0], DT, rng, n, exposure), sd)
        options = GridPostOptions(D_min_um2_s=1e-4, D_max_um2_s=10.)
        in_ms = GridPostOptions(D_min_um2_s=1e-7, D_max_um2_s=1e-2)  # um^2/ms
        for method in ("exact", "whittle"):
            with self.subTest(method=method):
                s = PA.alpha_posterior(PA.joint_loglik(t, Acquisition(DT, exposure), ALPHA, options.u_D(), method),
                                       PD.flat(options.u_D()))
                ms = PA.alpha_posterior(PA.joint_loglik(t, Acquisition(1e3 * DT, 1e3 * exposure), ALPHA,
                                                        in_ms.u_D(), method), PD.flat(in_ms.u_D()))
                np.testing.assert_allclose(ms, s, rtol=1e-7, atol=1e-12)

    def test_apparent_D_gives_every_alpha_the_brownian_blurred_step_variance(self):
        """The scale's definition: per unit D, one blurred step has Brownian's variance at any alpha."""
        for exposure in (0., .01, DT):
            for alpha in (.05, .5, 1., 1.6):
                step = (PA._scale_per_D(np.array([alpha]), DT, exposure)[0]
                        * fgn_motion_covariance(1, DT, alpha, exposure)[0, 0])
                self.assertAlmostEqual(step, 2 * (DT - exposure / 3), places=12)
        np.testing.assert_allclose(PA._scale_per_D(ALPHA, DT, 0.), DT ** (1 - ALPHA), rtol=1e-12)

    def test_joint_cells_tile_the_grid(self):
        """Summed over D the cells are the alpha posterior; at alpha=1 they bin the D likelihood."""
        rng = np.random.default_rng(15)
        n, acquisition = 9, Acquisition(DT, .02)
        sd = rng.uniform(.02, .05, (n, 2))
        t = track_table(1, np.arange(n), simulate(.05, .7, sd[:, 0], DT, rng, n, .02), sd)
        alphas = np.array([.3, 1., 1.5])
        ll = PA.joint_loglik(t, acquisition, alphas, U_D)
        joint = PA.log_joint_posterior(ll, PD.flat(U_D), 10)
        self.assertEqual(joint.shape, (3, 50))
        self.assertAlmostEqual(logsumexp(joint), 0., places=12)
        np.testing.assert_allclose(logsumexp(joint, axis=1), PA.log_alpha_posterior(ll, PD.flat(U_D)), atol=1e-12)
        d = PD.track_loglik(t, acquisition, U_D)
        cells = np.log([np.trapezoid(np.exp(d[j * 10:(j + 1) * 10 + 1] - d.max())) for j in range(50)])
        np.testing.assert_allclose(joint[1] - logsumexp(joint[1]), cells - logsumexp(cells), atol=1e-10)
        np.testing.assert_allclose(OPTIONS.u_joint_D(), (U_D[:-1:10] + U_D[10::10]) / 2)
        with self.assertRaisesRegex(ValueError, "D_bin"):
            PA.log_joint_posterior(ll, PD.flat(U_D), 7)

    def test_flat_alpha_prior_is_uniform_over_grid(self):
        np.testing.assert_array_equal(PA.flat_alpha(ALPHA), np.zeros_like(ALPHA))

    def test_blurred_matches_direct_gaussian(self):
        rng = np.random.default_rng(5)
        n, exposure = 7, .02
        sd = rng.uniform(.02, .05, (n, 2))
        for alpha in (0.4, 1.0, 1.6):
            pos = simulate(.05, alpha, sd[:, 0], DT, rng, n_frames=n, exposure=exposure)
            t = track_table(1, np.arange(n), pos, sd)
            ll = PA.track_loglik_given_alpha(t, Acquisition(DT, exposure), alpha, U_D[::5])
            direct = np.array([dense_loglik(t, alpha, D, exposure) for D in np.exp(U_D[::5])])
            np.testing.assert_allclose(ll, direct, atol=1e-8)

    def test_blurred_alpha_one_reduces_to_the_D_posterior(self):
        """With exposure blur too, alpha=1 is exactly the D posterior's (Berglund) model."""
        rng = np.random.default_rng(6)
        n, acquisition = 9, Acquisition(DT, exposure_s=.025)
        sd = rng.uniform(.02, .05, (n, 2))
        t = track_table(1, np.arange(n), simulate(.05, 1.0, sd[:, 0], DT, rng, n, .025), sd)
        np.testing.assert_allclose(PA.track_loglik_given_alpha(t, acquisition, 1.0, U_D),
                                   PD.track_loglik(t, acquisition, U_D), atol=1e-9)

    def test_joint_loglik_is_the_same_in_alpha_chunks(self):
        """A long track's alpha grid is eigendecomposed in chunks; chunking must not change the surface."""
        rng = np.random.default_rng(7)
        n = 40
        sd = rng.uniform(.02, .05, (n, 2))
        t = track_table(1, np.arange(n), simulate(.05, .7, sd[:, 0], DT, rng, n, .01), sd)
        acquisition = Acquisition(DT, .01)
        whole = PA.joint_loglik(t, acquisition, ALPHA, U_D)
        saved = PA._BATCH_ELEMENTS
        try:
            PA._BATCH_ELEMENTS = 5 * 2 * (n - 1) ** 2  # five alphas per chunk
            chunked = PA.joint_loglik(t, acquisition, ALPHA, U_D)
        finally:
            PA._BATCH_ELEMENTS = saved
        np.testing.assert_allclose(chunked, whole, rtol=0, atol=1e-9)
        by_alpha = np.array([PA.track_loglik_given_alpha(t, acquisition, a, U_D) for a in ALPHA[::10]])
        np.testing.assert_allclose(whole[::10], by_alpha, rtol=0, atol=1e-9)

    def test_blurred_credible_intervals_are_calibrated(self):
        """Blurred simulation, blur modelled: 90% intervals cover 90%."""
        D_prior, exposure = PD.flat(U_D), .02
        rng = np.random.default_rng(8)
        hits, n_tracks, n_frames = 0, 250, 10
        for _ in range(n_tracks):
            alpha_true = rng.uniform(ALPHA[0], ALPHA[-1])
            sd = rng.uniform(.025, .045, (n_frames, 2))
            pos = simulate(.05, alpha_true, sd[:, 0], DT, rng, n_frames, exposure)
            t = track_table(1, np.arange(n_frames), pos, sd)
            s = PA.track_alpha_posterior(t, Acquisition(DT, exposure), D_prior)
            hits += s["lo"] <= alpha_true <= s["hi"]
        self.assertLess(abs(hits/n_tracks - .9), .08)

    def test_ignoring_blur_biases_alpha_up(self):
        """Why exposure matters: blur correlates neighbouring steps positively, which an
        unblurred model reads as a larger alpha (0.5 -> ~0.7 at 20 ms of a 33 ms frame)."""
        rng = np.random.default_rng(9)
        exposure, n = .02, 30
        modelled, ignored = [], []
        for _ in range(60):
            sd = rng.uniform(.02, .035, (n, 2))
            t = track_table(1, np.arange(n), simulate(.1, .5, sd[:, 0], DT, rng, n, exposure), sd)
            modelled.append(PA.track_alpha_posterior(t, Acquisition(DT, exposure))["median"])
            ignored.append(PA.track_alpha_posterior(t, Acquisition(DT))["median"])
        self.assertLess(abs(np.median(modelled) - .5), .06)
        self.assertGreater(np.median(ignored) - np.median(modelled), .12)

    def test_whittle_matches_a_dense_expected_periodogram(self):
        """The FFT-built debiased Whittle surface against its definition, evaluated densely:
        S(w) = f(w)^H (D K_per_D A + B) f(w) / m, f(w)_t = exp(-i w t)."""
        rng = np.random.default_rng(10)
        n, exposure = 12, .02
        sd = rng.uniform(.02, .05, (n, 2))
        t = track_table(1, np.arange(n), simulate(.05, .8, sd[:, 0], DT, rng, n, exposure), sd)
        acquisition, alphas, u = Acquisition(DT, exposure), np.array([.4, 1.3]), U_D[::100]
        ours = PA.joint_loglik(t, acquisition, alphas, u, method="whittle")
        delta = np.diff(t.select("x_um", "y_um").to_numpy(), axis=0).T
        m = n - 1
        F = np.exp(-2j * np.pi * np.outer(np.arange(m), np.arange(m)) / m)  # rows f(w_j)
        I = np.abs(F @ delta.T).T ** 2 / m
        B = localization_covariance(sd)
        for i, alpha in enumerate(alphas):
            A = K_per_D(alpha, exposure) * fgn_motion_covariance(m, DT, alpha, exposure)
            for k, D in enumerate(np.exp(u)):
                S = np.stack([np.einsum("ji,il,jl->j", F.conj(), D * A + B[axis], F).real / m for axis in range(2)])
                direct = -.5 * np.sum(np.log(S) + I / S) - m * np.log(2 * np.pi)
                self.assertAlmostEqual(ours[i, k], direct, places=8)

    def test_whittle_agrees_with_exact_on_a_long_track(self):
        """Constant localization SDs, 150 frames: the two posteriors nearly coincide."""
        rng = np.random.default_rng(11)
        n, exposure = 150, .02
        sd = np.full((n, 2), .03)
        t = track_table(1, np.arange(n), simulate(.05, .7, sd[:, 0], DT, rng, n, exposure), sd)
        acquisition = Acquisition(DT, exposure)
        exact = PA.track_alpha_posterior(t, acquisition, options=GridPostOptions(alpha_method="exact"))
        whittle = PA.track_alpha_posterior(t, acquisition, options=GridPostOptions(alpha_method="whittle"))
        self.assertLess(abs(exact["median"] - whittle["median"]), .1 * (exact["hi"] - exact["lo"]))
        self.assertLess(abs((whittle["hi"] - whittle["lo"]) / (exact["hi"] - exact["lo"]) - 1), .1)

    def test_whittle_credible_intervals_are_calibrated(self):
        """Blurred simulation, per-frame SDs that vary: Whittle's 90% intervals still cover 90%."""
        options, exposure = GridPostOptions(alpha_method="whittle"), .02
        rng = np.random.default_rng(12)
        hits, n_tracks, n_frames = 0, 250, 60
        for _ in range(n_tracks):
            alpha_true = rng.uniform(ALPHA[0], ALPHA[-1])
            sd = rng.uniform(.015, .05, (n_frames, 2))
            pos = simulate(.05, alpha_true, sd[:, 0], DT, rng, n_frames, exposure)
            s = PA.track_alpha_posterior(track_table(1, np.arange(n_frames), pos, sd), Acquisition(DT, exposure),
                                         options=options)
            hits += s["lo"] <= alpha_true <= s["hi"]
        self.assertLess(abs(hits/n_tracks - .9), .06)

    def test_whittle_is_the_same_in_alpha_chunks(self):
        rng = np.random.default_rng(13)
        n = 40
        sd = rng.uniform(.02, .05, (n, 2))
        t = track_table(1, np.arange(n), simulate(.05, 1.2, sd[:, 0], DT, rng, n, .01), sd)
        acquisition = Acquisition(DT, .01)
        whole = PA.joint_loglik(t, acquisition, ALPHA, U_D, method="whittle")
        saved = PA._BATCH_ELEMENTS
        try:
            PA._BATCH_ELEMENTS = 7 * len(U_D) * 2 * (n - 1)  # seven alphas per chunk
            chunked = PA.joint_loglik(t, acquisition, ALPHA, U_D, method="whittle")
        finally:
            PA._BATCH_ELEMENTS = saved
        np.testing.assert_allclose(chunked, whole, rtol=0, atol=1e-9)

    def test_auto_picks_the_likelihood_by_length(self):
        options = GridPostOptions(alpha_whittle_min_frames=40)
        self.assertEqual(options.alpha_likelihood(39), "exact")
        self.assertEqual(options.alpha_likelihood(40), "whittle")
        self.assertEqual(GridPostOptions(alpha_method="exact").alpha_likelihood(500), "exact")
        self.assertEqual(GridPostOptions(alpha_method="whittle").alpha_likelihood(5), "whittle")
        with self.assertRaisesRegex(ValueError, "alpha_method"):
            GridPostOptions(alpha_method="fast")
        with self.assertRaisesRegex(ValueError, "method"):
            PA.joint_loglik(track_table(1, np.arange(5), np.zeros((5, 2)), np.full((5, 2), .03)),
                            Acquisition(DT), ALPHA, U_D, method="fast")

if __name__ == "__main__":
    unittest.main()
