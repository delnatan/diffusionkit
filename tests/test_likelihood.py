"""Brownian displacement MLE and non-Brownian score against independent dense calculations."""
from dataclasses import replace
import unittest

import numpy as np
import polars as pl
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal

from diffusionkit import Acquisition, Track
from diffusionkit.classic import (
    MLEOptions, MSDOptions, analyze_track, analyze_tracks, brownian_log_likelihood,
    fit_brownian_mle, nonbrownian_score,
)
from diffusionkit.classic.likelihood import motion_kernels


DT = .033


def brownian(n=8, D=.05, seed=1, sd=None, track_id=3):
    rng = np.random.default_rng(seed)
    sd = rng.uniform(.01, .04, (n, 2)) if sd is None else sd
    pos = np.cumsum(rng.normal(size=(n, 2))*np.sqrt(2*D*DT), axis=0) + rng.normal(size=(n, 2))*sd
    return Track(track_id, np.arange(n), pos, sd)


def dense_fbm_ll(t, K, alpha):
    # Independent oracle: dense fBm + localization covariance, SciPy density.
    m = len(t.frames) - 1
    k = np.abs(np.subtract.outer(np.arange(m), np.arange(m)))
    G = K*DT**alpha*(np.abs(k+1)**alpha + np.abs(k-1)**alpha - 2*k**alpha)
    d = np.diff(t.positions_um, axis=0)
    total = 0.
    for a in (0, 1):
        v = t.localization_sd_um[:, a]**2
        B = np.diag(v[:-1] + v[1:]) - np.diag(v[1:-1], 1) - np.diag(v[1:-1], -1)
        total += multivariate_normal(np.zeros(m), G + B).logpdf(d[:, a])
    return total


class LikelihoodTests(unittest.TestCase):
    def test_log_likelihood_matches_dense_density(self):
        t = brownian()
        for D in (0., .01, .3):
            self.assertAlmostEqual(brownian_log_likelihood(t, Acquisition(DT), D), dense_fbm_ll(t, D, 1.), places=9)

    def test_score_and_information_match_independent_derivatives(self):
        t, D, h = brownian(), .07, 1e-5
        s = nonbrownian_score(t, Acquisition(DT), D)
        fd_alpha = (dense_fbm_ll(t, D, 1+h) - dense_fbm_ll(t, D, 1-h))/(2*h)
        fd_D = (dense_fbm_ll(t, D+1e-7, 1.) - dense_fbm_ll(t, D-1e-7, 1.))/2e-7
        m = len(t.frames) - 1
        A, H = motion_kernels(m, DT, 0.)
        I = np.zeros((2, 2))
        for a in (0, 1):
            v = t.localization_sd_um[:, a]**2
            Si = np.linalg.inv(D*A + np.diag(v[:-1] + v[1:]) - np.diag(v[1:-1], 1) - np.diag(v[1:-1], -1))
            der = (A, D*H)
            I += [[.5*np.trace(Si@der[i]@Si@der[j]) for j in (0, 1)] for i in (0, 1)]
        self.assertAlmostEqual(s["I_eff"], I[1, 1] - I[0, 1]**2/I[0, 0], places=10)
        self.assertAlmostEqual(s["U"], fd_alpha - I[0, 1]/I[0, 0]*fd_D, places=5)
        # At the MLE, U_D = 0 and the efficient score is the plain alpha score.
        D_hat = fit_brownian_mle(t, Acquisition(DT), MLEOptions(n_boot=0)).parameters["D_um2_s"]
        at_mle = nonbrownian_score(t, Acquisition(DT), D_hat)["U"]
        self.assertAlmostEqual(at_mle, (dense_fbm_ll(t, D_hat, 1+h) - dense_fbm_ll(t, D_hat, 1-h))/(2*h), places=5)
        self.assertAlmostEqual(s["alpha_1step"], 1 + s["U"]/s["I_eff"])

    def test_score_is_invariant_to_time_units(self):
        # The ln(dt) part of d/d(alpha) lies along the D direction and is projected out.
        t = brownian()
        a = nonbrownian_score(t, Acquisition(DT), .05)
        b = nonbrownian_score(replace(t), Acquisition(DT*1000), .05/1000)
        self.assertAlmostEqual(a["z_asymptotic"], b["z_asymptotic"], places=8)

    def test_blur_kernels_match_double_integral(self):
        te, m = .02, 4
        A, H = motion_kernels(m, DT, te)
        R = te/(6*DT)
        self.assertAlmostEqual(A[0, 0], 2*DT*(1-2*R))
        self.assertAlmostEqual(A[0, 1], 2*DT*R)
        self.assertAlmostEqual(A[0, 2], 0.)

        def cov(alpha, i, j):
            c = lambda s, u: abs(s)**alpha + abs(u)**alpha - abs(s-u)**alpha  # noqa: E731
            p = lambda ti, tj: dblquad(lambda u, s: c(s, u), ti-te, ti, tj-te, tj, epsabs=1e-13)[0]/te**2  # noqa: E731
            T = lambda k: (k+1)*DT  # noqa: E731
            return p(T(i+1), T(j+1)) - p(T(i+1), T(j)) - p(T(i), T(j+1)) + p(T(i), T(j))
        for j in (0, 1, 2):
            with self.subTest(lag=j):
                self.assertAlmostEqual(H[0, j], (cov(1+1e-4, 0, j) - cov(1-1e-4, 0, j))/2e-4, places=7)
        A0, H0 = motion_kernels(m, DT, 0.)
        A1, H1 = motion_kernels(m, DT, 1e-9)
        np.testing.assert_allclose(A1, A0, atol=1e-8)
        np.testing.assert_allclose(H1, H0, atol=1e-7)

    def test_mle_is_the_constrained_maximum(self):
        for seed in range(5):
            t = brownian(n=6, seed=seed)
            fit = fit_brownian_mle(t, Acquisition(DT), MLEOptions(n_boot=0))
            D = fit.parameters["D_um2_s"]
            grid = np.concatenate([[0.], np.logspace(-6, 1, 400)])
            best = max(brownian_log_likelihood(t, Acquisition(DT), g) for g in grid)
            with self.subTest(seed=seed):
                self.assertGreaterEqual(fit.parameters["log_likelihood"], best - 1e-9)
                if D > 0:
                    for x in (D*(1-1e-4), D*(1+1e-4)):
                        self.assertLessEqual(brownian_log_likelihood(t, Acquisition(DT), x),
                                             fit.parameters["log_likelihood"])

    def test_pure_noise_is_unresolved_at_zero(self):
        rng = np.random.default_rng(5)
        sd = np.full((6, 2), .05)
        t = Track(1, np.arange(6), np.zeros((6, 2)) + rng.normal(size=(6, 2))*.001, sd)
        fit = fit_brownian_mle(t, Acquisition(DT))
        self.assertEqual(fit.status, "unresolved")
        self.assertEqual(fit.parameters["D_um2_s"], 0.)
        self.assertEqual(fit.parameters["lr_motion"], 0.)
        self.assertIsNone(fit.parameters["z_nonbrownian"])
        self.assertGreater(fit.parameters["D_upper_um2_s"], 0.)
        with self.assertRaises(ValueError):
            nonbrownian_score(t, Acquisition(DT), 0.)

    def test_long_track_recovers_D(self):
        t = brownian(n=4000, D=.05, seed=9, sd=np.full((4000, 2), .02))
        fit = fit_brownian_mle(t, Acquisition(DT), MLEOptions(n_boot=0))
        self.assertLess(abs(fit.parameters["D_um2_s"] - .05), .005)
        self.assertLess(abs(fit.parameters["alpha_1step"] - 1), 4*fit.parameters["alpha_1step_se"])

    def test_error_placement_and_zero_errors(self):
        t = brownian()
        moved = replace(t, localization_sd_um=np.roll(t.localization_sd_um, 1, axis=0))
        self.assertNotAlmostEqual(brownian_log_likelihood(t, Acquisition(DT), .05),
                                  brownian_log_likelihood(moved, Acquisition(DT), .05))
        zero = t.localization_sd_um.copy()
        zero[2, 0] = 0.
        with self.assertRaisesRegex(ValueError, "positive"):
            fit_brownian_mle(replace(t, localization_sd_um=zero), Acquisition(DT))
        self.assertEqual(analyze_track(replace(t, localization_sd_um=zero), Acquisition(DT)).brownian_mle.status,
                         "invalid_input")

    def test_bootstrap_is_reproducible_and_order_independent(self):
        tracks = [brownian(seed=s, track_id=s) for s in (1, 2)]
        acq = Acquisition(DT)
        a = fit_brownian_mle(tracks[0], acq, MLEOptions(n_boot=200, seed=4))
        b = fit_brownian_mle(tracks[0], acq, MLEOptions(n_boot=200, seed=4))
        self.assertEqual(a, b)
        rows = [pl.DataFrame({"track_id": t.track_id, "frame": t.frames, "x_um": t.positions_um[:, 0],
                              "y_um": t.positions_um[:, 1], "sigma_x_um": t.localization_sd_um[:, 0],
                              "sigma_y_um": t.localization_sd_um[:, 1]}) for t in tracks]
        opts = MLEOptions(n_boot=200, seed=4)
        forward = analyze_tracks(pl.concat(rows), acq, mle_options=opts).fits
        backward = analyze_tracks(pl.concat(rows[::-1]), acq, mle_options=opts).fits
        key = pl.col("model") == "brownian_mle"
        self.assertTrue(forward.filter(key).sort("track_id").equals(backward.filter(key).sort("track_id")))

    def test_ignore_localization_excludes_mle(self):
        out = analyze_track(brownian(), Acquisition(DT), MSDOptions(localization="ignore"))
        self.assertEqual(out.brownian_mle.status, "excluded")

    def test_invalid_options(self):
        for opts in (MLEOptions(n_boot=-1), MLEOptions(seed=-1), MLEOptions(upper_level=.4),
                     MLEOptions(n_boot=1.5)):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                fit_brownian_mle(brownian(), Acquisition(DT), opts)

    def test_null_calibration_smoke(self):
        # Short Brownian tracks: calibrated z should be roughly standard normal.
        acq, z = Acquisition(DT), []
        for seed in range(150):
            fit = fit_brownian_mle(brownian(n=6, D=.1, seed=100+seed, track_id=seed), acq, MLEOptions(n_boot=200))
            if fit.parameters["z_nonbrownian"] is not None:
                z.append(fit.parameters["z_nonbrownian"])
        z = np.array(z)
        self.assertGreater(len(z), 100)
        self.assertLess(abs(z.mean()), 4/np.sqrt(len(z)))
        self.assertLess(abs(z.std() - 1), .2)


if __name__ == "__main__":
    unittest.main()
