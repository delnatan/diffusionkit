"""Shared whitening plumbing against independent dense calculations."""
import unittest

import numpy as np
import polars as pl
from scipy.stats import multivariate_normal

from diffusionkit import Acquisition
from diffusionkit.gridpost import brownian_log_likelihood
from diffusionkit.gridpost.likelihood import (
    _BLUR_SERIES_RATIO, _blurred_abs_power, fgn_motion_covariance, motion_covariance)


DT = .033


def track_table(track_id, frames, positions, sd):
    return pl.DataFrame({
        "track_id": [track_id] * len(frames), "frame": frames,
        "x_um": positions[:, 0], "y_um": positions[:, 1],
        "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1],
    })


def brownian(n=8, D=.05, seed=1, sd=None, track_id=3):
    rng = np.random.default_rng(seed)
    sd = rng.uniform(.01, .04, (n, 2)) if sd is None else sd
    pos = np.cumsum(rng.normal(size=(n, 2))*np.sqrt(2*D*DT), axis=0) + rng.normal(size=(n, 2))*sd
    return track_table(track_id, np.arange(n), pos, sd)


def dense_brownian_ll(t, D):
    # Independent oracle: dense Brownian + localization covariance, SciPy density.
    m = t.height - 1
    A = D*2*DT*np.eye(m)
    d = np.diff(t.select("x_um", "y_um").to_numpy(), axis=0)
    sd = t.select("sigma_x_um", "sigma_y_um").to_numpy()
    total = 0.
    for a in (0, 1):
        v = sd[:, a]**2
        B = np.diag(v[:-1] + v[1:]) - np.diag(v[1:-1], 1) - np.diag(v[1:-1], -1)
        total += multivariate_normal(np.zeros(m), A + B).logpdf(d[:, a])
    return total


class LikelihoodTests(unittest.TestCase):
    def test_log_likelihood_matches_dense_density(self):
        t = brownian()
        for D in (0., .01, .3):
            self.assertAlmostEqual(brownian_log_likelihood(t, Acquisition(DT), D), dense_brownian_ll(t, D), places=9)

    def test_error_placement_matters(self):
        t = brownian()
        sd = t.select("sigma_x_um", "sigma_y_um").to_numpy()
        moved = t.with_columns(pl.Series("sigma_x_um", np.roll(sd[:, 0], 1)),
                               pl.Series("sigma_y_um", np.roll(sd[:, 1], 1)))
        self.assertNotAlmostEqual(brownian_log_likelihood(t, Acquisition(DT), .05),
                                  brownian_log_likelihood(moved, Acquisition(DT), .05))

    def test_zero_localization_sd_rejected(self):
        t = brownian()
        sd = t.select("sigma_x_um", "sigma_y_um").to_numpy().copy()
        sd[2, 0] = 0.
        zero = t.with_columns(pl.Series("sigma_x_um", sd[:, 0]))
        with self.assertRaisesRegex(ValueError, "positive"):
            brownian_log_likelihood(zero, Acquisition(DT), .05)

    def test_motion_covariance_berglund_closed_form(self):
        exposure, m = .02, 4
        A = motion_covariance(m, DT, exposure)
        R = exposure/(6*DT)
        self.assertAlmostEqual(A[0, 0], 2*DT*(1-2*R))
        self.assertAlmostEqual(A[0, 1], 2*DT*R)
        self.assertAlmostEqual(A[0, 2], 0.)
        no_blur = motion_covariance(m, DT, 0.)
        np.testing.assert_allclose(no_blur, 2*DT*np.eye(m))

    def test_motion_covariance_matches_fine_step_simulation(self):
        """Covariance of simulated displacements (no localization noise) is D * A."""
        rng = np.random.default_rng(2)
        D, dt, exposure, n, sims = .5, DT, .02, 4, 20000
        sub = 200
        h = dt/sub
        n_on = round(exposure/h)
        disp = np.empty((sims, n))
        for k in range(sims):
            path = np.concatenate([[0.], np.cumsum(rng.normal(0, np.sqrt(2*D*h), (n+1)*sub))])
            w = np.ones(n_on+1)
            w[[0, -1]] = .5
            pos = np.array([w @ path[i*sub:i*sub+n_on+1] / max(n_on, 1) if n_on else path[i*sub]
                            for i in range(n+1)])
            disp[k] = np.diff(pos)
        emp, model = np.cov(disp.T), D*motion_covariance(n, dt, exposure)
        self.assertLess(np.abs(emp-model).max(), .04*model.max())
        self.assertGreater(model[0, 1], 0.)  # blur, not localization noise, correlates neighbours positively


    def test_blurred_fgn_reduces_to_berglund_at_alpha_one(self):
        """fBm's exposure average at alpha=1 is the D posterior's Berglund closed form, at any exposure."""
        for exposure in (0., .001, .02, DT):
            np.testing.assert_allclose(fgn_motion_covariance(6, DT, 1.0, exposure),
                                       motion_covariance(6, DT, exposure), rtol=1e-12, atol=1e-15)

    def test_blurred_fgn_without_exposure_is_plain_fgn(self):
        k = np.arange(5.)
        for alpha in (.3, 1.4):
            gamma = DT**alpha * (np.abs(k + 1)**alpha - 2*k**alpha + np.abs(k - 1)**alpha)
            np.testing.assert_allclose(fgn_motion_covariance(5, DT, alpha)[0], gamma, rtol=1e-12)

    def test_blurred_fgn_matches_quadrature(self):
        """The closed form against a midpoint double integral of |x + u - v|^alpha over both exposures."""
        exposure, m, n = .02, 5, 600
        u = (np.arange(n) + .5) / n * exposure
        w = (u[:, None] - u[None, :]).ravel()
        for alpha in (.3, .8, 1.5, 1.9):
            G = lambda x: np.mean(np.abs(x + w)**alpha)  # noqa: E731
            gamma = [G((k+1)*DT) - 2*G(k*DT) + G((k-1)*DT) for k in range(m)]
            np.testing.assert_allclose(fgn_motion_covariance(m, DT, alpha, exposure)[0], gamma,
                                       rtol=1e-3, atol=1e-6 * gamma[0])

    def test_blurred_fgn_is_continuous_across_its_series_switch(self):
        """E|x + u - v|^alpha switches to its series at te/x = _BLUR_SERIES_RATIO; both sides must agree."""
        for alpha in (.05, .4, 1.3, 1.95):
            for te in (1e-4, .02):
                x = te / _BLUR_SERIES_RATIO * np.array([1 - 1e-12, 1 + 1e-12])
                below, above = _blurred_abs_power(x, alpha, te)
                self.assertLess(abs(below / above - 1), 1e-10)
        for alpha in (.4, 1.3):
            # Lag 0's blur fades like te^alpha, slowly at small alpha: 1e-30 s is invisible.
            np.testing.assert_allclose(fgn_motion_covariance(4, DT, alpha, 1e-30),
                                       fgn_motion_covariance(4, DT, alpha), rtol=1e-10)

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
            emp, model = np.cov(np.diff(pos, axis=1).T), fgn_motion_covariance(m, DT, alpha, exposure)
            self.assertLess(np.abs(emp - model).max(), .04 * model.max())

if __name__ == "__main__":
    unittest.main()
