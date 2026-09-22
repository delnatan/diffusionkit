"""Shared whitening plumbing against independent dense calculations."""
from dataclasses import replace
import unittest

import numpy as np
from scipy.stats import multivariate_normal

from diffusionkit import Acquisition, Track
from diffusionkit.classic import brownian_log_likelihood
from diffusionkit.classic.likelihood import motion_covariance


DT = .033


def brownian(n=8, D=.05, seed=1, sd=None, track_id=3):
    rng = np.random.default_rng(seed)
    sd = rng.uniform(.01, .04, (n, 2)) if sd is None else sd
    pos = np.cumsum(rng.normal(size=(n, 2))*np.sqrt(2*D*DT), axis=0) + rng.normal(size=(n, 2))*sd
    return Track(track_id, np.arange(n), pos, sd)


def dense_brownian_ll(t, D):
    # Independent oracle: dense Brownian + localization covariance, SciPy density.
    m = len(t.frames) - 1
    A = D*2*DT*np.eye(m)
    d = np.diff(t.positions_um, axis=0)
    total = 0.
    for a in (0, 1):
        v = t.localization_sd_um[:, a]**2
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
        moved = replace(t, localization_sd_um=np.roll(t.localization_sd_um, 1, axis=0))
        self.assertNotAlmostEqual(brownian_log_likelihood(t, Acquisition(DT), .05),
                                  brownian_log_likelihood(moved, Acquisition(DT), .05))

    def test_zero_localization_sd_rejected(self):
        t = brownian()
        zero = t.localization_sd_um.copy()
        zero[2, 0] = 0.
        with self.assertRaisesRegex(ValueError, "positive"):
            brownian_log_likelihood(replace(t, localization_sd_um=zero), Acquisition(DT), .05)

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


if __name__ == "__main__":
    unittest.main()
