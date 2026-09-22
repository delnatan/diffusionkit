"""Grid posterior over alpha (diffusionkit.gridpost.posterior_alpha) against simulation."""
import unittest

import numpy as np
import polars as pl
from scipy.stats import multivariate_normal

from diffusionkit import Acquisition
from diffusionkit.gridpost import posterior as PD
from diffusionkit.gridpost import posterior_alpha as PA
from diffusionkit.gridpost.likelihood import fgn_motion_covariance, localization_covariance

DT = .033


def track_table(track_id, frames, positions, sd):
    return pl.DataFrame({
        "track_id": [track_id] * len(frames), "frame": frames,
        "x_um": positions[:, 0], "y_um": positions[:, 1],
        "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1],
    })


def simulate(K, alpha, sd, dt, rng, n_frames=None):
    """(n, 2) measured positions: exact fGn increments (Cholesky) + localization noise."""
    n = len(sd) if n_frames is None else n_frames
    L = np.linalg.cholesky(K * fgn_motion_covariance(n - 1, dt, alpha))
    disp = np.stack([L @ rng.standard_normal(n - 1) for _ in range(2)], axis=1)
    true = np.vstack([np.zeros(2), np.cumsum(disp, axis=0)])
    return true + sd[:, None] * rng.standard_normal((n, 2))


def dense_loglik(track, alpha, K):
    """Independent oracle: dense per-axis fGn + localization covariance, SciPy density."""
    delta = np.diff(track.select("x_um", "y_um").to_numpy(), axis=0)
    sd = track.select("sigma_x_um", "sigma_y_um").to_numpy()
    m = delta.shape[0]
    A = K * fgn_motion_covariance(m, DT, alpha)
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
            ll = PA.track_loglik_given_alpha(t, Acquisition(DT), alpha)
            direct = np.array([dense_loglik(t, alpha, K) for K in np.exp(PA.U)])
            np.testing.assert_allclose(ll, direct, atol=1e-8)

    def test_alpha_one_reduces_to_the_D_posterior(self):
        """No exposure blur, alpha=1: the fGn model is exactly the Brownian D-posterior model."""
        rng = np.random.default_rng(2)
        n = 6
        sd = rng.uniform(.02, .05, (n, 2))
        pos = simulate(.05, 1.0, sd[:, 0], DT, rng, n_frames=n)
        t = track_table(1, np.arange(n), pos, sd)
        np.testing.assert_allclose(
            PA.track_loglik_given_alpha(t, Acquisition(DT), 1.0, u=PD.U),
            PD.track_loglik(t, Acquisition(DT)), atol=1e-10)

    def test_credible_intervals_are_calibrated(self):
        """Truth drawn from flat priors, data simulated independently: 90% intervals cover 90%."""
        K_true = .05
        K_prior = PA.flat_K(PA.U)
        rng = np.random.default_rng(3)
        for n_frames in (8, 15):
            hits, n_tracks = 0, 250
            for _ in range(n_tracks):
                alpha_true = rng.uniform(PA.ALPHA[0], PA.ALPHA[-1])
                sd = rng.uniform(.025, .045, (n_frames, 2))
                pos = simulate(K_true, alpha_true, sd[:, 0], DT, rng, n_frames=n_frames)
                t = track_table(1, np.arange(n_frames), pos, sd)
                s = PA.track_alpha_posterior(t, Acquisition(DT), K_prior, level=.9)
                hits += s["lo"] <= alpha_true <= s["hi"]
            self.assertLess(abs(hits/n_tracks - .9), .08)

    def test_short_track_posterior_is_wide(self):
        """A 5-frame track's alpha posterior stays wide (honest), not falsely confident."""
        rng = np.random.default_rng(4)
        n = 5
        sd = rng.uniform(.03, .045, (n, 2))
        pos = simulate(.05, 1.0, sd[:, 0], DT, rng, n_frames=n)
        t = track_table(1, np.arange(n), pos, sd)
        s = PA.track_alpha_posterior(t, Acquisition(DT), PA.flat_K(PA.U))
        self.assertGreater(s["hi"] - s["lo"], 0.6 * (PA.ALPHA[-1] - PA.ALPHA[0]))

    def test_flat_alpha_prior_is_uniform_over_grid(self):
        np.testing.assert_array_equal(PA.flat_alpha(), np.zeros_like(PA.ALPHA))

    def test_exposure_blur_is_not_modeled(self):
        rng = np.random.default_rng(5)
        n = 6
        sd = rng.uniform(.02, .05, (n, 2))
        pos = simulate(.05, 1.0, sd[:, 0], DT, rng, n_frames=n)
        t = track_table(1, np.arange(n), pos, sd)
        with self.assertRaisesRegex(ValueError, "exposure"):
            PA.track_alpha_posterior(t, Acquisition(DT, exposure_s=.01))


if __name__ == "__main__":
    unittest.main()
