"""The NUTS models' exposure blur (diffusionkit.bayes) against gridpost's numpy implementation."""
import importlib.util
import unittest

import numpy as np
import polars as pl

HAS_BAYES = all(importlib.util.find_spec(name) is not None for name in ("jax", "numpyro"))

DT = .033


@unittest.skipUnless(HAS_BAYES, "needs the [bayes] extra (jax, numpyro)")
class BayesBlurTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp

        import diffusionkit.bayes  # noqa: F401  (enables float64)
        from diffusionkit.bayes import likelihood as BL
        cls.jax, cls.jnp, cls.BL = jax, jnp, BL

    def test_covariance_matches_gridpost(self):
        """One blur model: the jax covariance is gridpost's numpy one, exposure 0 to a full frame."""
        from diffusionkit.gridpost.likelihood import fgn_motion_covariance
        for alpha in (.05, .4, 1., 1.6, 1.95):
            for exposure in (0., 1e-4, .02, DT):
                # Absolute, against the diagonal: long-lag entries are second differences, so
                # exp(p log y) vs y**p rounding shows up in them relatively large (~1e-13 absolute).
                expected = fgn_motion_covariance(10, DT, alpha, exposure)
                np.testing.assert_allclose(np.asarray(self.BL.fgn_covariance(10, 1., DT, alpha, exposure)),
                                           expected, rtol=0, atol=1e-11 * expected[0, 0])

    def test_normal_model_likelihood_matches_gridpost(self):
        """Brownian + iid noise + blur: the NUTS normal model's density is gridpost's likelihood."""
        from scipy.stats import multivariate_normal

        from diffusionkit import Acquisition
        from diffusionkit.gridpost import brownian_log_likelihood
        rng = np.random.default_rng(1)
        n, sigma, D, exposure = 9, .03, .08, .02
        pos = np.cumsum(rng.normal(size=(n, 2)) * np.sqrt(2 * D * DT), axis=0)
        track = pl.DataFrame({"track_id": [1] * n, "frame": np.arange(n), "x_um": pos[:, 0], "y_um": pos[:, 1],
                              "sigma_x_um": [sigma] * n, "sigma_y_um": [sigma] * n})
        cov = np.asarray(self.BL.displacement_covariance(n - 1, D, DT, 1.0, sigma**2, exposure))
        mvn = multivariate_normal(np.zeros(n - 1), cov)
        ours = mvn.logpdf(np.diff(pos[:, 0])) + mvn.logpdf(np.diff(pos[:, 1]))
        self.assertAlmostEqual(ours, brownian_log_likelihood(track, Acquisition(DT, exposure), D), places=8)

    def test_gradients_are_finite(self):
        """NUTS differentiates in K and alpha; 0**alpha (lag 0, and lag 1 at exposure = dt) must not NaN."""
        jnp = self.jnp
        data = jnp.linspace(-.1, .1, 8)

        def loglik(K, alpha, exposure):
            cov = self.BL.displacement_covariance(8, K, DT, alpha, .03**2, exposure)
            return self.jax.scipy.stats.multivariate_normal.logpdf(data, jnp.zeros(8), cov)

        for exposure in (0., .02, DT):
            for alpha in (.3, 1., 1.8):
                grads = self.jax.grad(loglik, argnums=(0, 1))(.1, alpha, exposure)
                self.assertTrue(all(np.isfinite(float(g)) for g in grads), (exposure, alpha, grads))

    def test_batched_covariance_is_per_track(self):
        jnp = self.jnp
        K = jnp.array([.1, .2, .3])[:, None, None]
        alpha = jnp.array([.5, 1., 1.5])[:, None, None]
        batched = np.asarray(self.BL.fgn_covariance(6, K, DT, alpha, .02))
        for i in range(3):
            np.testing.assert_allclose(
                batched[i], np.asarray(self.BL.fgn_covariance(6, float(K[i, 0, 0]), DT, float(alpha[i, 0, 0]), .02)))

    def test_fit_track_takes_the_exposure(self):
        from diffusionkit.bayes import fit_track, simulate_fbm_tracks
        track = simulate_fbm_tracks([(.1, .7)], 1, 25, DT, .02, seed=2, exposure_s=.02)
        fit = fit_track(track, DT, "anomalous", num_warmup=100, num_samples=100, num_chains=1, exposure_s=.02)
        self.assertTrue(all(np.isfinite(v) for v in fit.params.values()))
        with self.assertRaisesRegex(ValueError, "exposure_s cannot exceed dt_s"):
            fit_track(track, DT, exposure_s=2 * DT)


if __name__ == "__main__":
    unittest.main()
