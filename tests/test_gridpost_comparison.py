"""Profile likelihood ratio checked against an independent dense covariance fit."""
import unittest

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions, analyze_track, analyze_tracks, brownian_motion_lrt
from diffusionkit.gridpost.comparison import _motion_lrt


def track(seed=31, n=12):
    rng = np.random.default_rng(seed)
    sd = rng.uniform(.01, .025, (n, 2))
    positions = np.cumsum(rng.normal(size=(n, 2)) * .04, axis=0) + rng.normal(size=(n, 2)) * sd
    return pl.DataFrame(dict(track_id=[1]*n, frame=np.arange(n), x_um=positions[:, 0],
                             y_um=positions[:, 1], sigma_x_um=sd[:, 0], sigma_y_um=sd[:, 1]))


class ComparisonTests(unittest.TestCase):
    def test_matches_dense_two_parameter_fit(self):
        for exposure in (0., .02):
            t = track()
            delta = np.diff(t.select("x_um", "y_um").to_numpy(), axis=0)
            sd = t.select("sigma_x_um", "sigma_y_um").to_numpy()
            m, dt = len(delta), .03
            A = np.diag(np.full(m, 2*dt - 2*exposure/3))
            A += np.diag(np.full(m-1, exposure/3), 1) + np.diag(np.full(m-1, exposure/3), -1)
            Bs = []
            for a in range(2):
                v = sd[:, a]**2
                Bs.append(np.diag(v[:-1]+v[1:]) - np.diag(v[1:-1], 1) - np.diag(v[1:-1], -1))

            def nll(D, c):
                return -sum(multivariate_normal.logpdf(delta[:, a], cov=D*A+c*c*Bs[a]) for a in range(2))

            null = minimize(lambda z: nll(0., np.exp(z[0])), [0.], method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-10})
            candidates = [null.fun]
            for start in ((.01, 1.), (.1, .1), (.001, 3.)):
                fit = minimize(lambda z: nll(*np.exp(z)), np.log(start), method="Nelder-Mead",
                               options={"maxiter": 3000, "xatol": 1e-9, "fatol": 1e-9})
                candidates.append(fit.fun)
            expected = 2*(null.fun-min(candidates))
            self.assertAlmostEqual(brownian_motion_lrt(t, Acquisition(dt, exposure)), expected, places=6)

    def test_scale_invariance_and_grid_independence(self):
        t, acq = track(), Acquisition(.03)
        expected = brownian_motion_lrt(t, acq)
        for cols, factor in ((["x_um", "y_um"], 100.), (["sigma_x_um", "sigma_y_um"], .03)):
            scaled = t.with_columns([pl.col(col)*factor for col in cols])
            self.assertAlmostEqual(brownian_motion_lrt(scaled, acq), expected, places=8)
        options = GridPostOptions(compute_alpha=False, D_min_um2_s=.1, D_max_um2_s=.2)
        self.assertEqual(analyze_track(t, acq, options).posterior_D.parameters["D_motion_lrt"], expected)

    def test_noise_and_motion_shapes_and_zero_displacements(self):
        lam = np.geomspace(.01, 100., 20).reshape(2, 10)
        self.assertAlmostEqual(_motion_lrt(lam, np.ones_like(lam)), 0., places=10)
        self.assertGreater(_motion_lrt(lam, np.sqrt(lam)), 20.)
        self.assertIsNone(_motion_lrt(lam, np.zeros_like(lam)))
        t = track().with_columns(pl.lit(0.).alias("x_um"), pl.lit(0.).alias("y_um"))
        out = analyze_track(t, Acquisition(.03), GridPostOptions(compute_alpha=False))
        self.assertEqual(out.posterior_D.status, "ok")
        self.assertIsNone(out.posterior_D.parameters["D_motion_lrt"])
        self.assertIn("zero displacements", out.posterior_D.message)

    def test_known_interior_optimum(self):
        # Each squared mode equals its fitted variance, attaining the maximum
        # even if every mode were allowed a separate variance. Both covariance
        # components are strictly positive at this known global optimum.
        lam = np.geomspace(.001, 1000., 30).reshape(2, 15)
        variance = 2. + .7*lam
        expected = variance.size*np.log(variance.mean()) - np.log(variance).sum()
        self.assertAlmostEqual(_motion_lrt(lam, np.sqrt(variance)), expected, places=9)

    def test_table_summary_and_exclusions(self):
        t, acq = track(), Acquisition(.03)
        short = t.head(2).with_columns(pl.lit(2).alias("track_id"))
        invalid = t.with_columns(pl.lit(3).alias("track_id"), pl.lit(0.).alias("sigma_x_um"))
        out = analyze_tracks(pl.concat([t, short, invalid], how="vertical_relaxed"), acq,
                             GridPostOptions(compute_alpha=False)).fits
        scores = out.filter(pl.col("model") == "posterior_D")["D_motion_lrt"].to_list()
        self.assertEqual(scores, [brownian_motion_lrt(t, acq), None, None])
        self.assertEqual(out.filter(pl.col("model") == "posterior_alpha")["D_motion_lrt"].null_count(), 3)


if __name__ == "__main__":
    unittest.main()
