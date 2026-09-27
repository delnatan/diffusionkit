"""D at two timescales (diffusionkit.gridpost.timescale)."""
import unittest

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions, analyze_track, analyze_tracks
from diffusionkit.gridpost import posterior as P
from diffusionkit.gridpost import timescale as T
from diffusionkit.gridpost.likelihood import _prepared

DT = .033
OPTIONS = GridPostOptions(compute_alpha=False, D_long_stride=4)
U = OPTIONS.u_D()


def table(positions, sd, track_id=1):
    n = len(positions)
    return pl.DataFrame({"track_id": [track_id] * n, "frame": np.arange(n),
                         "x_um": positions[:, 0], "y_um": positions[:, 1],
                         "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1]})


def brownian(n, rng, D=.3, sd_range=(.015, .04)):
    sd = rng.uniform(*sd_range, (n, 2))
    return table(np.cumsum(rng.normal(0, np.sqrt(2 * D * DT), (n, 2)), axis=0) + sd * rng.standard_normal((n, 2)), sd)


def confined(n, rng, D=.3, L=.12, sd=.02):
    """A particle in a harmonic trap (exact AR(1) update, instantaneous positions)."""
    a = np.exp(-DT * D / L**2)
    x = np.empty((n, 2))
    x[0] = rng.normal(0, L, 2)
    for i in range(1, n):
        x[i] = a * x[i - 1] + rng.normal(0, L * np.sqrt(1 - a * a), 2)
    sds = np.full((n, 2), sd)
    return table(x + sds * rng.standard_normal((n, 2)), sds)


class TimescaleTests(unittest.TestCase):
    def test_thinned_loglik_is_the_mean_over_phases(self):
        """Each phase is the thinned track at interval stride * dt, with its frames' own SDs and the same exposure."""
        rng = np.random.default_rng(1)
        t = brownian(23, rng)
        acquisition, k = Acquisition(DT, .02), 4
        ll, n_phases = T.thinned_loglik(_prepared(t, acquisition), acquisition, U, k)
        phases = []
        for p in range(k):
            sub = t[p::k].with_columns(pl.Series("frame", np.arange(len(t[p::k]))))
            phases.append(P.track_loglik(sub, Acquisition(k * DT, .02), U))
        self.assertEqual(n_phases, k)
        np.testing.assert_allclose(ll, np.mean(phases, axis=0), rtol=1e-12)

    def test_short_phases_are_skipped(self):
        rng = np.random.default_rng(2)
        t = brownian(10, rng)  # stride 4: phases of 3, 3, 2, 2 frames
        acquisition = Acquisition(DT)
        _, n_phases = T.thinned_loglik(_prepared(t, acquisition), acquisition, U, 4)
        self.assertEqual(n_phases, 2)
        self.assertEqual(T.thinned_loglik(_prepared(t, acquisition), acquisition, U, 5)[1], 0)

    def test_ratio_of_shifted_posteriors(self):
        """Identical posteriors: ratio 1, P(decrease) 1/2; one shifted by j cells: ratio exp(j du)."""
        lp = P.log_posterior(-.5 * ((U - np.log(.3)) / .2) ** 2, P.flat(U))
        same = T.ratio_summary(*T.ratio_posterior(lp, lp, U))
        self.assertAlmostEqual(same["median"], 1., places=6)
        self.assertAlmostEqual(same["p_decrease"], .5, places=6)
        j = 30
        shifted = np.roll(lp, -j)  # mass moved j cells down: D_long smaller
        s = T.ratio_summary(*T.ratio_posterior(shifted, lp, U))
        self.assertAlmostEqual(np.log(s["median"]), -j * (U[1] - U[0]), places=6)
        self.assertGreater(s["p_decrease"], .99)

    def test_rows_and_exclusions(self):
        rng = np.random.default_rng(3)
        tracks = pl.concat([brownian(30, rng).with_columns(pl.lit(1).alias("track_id")),
                            brownian(8, rng).with_columns(pl.lit(2).alias("track_id"))])
        result = analyze_tracks(tracks, Acquisition(DT, .02), OPTIONS, keep_posteriors=True)
        rows = result.fits.filter(pl.col("model") == "posterior_D_timescale").sort("track_id")
        self.assertEqual(rows["status"].to_list(), ["ok", "excluded"])
        self.assertIn("D_long_stride=4 needs >= 9", rows["message"][1])
        self.assertAlmostEqual(rows["tau_long_s"][0], 4 * DT)
        self.assertEqual(result.posteriors.D_long_track_ids.tolist(), [1])
        self.assertEqual(result.posteriors.log_post_D_long.shape, (1, OPTIONS.n_D))
        without = analyze_tracks(tracks, Acquisition(DT), GridPostOptions(compute_alpha=False)).fits
        self.assertNotIn("posterior_D_timescale", without["model"].to_list())
        for bad in (1, 0, True):
            with self.assertRaisesRegex(ValueError, "D_long_stride"):
                GridPostOptions(D_long_stride=bad)

    def test_brownian_intervals_cover_and_rarely_flag(self):
        """Brownian: D(tau) = D, so D_long's interval covers D and the ratio's covers 1 (both err wide)."""
        rng = np.random.default_rng(4)
        cover_long = cover_ratio = flagged = 0
        n_tracks = 150
        for i in range(n_tracks):
            out = analyze_track(brownian(40, rng), Acquisition(DT, .02), OPTIONS).posterior_D_long.parameters
            cover_long += out["D_long_post_lo_um2_s"] <= .3 <= out["D_long_post_hi_um2_s"]
            cover_ratio += out["D_ratio_post_lo"] <= 1 <= out["D_ratio_post_hi"]
            flagged += out["P_D_decrease"] > .95
        self.assertGreaterEqual(cover_long / n_tracks, .88)
        self.assertGreaterEqual(cover_ratio / n_tracks, .88)
        self.assertLessEqual(flagged / n_tracks, .05)

    def test_confinement_lowers_the_ratio(self):
        """Strongly trapped (L^2/D ~ 1.5 frames): D at 4 frames is well below D at 1."""
        rng = np.random.default_rng(5)
        ratios = [analyze_track(confined(100, rng), Acquisition(DT), OPTIONS).posterior_D_long
                  .parameters["D_ratio_post_median"] for _ in range(20)]
        self.assertLess(np.median(ratios), .7)


if __name__ == "__main__":
    unittest.main()
