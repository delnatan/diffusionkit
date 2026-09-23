"""Distribution of D across tracks (diffusionkit.gridpost.deconvolve) against simulation."""
import unittest

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.gridpost import deconvolve as D
from diffusionkit.gridpost import posterior as P
from diffusionkit.gridpost.data import GridPostOptions

DT = .033
U = GridPostOptions().u_D()


def track_table(track_id, frames, positions, sd):
    return pl.DataFrame({
        "track_id": [track_id] * len(frames), "frame": frames,
        "x_um": positions[:, 0], "y_um": positions[:, 1],
        "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1],
    })


def simulate(D_true, sd, dt, rng, exposure=0., n_frames=None):
    n = len(sd) if n_frames is None else n_frames
    sub = 100
    h = dt / sub
    n_on = round(exposure / h)
    path = np.concatenate([np.zeros((1, 2)),
                           np.cumsum(rng.normal(0, np.sqrt(2 * D_true * h), ((n - 1) * sub + n_on, 2)), axis=0)])
    w = np.ones(n_on + 1)
    w[[0, -1]] = .5
    pos = np.array([w @ path[i * sub:i * sub + n_on + 1] / max(n_on, 1) if n_on else path[i * sub]
                    for i in range(n)])
    return pos + sd[:, None] * rng.standard_normal((n, 2))


def simulated_table(D_values, rng, frames=(5, 13), track_id0=0):
    """One track per entry of D_values, each with a distinct track_id."""
    tables = []
    for i, d in enumerate(D_values):
        n = rng.integers(*frames)
        sd = rng.uniform(.03, .045, (n, 2))
        pos = simulate(d, sd[:, 0], DT, rng, n_frames=n)
        tables.append(track_table(track_id0 + i, np.arange(n), pos, sd))
    return pl.concat(tables)


class DeconvolveTests(unittest.TestCase):
    def test_short_and_invalid_tracks_are_excluded_not_fatal(self):
        rng = np.random.default_rng(1)
        good = simulated_table([.05, .05, .05], rng, frames=(8, 15), track_id0=0)
        n = 5
        sd = rng.uniform(.03, .045, (n, 2))
        short = track_table(100, np.arange(2), simulate(.05, sd[:2, 0], DT, rng, n_frames=2), sd[:2])
        zero_sd = track_table(101, np.arange(n), simulate(.05, sd[:, 0], DT, rng, n_frames=n),
                              np.zeros_like(sd))
        table = pl.concat([good, short, zero_sd])
        result = D.deconvolve_tracks(table, Acquisition(DT))
        self.assertEqual(result.n_tracks, 3)
        self.assertEqual(result.n_excluded, 2)
        self.assertAlmostEqual(result.weights.sum(), 1., places=8)
        self.assertEqual(len(result.weights), len(result.u))

    def test_empty_after_exclusion_raises(self):
        rng = np.random.default_rng(2)
        n = 2
        sd = rng.uniform(.03, .045, (n, 2))
        short = track_table(0, np.arange(n), simulate(.05, sd[:, 0], DT, rng, n_frames=n), sd)
        with self.assertRaises(ValueError):
            D.deconvolve_tracks(short, Acquisition(DT))

    def test_min_frames_option_is_respected(self):
        rng = np.random.default_rng(3)
        table = simulated_table([.05] * 3, rng, frames=(5, 6), track_id0=0)  # all 5-frame tracks
        with self.assertRaises(ValueError):  # all excluded by the raised min_frames
            D.deconvolve_tracks(table, Acquisition(DT), options=GridPostOptions(min_frames=6))
        table2 = pl.concat([table, simulated_table([.05], rng, frames=(9, 10), track_id0=3)])
        result = D.deconvolve_tracks(table2, Acquisition(DT), options=GridPostOptions(min_frames=6))
        self.assertEqual(result.n_tracks, 1)
        self.assertEqual(result.n_excluded, 3)

    def test_default_log_prior_is_flat(self):
        rng = np.random.default_rng(4)
        table = simulated_table([.05] * 5, rng, track_id0=0)
        result = D.deconvolve_tracks(table, Acquisition(DT))
        np.testing.assert_array_equal(result.u, U)

    def test_options_grid_is_the_one_used(self):
        rng = np.random.default_rng(7)
        table = simulated_table([.05] * 5, rng, track_id0=0)
        options = GridPostOptions(D_min_um2_s=1e-3, D_max_um2_s=1., n_D=101)
        result = D.deconvolve_tracks(table, Acquisition(DT), options=options)
        np.testing.assert_array_equal(result.u, options.u_D())
        self.assertEqual(len(result.weights), 101)

    def test_one_unsmoothed_em_step_is_the_mean_of_posteriors(self):
        rng = np.random.default_rng(5)
        D_values = np.exp(rng.uniform(np.log(.01), np.log(.5), 20))
        table = simulated_table(D_values, rng)
        prior = P.log_uniform(1e-3, 1., U)
        lls = np.array([
            P.track_loglik(track, Acquisition(DT), U)
            for track in table.sort("track_id", "frame").partition_by("track_id", maintain_order=True)
        ])
        one_step = D.deconvolve(lls, prior, iters=1, smooth=0)
        by_hand = np.mean([P.posterior(ll, prior) for ll in lls], axis=0)
        np.testing.assert_allclose(one_step, by_hand)

    def test_recovers_mass_of_each_mode(self):
        rng = np.random.default_rng(6)
        D_values = np.exp(np.log(rng.choice([.02, .2], 300)) + .15 * rng.standard_normal(300))
        table = simulated_table(D_values, rng, frames=(8, 21))
        result = D.deconvolve_tracks(table, Acquisition(DT), log_prior=P.log_uniform(1e-3, 1., U))
        mid = np.log(np.sqrt(.02 * .2))
        recovered = result.weights[result.u > mid].sum()
        truth = np.mean(D_values > np.sqrt(.02 * .2))
        self.assertLess(abs(recovered - truth), .06)

    def test_import_does_not_load_bayes_or_plotting(self):
        import subprocess
        import sys
        code = ("import diffusionkit.gridpost, sys; "
                "assert not any(x in sys.modules for x in ('jax','numpyro','matplotlib'))")
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
