"""Grid posterior over D (diffusionkit.gridpost.posterior) against simulation."""
import unittest

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions
from diffusionkit.gridpost import posterior as P

DT = .033
OPTIONS = GridPostOptions()
U = OPTIONS.u_D()


def track_table(track_id, frames, positions, sd):
    return pl.DataFrame({
        "track_id": [track_id] * len(frames), "frame": frames,
        "x_um": positions[:, 0], "y_um": positions[:, 1],
        "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1],
    })


def simulate(D, sd, dt, rng, exposure=0., n_frames=None):
    n = len(sd) if n_frames is None else n_frames
    sub = 100
    h = dt/sub
    n_on = round(exposure/h)
    path = np.concatenate([np.zeros((1, 2)),
                           np.cumsum(rng.normal(0, np.sqrt(2*D*h), ((n-1)*sub+n_on, 2)), axis=0)])
    w = np.ones(n_on+1)
    w[[0, -1]] = .5
    pos = np.array([w @ path[i*sub:i*sub+n_on+1] / max(n_on, 1) if n_on else path[i*sub]
                    for i in range(n)])
    return pos + sd[:, None]*rng.standard_normal((n, 2))


class PosteriorTests(unittest.TestCase):
    def test_matches_direct_gaussian(self):
        rng = np.random.default_rng(1)
        n = 7
        sd = rng.uniform(.02, .05, (n, 2))
        pos = simulate(.05, sd[:, 0], DT, rng, n_frames=n)
        t = track_table(1, np.arange(n), pos, sd)
        ll = P.track_loglik(t, Acquisition(DT), U)
        from diffusionkit.gridpost.likelihood import brownian_log_likelihood
        direct = np.array([brownian_log_likelihood(t, Acquisition(DT), D) for D in np.exp(U)])
        np.testing.assert_allclose(ll, direct, atol=1e-8)

    def test_credible_intervals_are_calibrated(self):
        """Truth drawn from the prior, data simulated independently: 90% intervals cover 90%."""
        lo, hi = 1e-3, 1.
        prior = P.log_uniform(lo, hi, U)
        rng = np.random.default_rng(2)
        for n_frames in (5, 12):
            hits = 0
            n_tracks = 300
            for _ in range(n_tracks):
                u_true = rng.uniform(np.log(lo), np.log(hi))
                D_true = np.exp(u_true)
                sd = rng.uniform(.025, .045, (n_frames, 2))
                pos = simulate(D_true, sd[:, 0], DT, rng, n_frames=n_frames)
                t = track_table(1, np.arange(n_frames), pos, sd)
                s = P.track_posterior(t, Acquisition(DT), prior, GridPostOptions(level=.9))
                hits += s["lo"] <= D_true <= s["hi"]
            self.assertLess(abs(hits/n_tracks - .9), .06)

    def test_short_track_posterior_is_wide(self):
        """A 5-frame track's posterior stays wide (honest), not falsely confident."""
        rng = np.random.default_rng(3)
        n = 5
        sd = rng.uniform(.03, .045, (n, 2))
        pos = simulate(.05, sd[:, 0], DT, rng, n_frames=n)
        t = track_table(1, np.arange(n), pos, sd)
        s = P.track_posterior(t, Acquisition(DT), P.log_uniform(1e-3, 1., U))
        self.assertGreater(s["hi"]/s["lo"], 3.)

    def test_flat_prior_is_uniform_over_grid(self):
        np.testing.assert_array_equal(P.flat(U), np.zeros_like(U))


if __name__ == "__main__":
    unittest.main()
