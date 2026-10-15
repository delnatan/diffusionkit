"""Wall-clock time of the per-track and population steps on simulated tracks.

Brownian tracks with per-frame localization SDs and a log-normal spread of D. Times are for one
process with the caller's BLAS threads; pass a pool's `map` to `analyze_tracks` for more.

    python scripts/benchmark.py            # the default sizes
    python scripts/benchmark.py --quick    # small sizes, a smoke test
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit import fbmpost, gridpost


def simulate(n_tracks, lengths, rng, D=.1, dt=.02, sd=.03):
    rows = []
    for i in range(n_tracks):
        n = int(rng.choice(lengths))
        Di = D * np.exp(rng.normal(0, 1.5))
        s = sd * np.exp(rng.normal(0, .3, (n, 2)))
        xy = np.cumsum(rng.normal(0, np.sqrt(2 * Di * dt), (n, 2)), axis=0) + rng.normal(size=(n, 2)) * s
        rows.append(pl.DataFrame({"track_id": [i] * n, "frame": np.arange(n), "x_um": xy[:, 0], "y_um": xy[:, 1],
                                  "sigma_x_um": s[:, 0], "sigma_y_um": s[:, 1]}))
    return pl.concat(rows)


def timed(label, f):
    t0 = time.perf_counter()
    out = f()
    print(f"{label:<55s} {time.perf_counter() - t0:8.2f} s")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    acq = Acquisition(dt_s=.02, exposure_s=.01)
    k = 10 if args.quick else 1
    short = simulate(2000 // k, np.arange(5, 31), rng)
    long = simulate(200 // k, [500], rng)
    fbm_short = simulate(200 // k, np.arange(5, 31), rng)
    fbm_long = simulate(max(20 // k, 2), [200], rng)

    r = timed(f"gridpost.analyze_tracks, {2000 // k} tracks of 5-30 frames",
              lambda: gridpost.analyze_tracks(short, acq, keep_likelihoods=True))
    timed(f"gridpost.analyze_tracks, {200 // k} tracks of 500 frames", lambda: gridpost.analyze_tracks(long, acq))
    timed(f"gridpost.deconvolve_tracks, {2000 // k} tracks", lambda: gridpost.deconvolve_tracks(r))
    f = timed(f"fbmpost.analyze_tracks, {200 // k} tracks of 5-30 frames",
              lambda: fbmpost.analyze_tracks(fbm_short, acq, keep_likelihoods=True))
    timed(f"fbmpost.analyze_tracks, {max(20 // k, 2)} tracks of 200 frames",
          lambda: fbmpost.analyze_tracks(fbm_long, acq))
    timed(f"fbmpost.deconvolve_tracks, {200 // k} tracks", lambda: fbmpost.deconvolve_tracks(f.likelihoods))


if __name__ == "__main__":
    main()
