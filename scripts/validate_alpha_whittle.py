"""Debiased Whittle vs exact alpha posterior: where can the fast likelihood stand in?

For each (track length, alpha, exposure, localization noise) cell, simulated
tracks get both alpha posteriors on the same grids and flat priors. Reported
per cell: each method's 90% coverage of the true alpha, the RMSE of its
posterior median, the median ratio of interval widths, the Hellinger distance
between the two posteriors, and the time per track.

The two posteriors do not converge to each other with length: Whittle keeps
only the noise's spectrum, not which frames were noisy, so with per-frame
SDs that vary (spotsolve's CRLB does) it discards information the exact
likelihood uses, and stays calibrated by being wider. That efficiency loss
(a few to ~15% in RMSE) does not shrink with length, so where to switch is
a cost decision, not a convergence point. The recommended
`GridPostOptions.alpha_whittle_min_frames` is the shortest length from which,
at every length at or above it, Whittle is at least `MIN_SPEEDUP` times
faster (median over cells), no cell's Whittle coverage falls below
`MIN_COVERAGE`, and the median RMSE ratio over cells is at most
`MAX_MEDIAN_RMSE_RATIO`. Judging each length by its median rather than its
worst cell: an RMSE ratio from 100 tracks is itself uncertain by several
percent.

The simulator draws from `likelihood.fgn_motion_covariance`, which
tests/test_gridpost_likelihood.py checks against box-averaged, exactly
simulated fine-step fBm. This compares two likelihoods under the model; it is
not a calibration claim on experimental tracks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions
from diffusionkit.gridpost import posterior as PD
from diffusionkit.gridpost import posterior_alpha as PA
from diffusionkit.gridpost.likelihood import _prepared, fgn_motion_covariance

DT = .033
K = .05
LENGTHS = (10, 15, 20, 30, 40, 50, 70, 100, 150, 200)
ALPHAS = (.3, .7, 1., 1.4, 1.8)
EXPOSURES = (0., .02)
# Localization SDs per frame and axis: constant, or varying as a CRLB does
# with each frame's photon count.
NOISE = {"constant": (.03, .03), "varying": (.015, .05)}
# See the module docstring. MIN_COVERAGE is ~2.7 binomial SEs under 0.9 at
# the default 100 replicates.
MIN_COVERAGE = .82
MAX_MEDIAN_RMSE_RATIO = 1.15
MIN_SPEEDUP = 2.


def simulate(n, alpha, exposure, noise, count, rng):
    L = np.linalg.cholesky(K * fgn_motion_covariance(n - 1, DT, alpha, exposure))
    disp = np.einsum("ij,cja->cia", L, rng.standard_normal((count, n - 1, 2)))
    true = np.concatenate([np.zeros((count, 1, 2)), disp.cumsum(axis=1)], axis=1)
    sd = rng.uniform(*NOISE[noise], (count, n, 2))
    observed = true + sd * rng.standard_normal(true.shape)
    return [pl.DataFrame({"track_id": [i] * n, "frame": np.arange(n),
                          "x_um": observed[i, :, 0], "y_um": observed[i, :, 1],
                          "sigma_x_um": sd[i, :, 0], "sigma_y_um": sd[i, :, 1]}) for i in range(count)]


def compare(track, acquisition, options):
    alphas, u, prior = options.alphas(), options.u_D(), PD.flat(options.u_D())
    track = _prepared(track, acquisition)
    out = {}
    for name, loglik in (("exact", PA._joint_loglik), ("whittle", PA._whittle_joint_loglik)):
        t0 = time.perf_counter()
        p = np.exp(PA.log_alpha_posterior(loglik(track, acquisition, alphas, u), prior))
        out[name] = (p, PA.summary(p, alphas, options.level), time.perf_counter() - t0)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    options = GridPostOptions()
    rng = np.random.default_rng(args.seed)
    cells = []
    with ThreadPoolExecutor(args.threads) as pool:
        for noise in NOISE:
            for exposure in EXPOSURES:
                acquisition = Acquisition(DT, exposure)
                for n in LENGTHS:
                    for alpha in ALPHAS:
                        tracks = simulate(n, alpha, exposure, noise, args.replicates, rng)
                        results = list(pool.map(lambda t: compare(t, acquisition, options), tracks))
                        cells.append(summarize(results, n, alpha, exposure, noise))
                        c = cells[-1]
                        print(f"{noise:8s} te={exposure:.3f} n={n:3d} alpha={alpha:.1f}: coverage exact "
                              f"{c['coverage_90pct_exact']:.2f} whittle {c['coverage_90pct_whittle']:.2f}  "
                              f"RMSE W/E {c['rmse_ratio']:.2f}  width W/E {c['width_ratio']:.2f}  "
                              f"H p90 {c['hellinger_p90']:.3f}  speedup {c['speedup']:.1f}x", flush=True)

    by_length = {}
    for n in LENGTHS:
        at_n = [c for c in cells if c["n_frames"] == n]
        by_length[n] = {
            "median_speedup": float(np.median([c["speedup"] for c in at_n])),
            "min_coverage_whittle": min(c["coverage_90pct_whittle"] for c in at_n),
            **{f"median_rmse_ratio_{noise}": float(np.median([c["rmse_ratio"] for c in at_n if c["noise"] == noise]))
               for noise in NOISE},
        }

    def passes(n):
        s = by_length[n]
        return (s["median_speedup"] >= MIN_SPEEDUP and s["min_coverage_whittle"] >= MIN_COVERAGE
                and all(s[f"median_rmse_ratio_{noise}"] <= MAX_MEDIAN_RMSE_RATIO for noise in NOISE))
    recommended = next((n for n in LENGTHS if all(passes(m) for m in LENGTHS if m >= n)), None)
    print("  n   speedup  min coverage W  " + "  ".join(f"RMSE W/E {noise}" for noise in NOISE))
    for n, s in by_length.items():
        print(f"{n:3d}   {s['median_speedup']:5.1f}x      {s['min_coverage_whittle']:.2f}        "
              + "         ".join(f"{s[f'median_rmse_ratio_{noise}']:.2f}" for noise in NOISE)
              + ("" if passes(n) else "   x"))
    report = {"dt_s": DT, "K": K, "noise_sd_um": NOISE, "replicates": args.replicates, "seed": args.seed,
              "criteria": {"min_coverage": MIN_COVERAGE, "max_median_rmse_ratio": MAX_MEDIAN_RMSE_RATIO,
                           "min_speedup": MIN_SPEEDUP},
              "recommended_alpha_whittle_min_frames": recommended,
              "by_length": {str(n): v for n, v in by_length.items()}, "cells": cells}
    print(f"recommended alpha_whittle_min_frames: {recommended}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def summarize(results, n, alpha, exposure, noise):
    stats = {name: {"cover": [], "err": [], "width": [], "t": []} for name in ("exact", "whittle")}
    hellinger = []
    for r in results:
        hellinger.append(float(np.sqrt(max(0., 1 - np.sum(np.sqrt(r["exact"][0] * r["whittle"][0]))))))
        for name, (_, s, t) in r.items():
            stats[name]["cover"].append(s["lo"] <= alpha <= s["hi"])
            stats[name]["err"].append(s["median"] - alpha)
            stats[name]["width"].append(s["hi"] - s["lo"])
            stats[name]["t"].append(t)
    rmse = {name: float(np.sqrt(np.mean(np.square(v["err"])))) for name, v in stats.items()}
    ms = {name: 1e3 * float(np.mean(v["t"])) for name, v in stats.items()}
    return {
        "n_frames": n, "alpha": alpha, "exposure_s": exposure, "noise": noise,
        "coverage_90pct_exact": float(np.mean(stats["exact"]["cover"])),
        "coverage_90pct_whittle": float(np.mean(stats["whittle"]["cover"])),
        "rmse_exact": rmse["exact"], "rmse_whittle": rmse["whittle"],
        "rmse_ratio": rmse["whittle"] / rmse["exact"],
        "width_ratio": float(np.median(np.array(stats["whittle"]["width"]) / np.array(stats["exact"]["width"]))),
        "hellinger_median": float(np.median(hellinger)), "hellinger_p90": float(np.percentile(hellinger, 90)),
        "ms_per_track_exact": ms["exact"], "ms_per_track_whittle": ms["whittle"],
        "speedup": ms["exact"] / ms["whittle"],
    }

if __name__ == "__main__":
    main()
