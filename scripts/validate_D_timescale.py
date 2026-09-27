"""D at two timescales (gridpost.timescale): calibration and what it detects.

Tracks are simulated in position space at a fine time step and averaged over
the exposure -- no production covariance generates them -- with per-frame
localization SDs that vary (as a CRLB does), then analyzed with
`GridPostOptions(D_long_stride=k)`:

  - Brownian: D(tau) = D exactly, so the D_long interval should cover D and
    the ratio interval 1 in >= 90% of tracks (both are built to err wide),
    and P_D_decrease > 0.95 is a false positive;
  - confined (a particle in a harmonic trap, per-axis rms excursion L,
    relaxation time L^2 / D), and fBm with alpha < 1: the ratio should fall
    below 1, and P_D_decrease > 0.95 is a detection. The reference ratio is
    the noise- and blur-free one, MSD(k dt) / (k MSD(dt)).

It also compares the phase-averaged D_long with one thinned phase alone
(what the averaging buys). A check under the models, not a calibration
claim on experimental tracks.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions, analyze_tracks
from diffusionkit.gridpost import posterior as P
from diffusionkit.gridpost.likelihood import _loglik, _whiten

DT, EXPOSURE, SUB = .033, .02, 20  # fine steps per frame
D = .3
LENGTHS = (20, 50, 100)
SD_RANGE = (.015, .05)
SCENARIOS = {
    "brownian": {},
    "confined L=0.3um": {"L": .3},
    "confined L=0.15um": {"L": .15},
    "fbm alpha=0.6": {"alpha": .6},
}


def fine_paths(scenario, n, count, rng):
    """(count, n_fine, 2) true positions at step h = DT / SUB."""
    h, n_fine = DT / SUB, (n - 1) * SUB + round(EXPOSURE / (DT / SUB)) + 1
    if "L" in scenario:  # exact OU update, started in its stationary distribution
        tau = scenario["L"] ** 2 / D
        a = np.exp(-h / tau)
        x = np.empty((count, n_fine, 2))
        x[:, 0] = rng.normal(0, scenario["L"], (count, 2))
        noise = rng.normal(0, scenario["L"] * np.sqrt(1 - a * a), (count, n_fine - 1, 2))
        for i in range(1, n_fine):
            x[:, i] = a * x[:, i - 1] + noise[:, i - 1]
        return x
    if "alpha" in scenario:  # exact fBm on the fine grid, MSD_1D(t) = 2 K t^alpha with K = D
        alpha = scenario["alpha"]
        t = np.arange(1, n_fine) * h
        cov = D * (t[:, None] ** alpha + t[None, :] ** alpha - np.abs(t[:, None] - t[None, :]) ** alpha)
        L = np.linalg.cholesky(cov)
        path = np.einsum("ij,cja->cia", L, rng.standard_normal((count, n_fine - 1, 2)))
        return np.concatenate([np.zeros((count, 1, 2)), path], axis=1)
    steps = rng.normal(0, np.sqrt(2 * D * h), (count, n_fine - 1, 2))
    return np.concatenate([np.zeros((count, 1, 2)), steps.cumsum(axis=1)], axis=1)


def simulate(scenario, n, count, rng):
    x = fine_paths(scenario, n, count, rng)
    n_on = round(EXPOSURE / (DT / SUB))
    w = np.ones(n_on + 1)
    w[[0, -1]] = .5
    pos = np.stack([x[:, i * SUB:i * SUB + n_on + 1] .transpose(0, 2, 1) @ w / n_on for i in range(n)], axis=1)
    sd = rng.uniform(*SD_RANGE, (count, n, 2))
    obs = pos + sd * rng.standard_normal(pos.shape)
    return pl.DataFrame({
        "track_id": np.repeat(np.arange(count), n), "frame": np.tile(np.arange(n), count),
        "x_um": obs[:, :, 0].ravel(), "y_um": obs[:, :, 1].ravel(),
        "sigma_x_um": sd[:, :, 0].ravel(), "sigma_y_um": sd[:, :, 1].ravel()})


def reference_ratio(scenario, k):
    msd = lambda t: (2 * scenario["L"] ** 2 * (1 - np.exp(-t * D / scenario["L"] ** 2)) if "L" in scenario  # noqa: E731
                     else 2 * D * t ** scenario.get("alpha", 1.))
    return float(msd(k * DT) / (k * msd(DT)))


def single_phase(tracks, k, options):
    """D_long from phase 0 alone: (median, lo, hi) per track."""
    u, out = options.u_D(), {}
    acquisition = Acquisition(k * DT, EXPOSURE)
    for (track_id,), g in tracks.group_by("track_id"):
        w = _whiten(g.sort("frame")[::k], acquisition)
        s = P.summary(P.posterior(_loglik(np.exp(u), w["lam"], w["y"][None], w["const"]), P.flat(u)), u)
        out[track_id] = (s["median"], s["lo"], s["hi"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    k = args.stride
    options = GridPostOptions(compute_alpha=False, D_long_stride=k)
    rng = np.random.default_rng(args.seed)
    cells = []
    print(f"stride {k}: tau {DT * 1e3:.0f} ms -> {k * DT * 1e3:.0f} ms, exposure {EXPOSURE * 1e3:.0f} ms, D {D}")
    for name, scenario in SCENARIOS.items():
        ref = reference_ratio(scenario, k)
        for n in LENGTHS:
            tracks = simulate(scenario, n, args.replicates, rng)
            fits = analyze_tracks(tracks, Acquisition(DT, EXPOSURE), options).fits
            long = fits.filter((pl.col("model") == "posterior_D_timescale") & (pl.col("status") == "ok"))
            ratio_med = long["D_ratio_post_median"].to_numpy()
            cell = {"scenario": name, "n_frames": n, "stride": k, "reference_ratio": ref, "n_ok": long.height,
                    "ratio_median_of_medians": float(np.median(ratio_med)),
                    "ratio_interval_covers_reference": float(np.mean(
                        (long["D_ratio_post_lo"].to_numpy() <= ref) & (ref <= long["D_ratio_post_hi"].to_numpy()))),
                    "frac_P_decrease_gt_0.95": float(np.mean(long["P_D_decrease"].to_numpy() > .95)),
                    "frac_P_decrease_lt_0.05": float(np.mean(long["P_D_decrease"].to_numpy() < .05))}
            if name == "brownian":
                one = single_phase(tracks.filter(pl.col("track_id").is_in(long["track_id"].implode())), k, options)
                one = np.array([one[t] for t in long["track_id"]])
                avg = long.select("D_long_post_median_um2_s", "D_long_post_lo_um2_s", "D_long_post_hi_um2_s").to_numpy()
                cell.update({
                    "D_long_coverage_averaged": float(np.mean((avg[:, 1] <= D) & (D <= avg[:, 2]))),
                    "D_long_coverage_one_phase": float(np.mean((one[:, 1] <= D) & (D <= one[:, 2]))),
                    "D_long_rmse_ln_averaged": float(np.sqrt(np.mean(np.log(avg[:, 0] / D) ** 2))),
                    "D_long_rmse_ln_one_phase": float(np.sqrt(np.mean(np.log(one[:, 0] / D) ** 2))),
                    "D_long_width_ln_averaged": float(np.median(np.log(avg[:, 2] / avg[:, 1]))),
                    "D_long_width_ln_one_phase": float(np.median(np.log(one[:, 2] / one[:, 1]))),
                })
            cells.append(cell)
            line = (f"{name:18s} n={n:3d}: ratio median {cell['ratio_median_of_medians']:.2f} (reference {ref:.2f}), "
                    f"interval covers reference {cell['ratio_interval_covers_reference']:.2f}, "
                    f"P(decrease)>0.95 in {cell['frac_P_decrease_gt_0.95']:.2f}")
            if name == "brownian":
                line += (f"\n{'':25s}D_long coverage {cell['D_long_coverage_averaged']:.2f} (one phase "
                         f"{cell['D_long_coverage_one_phase']:.2f}), RMSE ln D {cell['D_long_rmse_ln_averaged']:.3f} "
                         f"(one phase {cell['D_long_rmse_ln_one_phase']:.3f}), 90% width ln D "
                         f"{cell['D_long_width_ln_averaged']:.2f} (one phase {cell['D_long_width_ln_one_phase']:.2f})")
            print(line, flush=True)
    if args.output:
        args.output.write_text(json.dumps({"dt_s": DT, "exposure_s": EXPOSURE, "D_um2_s": D, "sd_range_um": SD_RANGE,
                                           "replicates": args.replicates, "seed": args.seed, "cells": cells},
                                          indent=2) + "\n")


if __name__ == "__main__":
    main()
