"""Coverage/calibration study for the production D-posterior (gridpost.posterior).

Independent position-space Gaussian simulator; no production likelihood code
generates the ground truth. This is a simulation-based coverage check under
the model, not a calibration claim on experimental tracks (see
prototypes/README.md).
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions, analyze_tracks


def simulate(n, D, count, rng, dt=.033):
    steps = rng.normal(size=(count, n-1, 2))*np.sqrt(2*D*dt)
    true_positions = np.concatenate([np.zeros((count, 1, 2)), steps.cumsum(axis=1)], axis=1)
    sd = .025*np.column_stack((np.linspace(.4, 1.6, n), np.linspace(1.5, .5, n)))
    observed = true_positions + rng.normal(size=true_positions.shape)*sd
    return pl.DataFrame({
        "track_id": np.repeat(np.arange(count), n), "frame": np.tile(np.arange(n), count),
        "x_um": observed[:, :, 0].ravel(), "y_um": observed[:, :, 1].ravel(),
        "sigma_x_um": np.tile(sd[:, 0], count), "sigma_y_um": np.tile(sd[:, 1], count),
    })


def summarize(fits, D):
    lo, hi, median = (fits[c].drop_nulls().to_numpy() for c in
                      ("D_post_lo_um2_s", "D_post_hi_um2_s", "D_post_median_um2_s"))
    return {"n_estimated": len(median), "status_counts": dict(Counter(fits["status"])),
            "median_bias_ln": float(np.mean(np.log(median/D))) if len(median) else None,
            "median_rmse_ln": float(np.sqrt(np.mean(np.log(median/D)**2))) if len(median) else None,
            "coverage_90pct": float(np.mean((lo <= D) & (D <= hi))) if len(lo) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    rng = np.random.default_rng(args.seed)
    results = []
    start = time.perf_counter()
    for n in (5, 8, 10, 20):
        for D in (.01, .05, .2):
            fitted = analyze_tracks(simulate(n, D, args.replicates, rng), Acquisition(.033))
            post = fitted.fits.filter(pl.col("model") == "posterior_D")
            results.append({"n_frames": n, "D_um2_s": D, "n_tracks": args.replicates,
                            **summarize(post, D)})
    report = {"seed": args.seed, "dt_s": .033, "options": vars(GridPostOptions()),
              "prior": "flat in ln D over gridpost.posterior.U (1e-4 to 10 um^2/s)",
              "credible_level": .9, "elapsed_s": time.perf_counter()-start, "cells": results}
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output+"\n")
    print(output)


if __name__ == "__main__":
    main()
