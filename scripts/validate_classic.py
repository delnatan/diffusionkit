"""Descriptive short-track recovery study for the new classical API.

Independent position-space Gaussian simulator; no production likelihood or
legacy fitting code is used. This is not a confidence-interval calibration.
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
from diffusionkit.classic import MSDOptions, analyze_tracks


def simulate(n, K, alpha, count, rng, dt=.033):
    # Cov(X(t), X(s)) = K(t^alpha + s^alpha - |t-s|^alpha) per axis.
    times = np.arange(1, n)*dt
    covariance = K*(times[:, None]**alpha + times[None, :]**alpha
                    - np.abs(times[:, None]-times[None, :])**alpha)
    factor = np.linalg.cholesky(covariance)
    true_positions = np.zeros((count, n, 2))
    true_positions[:, 1:, :] = np.einsum("ij,bjd->bid", factor, rng.normal(size=(count, n-1, 2)))
    sd = .025*np.column_stack((np.linspace(.4, 1.6, n), np.linspace(1.5, .5, n)))
    observed = true_positions + rng.normal(size=true_positions.shape)*sd
    return pl.DataFrame({
        "track_id": np.repeat(np.arange(count), n), "frame": np.tile(np.arange(n), count),
        "x_um": observed[:, :, 0].ravel(), "y_um": observed[:, :, 1].ravel(),
        "sigma_x_um": np.tile(sd[:, 0], count), "sigma_y_um": np.tile(sd[:, 1], count),
    })


def summarize(fits, name, truth):
    values = fits[name].drop_nulls().to_numpy()
    return {"n_estimated": len(values), "status_counts": dict(Counter(fits["status"])),
            "mean": float(np.mean(values)) if len(values) else None,
            "bias": float(np.mean(values)-truth) if len(values) else None,
            "rmse": float(np.sqrt(np.mean((values-truth)**2))) if len(values) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    rng = np.random.default_rng(args.seed)
    results = []
    start = time.perf_counter()
    for n in (5, 10, 20):
        for K in (.01, .05):
            for alpha in (.5, 1., 1.5):
                fitted = analyze_tracks(simulate(n, K, alpha, args.replicates, rng), Acquisition(.033))
                row = {"n_frames": n, "K": K, "alpha": alpha, "n_tracks": args.replicates,
                       "alpha_fit": summarize(fitted.fits.filter(pl.col("model") == "power_law"), "alpha", alpha)}
                if alpha == 1:
                    row["D_fit"] = summarize(fitted.fits.filter(pl.col("model") == "brownian"), "D_um2_s", K)
                results.append(row)
    report = {"seed": args.seed, "dt_s": .033, "options": vars(MSDOptions()),
              "localization": "Known SDs vary by frame and axis; independent position noise",
              "summary_policy": "All returned non-null estimates, including boundary/failed fits; counts by status are retained",
              "intervals": "Not estimated; no coverage claim", "elapsed_s": time.perf_counter()-start,
              "cells": results}
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output+"\n")
    print(output)


if __name__ == "__main__":
    main()
