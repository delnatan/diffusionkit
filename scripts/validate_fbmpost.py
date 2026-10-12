"""Coverage/calibration study for the fBm (alpha, D) grid posterior (diffusionkit.fbmpost).

Ground truth comes from `fbmpost.simulate_tracks`: fBm drawn on a fine time grid and averaged over each
exposure, so the posterior's closed-form blur is not what generates the data. Simulation-based coverage
under the model, not a calibration claim on experimental tracks.

Cells: alpha x D_ref x track length, gelatin-like acquisition (dt 44 ms, exposure 20 ms, SD 13 nm), plus
immobile emitters, which should return the flat alpha prior.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.fbmpost import FbmPostOptions, analyze_tracks, simulate_tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", type=int, default=40, help="tracks per cell")
    parser.add_argument("--out", type=Path, default=None, help="write the summary as JSON here")
    args = parser.parse_args()
    acq, options, sigma = Acquisition(.044, .02), FbmPostOptions(), .013
    rng = np.random.default_rng(0)
    rows, t0 = [], time.perf_counter()
    with ThreadPoolExecutor() as pool:
        for n in (10, 20, 40):
            for alpha in (.3, .6, 1., 1.4):
                for D in (1e-3, 3e-3, 1e-2, 3e-2):
                    fits = analyze_tracks(simulate_tracks(args.tracks, n, alpha, D, sigma, acq, options.tau_ref_s, rng),
                                          acq, options, map_fn=pool.map).fits
                    rows.append({
                        "n_frames": n, "alpha": alpha, "D_ref": D,
                        "alpha_cover": float(((fits["alpha_post_lo"] <= alpha) & (alpha <= fits["alpha_post_hi"])).mean()),
                        "D_cover": float(((fits["D_ref_post_lo_um2_s"] <= D) & (D <= fits["D_ref_post_hi_um2_s"])).mean()),
                        "alpha_mean_median": float(fits["alpha_post_mean"].median()),
                        "alpha_sd_median": float(fits["alpha_post_sd"].median()),
                        "log10_D_bias_median": float((fits["log10_D_ref_post_mean"] - np.log10(D)).median()),
                        "corr_median": float(fits["alpha_log10_D_ref_corr"].median()),
                        "alpha_bits_median": float(fits["alpha_post_info_bits"].median())})
            r = analyze_tracks(simulate_tracks(4 * args.tracks, n, 1., 0., sigma, acq, options.tau_ref_s, rng), acq,
                               options, keep_posteriors=True, map_fn=pool.map)
            p = np.exp(r.posteriors.log_post_alpha).mean(axis=0)
            rows.append({"n_frames": n, "alpha": None, "D_ref": 0.,
                         "immobile_max_rel_dev_from_flat": float(np.abs(p * len(p) - 1).max()),
                         "immobile_alpha_mean": float(p @ options.alphas())})
    table = pl.DataFrame(rows)
    grid = table.filter(pl.col("D_ref") > 0)
    summary = {
        "seconds": round(time.perf_counter() - t0, 1),
        "alpha_cover_by_length": dict(grid.group_by("n_frames").agg(pl.col("alpha_cover").mean()).sort("n_frames").rows()),
        "D_cover_by_length": dict(grid.group_by("n_frames").agg(pl.col("D_cover").mean()).sort("n_frames").rows()),
        "cells": rows,
    }
    with pl.Config(tbl_rows=100, tbl_cols=20, float_precision=2, tbl_width_chars=200):
        print(grid.drop([c for c in grid.columns if c.startswith("immobile")]))
        print(table.filter(pl.col("D_ref") == 0).select("n_frames", "immobile_max_rel_dev_from_flat",
                                                        "immobile_alpha_mean"))
    print("90% coverage, alpha:", summary["alpha_cover_by_length"], " D_ref:", summary["D_cover_by_length"])
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
