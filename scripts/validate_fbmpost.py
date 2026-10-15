"""Coverage study for the fBm (alpha, D_ref) profile fit (diffusionkit.fbmpost).

Ground truth comes from `fbmpost.simulate_tracks`: fBm drawn on a fine time grid and averaged over each
exposure, so the likelihood's closed-form blur is not what generates the data. Simulation-based coverage
under the model, not a calibration claim on experimental tracks.

Cells: alpha x D_ref x track length, one acquisition (dt 44 ms, exposure 20 ms, SD 13 nm). Per cell:
coverage of the per-track likelihood-ratio intervals, the median alpha_hat, how often alpha_hat sits at a grid
end, the Brownian test's rejection rate, and the shared alpha of the cell's tracks with its interval.
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
from diffusionkit.fbmpost import FbmPostOptions, analyze_tracks, shared_alpha, simulate_tracks


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
                    r = analyze_tracks(simulate_tracks(args.tracks, n, alpha, D, sigma, acq, options.tau_ref_s, rng),
                                       acq, options, keep_profiles=True, map_fn=pool.map)
                    fits, shared = r.fits, shared_alpha(r.profiles, options.level)
                    rows.append({
                        "n_frames": n, "alpha": alpha, "D_ref": D,
                        "alpha_cover": float(((fits["alpha_lo"] <= alpha) & (alpha <= fits["alpha_hi"])).mean()),
                        "D_cover": float(((fits["D_ref_lo_um2_s"] <= D) & (D <= fits["D_ref_hi_um2_s"])).mean()),
                        "alpha_hat_median": float(fits["alpha_hat"].median() or np.nan),
                        "alpha_hat_at_grid_end": float(fits["alpha_hat"].is_in([options.alpha_min, options.alpha_max])
                                                       .fill_null(False).mean()),
                        "not_identified": float(fits["alpha_hat"].is_null().mean()),
                        "brownian_rejected": float((fits["alpha_p_brownian"] < .05).mean()),
                        "shared_alpha": shared.alpha_hat, "shared_lo": shared.alpha_lo, "shared_hi": shared.alpha_hi})
    table = pl.DataFrame(rows)
    summary = {
        "seconds": round(time.perf_counter() - t0, 1),
        "level": options.level,
        "alpha_cover_by_length": dict(table.group_by("n_frames").agg(pl.col("alpha_cover").mean()).sort("n_frames").rows()),
        "D_cover_by_length": dict(table.group_by("n_frames").agg(pl.col("D_cover").mean()).sort("n_frames").rows()),
        "shared_alpha_cover": float(((table["shared_lo"] <= table["alpha"]) & (table["alpha"] <= table["shared_hi"])).mean()),
        "cells": rows,
    }
    with pl.Config(tbl_rows=100, tbl_cols=20, float_precision=2, tbl_width_chars=200):
        print(table)
    print(f"{options.level:.0%} coverage, alpha:", summary["alpha_cover_by_length"], " D_ref:", summary["D_cover_by_length"],
          " shared alpha:", summary["shared_alpha_cover"])
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
