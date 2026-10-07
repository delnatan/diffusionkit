"""Coverage study for the production distribution of D across tracks (gridpost.deconvolve).

Five populations (log-normal mixtures in D, truncated to [1e-3, 1] um^2/s), 1000
tracks each of 5-20 frames, dt 35 ms, localization SD 30-45 nm, on the default
GridPostOptions grid (flat in ln D over 1e-5..10). Independent position-space
simulator; no production likelihood code generates the ground truth. Per replicate
it records the evidence's lam and whether its maximum was interior, the
Wasserstein-1 distance in ln D to the population CDF, and whether the 68%/95% bands
from `samples` cover the population CDF at fixed D, the mass above a cut, and the
(zero) mass above 1 um^2/s. This is a simulation check under the model, not a
calibration claim on experimental tracks.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from scipy.stats import norm

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions, analyze_tracks, deconvolve_tracks

DT, N_TRACKS, LO, HI = .035, 1000, np.log(1e-3), np.log(1.)
CHECK_D = (.003, .01, .03, .1, .3)

# name: (components [(weight, median D, SD of ln D)], cut D for "mass above")
TRUTHS = {
    "spike 0.05": ([(1., .05, 0.)], .1),
    "0.02 | 0.2, sd .15": ([(.5, .02, .15), (.5, .2, .15)], np.sqrt(.02 * .2)),
    "lognormal 0.05, sd .8": ([(1., .05, .8)], .05),
    ".3 @ 0.01 | .7 @ 0.1, sd .4": ([(.3, .01, .4), (.7, .1, .4)], np.sqrt(.01 * .1)),
    ".4 @ 0.002 (sd .3) | .6 @ 0.1 (sd .5)": ([(.4, .002, .3), (.6, .1, .5)], np.sqrt(.002 * .1)),
}


def true_cdf(comps, u):
    out = 0.
    for w, med, sd in comps:
        if sd == 0:
            out = out + w * (u >= np.log(med))
            continue
        a, b = norm.cdf([(LO - np.log(med)) / sd, (HI - np.log(med)) / sd])
        out = out + w * np.clip((norm.cdf((u - np.log(med)) / sd) - a) / (b - a), 0, 1)
    return out


def sample_D(comps, n, rng):
    out = np.empty(n)
    which = rng.choice(len(comps), n, p=[c[0] for c in comps])
    for k, (_, med, sd) in enumerate(comps):
        idx = np.flatnonzero(which == k)
        x = np.full(len(idx), np.log(med))
        bad = np.full(len(idx), sd > 0)
        while bad.any():
            x[bad] = np.log(med) + sd * rng.standard_normal(bad.sum())
            bad = (x < LO) | (x > HI)
        out[idx] = np.exp(x)
    return out


def simulate(D_values, rng):
    tables = []
    for i, D in enumerate(D_values):
        n = rng.integers(5, 21)
        sd = rng.uniform(.03, .045, (n, 2))
        steps = rng.normal(size=(n - 1, 2)) * np.sqrt(2 * D * DT)
        pos = np.vstack([np.zeros((1, 2)), steps.cumsum(axis=0)]) + sd * rng.normal(size=(n, 2))
        tables.append(pl.DataFrame({"track_id": i, "frame": np.arange(n), "x_um": pos[:, 0], "y_um": pos[:, 1],
                                    "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1]}))
    return pl.concat(tables)


def cdf_at(G, u, u0):
    """CDF of grid weights G (..., K) on cell centers u at points u0, linear within cells."""
    edges = np.concatenate([[u[0] - (u[1] - u[0]) / 2], u + (u[1] - u[0]) / 2])
    C = np.concatenate([np.zeros(G.shape[:-1] + (1,)), np.cumsum(G, axis=-1)], axis=-1)
    return np.apply_along_axis(lambda c: np.interp(u0, edges, c), -1, C)


def covered(draws, truth, level):
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return ((lo <= truth) & (truth <= hi)).tolist()


def replicate(comps, cut, rng):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracks = simulate(sample_D(comps, N_TRACKS, rng), rng)
        r = deconvolve_tracks(analyze_tracks(tracks, Acquisition(DT), keep_likelihoods=True), rng=rng)
    u0 = np.log(CHECK_D)
    truth = true_cdf(comps, u0)
    fine = np.linspace(LO - 1, HI + 1, 2001)
    above = r.samples[:, r.u > np.log(cut)].sum(axis=1)
    beyond = r.samples[:, r.u > HI].sum(axis=1)
    mass_true = 1 - true_cdf(comps, np.log(cut))
    return {"lam": r.lam, "warnings": [str(w.message) for w in caught],
            "w1_lnD": float(np.trapezoid(np.abs(cdf_at(r.weights, r.u, fine) - true_cdf(comps, fine)), fine)),
            "mass_above_cut_err": float(r.weights[r.u > np.log(cut)].sum() - mass_true),
            "mass_beyond_1_q975": float(np.quantile(beyond, .975)),
            **{f"cover{int(100 * lv)}": {"cdf": covered(cdf_at(r.samples, r.u, u0), truth, lv),
                                         "mass_above_cut": covered(above, mass_true, lv)}
               for lv in (.68, .95)}}


def summarize(reps, comps):
    truth = true_cdf(comps, np.log(CHECK_D))
    informative = (truth > 1e-3) & (truth < 1 - 1e-3)  # a CDF pinned at 0 or 1 is covered trivially
    out = {"lam_median": float(np.median([r["lam"] for r in reps])),
           "lam_range": [float(min(r["lam"] for r in reps)), float(max(r["lam"] for r in reps))],
           "n_with_warnings": sum(bool(r["warnings"]) for r in reps),
           "w1_lnD_mean": float(np.mean([r["w1_lnD"] for r in reps])),
           "mass_above_cut_rmse": float(np.sqrt(np.mean([r["mass_above_cut_err"] ** 2 for r in reps]))),
           "mass_beyond_1_q975_max": float(max(r["mass_beyond_1_q975"] for r in reps))}
    for lv in ("cover68", "cover95"):
        cdf = np.array([r[lv]["cdf"] for r in reps])[:, informative]
        out[lv] = {"cdf": float(cdf.mean()) if cdf.size else None,
                   "mass_above_cut": float(np.mean([r[lv]["mass_above_cut"] for r in reps]))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    cells = []
    for name, (comps, cut) in TRUTHS.items():
        reps = []
        for i in range(args.replicates):
            reps.append(replicate(comps, cut, rng))
            print(f"{name}: {i + 1}/{args.replicates} ({time.perf_counter() - start:.0f} s)", file=sys.stderr, flush=True)
        cells.append({"truth": name, "components_w_median_sdlnD": comps, "cut_um2_s": cut,
                      "summary": summarize(reps, comps), "replicates": reps})
    report = {"seed": args.seed, "dt_s": DT, "n_tracks": N_TRACKS, "frames": [5, 20],
              "localization_sd_um": [.03, .045], "options": vars(GridPostOptions()),
              "check_D_um2_s": CHECK_D, "elapsed_s": time.perf_counter() - start, "cells": cells}
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(json.dumps([{"truth": c["truth"], **c["summary"]} for c in cells], indent=2))


if __name__ == "__main__":
    main()
