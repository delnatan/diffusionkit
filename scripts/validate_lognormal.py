"""Coverage study for log-normal partial pooling and complete pooling (gridpost.lognormal).

Populations of D across tracks, 1000 tracks each of 5-20 frames, dt 35 ms, localization SD
30-45 nm, on the default GridPostOptions grid (flat in ln D over 1e-5..10), simulated in position
space by the same independent simulator as validate_deconvolve.py. Four truths are log-normal
(the model holds): one D for all tracks (sigma = 0), and spreads of 0.3 and 0.8 in ln D, one of
them centred below the localization floor. Per replicate it records whether the 68%/95% intervals
of the population's median D (exp mu), its spread sigma and its mean D cover the truth (for
sigma = 0, the one-sided upper bounds), and the shared D's median against the population's median
and mean: complete pooling's answer when the tracks differ. A fifth truth is two modes, where the
log-normal is the wrong shape; there it records the W1 distance in ln D of the log-normal and of
the deconvolution to the population CDF, the cost of the misfit. This is a simulation check under
the model, not a calibration claim on experimental tracks.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import norm

from diffusionkit import Acquisition
from diffusionkit.gridpost import (GridPostOptions, analyze_tracks, deconvolve_tracks, lognormal_tracks,
                                   shared_D_tracks)

from validate_deconvolve import DT, N_TRACKS, simulate

# name: [(weight, median D, SD of ln D)]; a single component is a log-normal truth
TRUTHS = {
    "one D, 0.05": [(1., .05, 0.)],
    "lognormal 0.05, sd .3": [(1., .05, .3)],
    "lognormal 0.05, sd .8": [(1., .05, .8)],
    "lognormal 0.003, sd .8 (below the floor)": [(1., .003, .8)],
    "two modes: .4 @ 0.005 | .6 @ 0.2, sd .3": [(.4, .005, .3), (.6, .2, .3)],
}
LEVELS = (.68, .95)


def sample_D(comps, n, rng):
    which = rng.choice(len(comps), n, p=[c[0] for c in comps])
    med = np.array([c[1] for c in comps])[which]
    sd = np.array([c[2] for c in comps])[which]
    return med * np.exp(sd * rng.standard_normal(n))


def true_cdf(comps, u):
    return sum(w * (norm.cdf((u - np.log(med)) / sd) if sd > 0 else (u >= np.log(med)))
               for w, med, sd in comps)


def w1(weights, u, comps):
    """W1 distance in ln D between grid weights (cell masses on u) and the population."""
    fine = np.linspace(u[0], u[-1], 20001)
    edges = np.concatenate([[u[0]], (u[1:] + u[:-1]) / 2, [u[-1]]])
    cdf = np.interp(fine, edges, np.concatenate([[0.], np.cumsum(weights)]))
    return float(np.trapezoid(np.abs(cdf - true_cdf(comps, fine)), fine))


def interval(pop, key, level):
    return pop.summary(level)[key]


def replicate(comps, rng):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analysis = analyze_tracks(simulate(sample_D(comps, N_TRACKS, rng), rng), Acquisition(DT), keep_likelihoods=True)
        pop = lognormal_tracks(analysis, rng=rng)
        shared = shared_D_tracks(analysis).summary(.95)
    out = {"warnings": [str(w.message) for w in caught], "problem": pop.problem,
           "estimate": {k: v["median"] for k, v in pop.summary(.95).items()},
           "shared_D_median_um2_s": shared["median"],
           "sigma_upper": {f"{lv}": float(pop.summary(2 * lv - 1)["sigma_ln_D"]["hi"]) for lv in LEVELS}}
    if len(comps) == 1:
        _, med, sd = comps[0]
        truth = {"D_median_um2_s": med, "sigma_ln_D": sd, "D_mean_um2_s": med * np.exp(sd ** 2 / 2)}
        for lv in LEVELS:
            cover = {}
            for key, t in truth.items():
                s = interval(pop, key, lv)
                cover[key] = bool(s["lo"] <= t <= s["hi"])
            # sigma = 0 sits on its bound, where an equal-tailed interval cannot reach: one-sided instead
            cover["sigma_upper"] = bool(sd <= out["sigma_upper"][f"{lv}"])
            out[f"cover{int(100 * lv)}"] = cover
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dec = deconvolve_tracks(analysis, rng=rng)
        out["w1_lnD"] = {"lognormal": w1(pop.weights, pop.u, comps), "deconvolve": w1(dec.weights, dec.u, comps)}
    return out


def summarize(reps, comps):
    out = {"n_with_warnings": sum(bool(r["warnings"]) for r in reps),
           "estimate_median": {k: float(np.median([r["estimate"][k] for r in reps])) for k in reps[0]["estimate"]},
           "shared_D_median_um2_s": float(np.median([r["shared_D_median_um2_s"] for r in reps]))}
    if len(comps) == 1:
        for lv in LEVELS:
            key = f"cover{int(100 * lv)}"
            out[key] = {k: float(np.mean([r[key][k] for r in reps])) for k in reps[0][key]}
    else:
        out["w1_lnD_mean"] = {k: float(np.mean([r["w1_lnD"][k] for r in reps])) for k in ("lognormal", "deconvolve")}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    cells = []
    for name, comps in TRUTHS.items():
        reps = []
        for i in range(args.replicates):
            reps.append(replicate(comps, rng))
            print(f"{name}: {i + 1}/{args.replicates} ({time.perf_counter() - start:.0f} s)", file=sys.stderr, flush=True)
        cells.append({"truth": name, "components_w_median_sdlnD": comps, "summary": summarize(reps, comps),
                      "replicates": reps})
    report = {"seed": args.seed, "dt_s": DT, "n_tracks": N_TRACKS, "frames": [5, 20],
              "localization_sd_um": [.03, .045], "options": vars(GridPostOptions()),
              "elapsed_s": time.perf_counter() - start, "cells": cells}
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(json.dumps([{"truth": c["truth"], **c["summary"]} for c in cells], indent=2))


if __name__ == "__main__":
    main()
