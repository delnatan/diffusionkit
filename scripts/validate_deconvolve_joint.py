"""Recovery study for the joint (alpha, D) distribution across tracks (gridpost.deconvolve_joint).

Populations of caged (alpha 0.1, apparent D log-normal around 0.03 um^2/s), mobile (alpha
~ N(0.9, 0.1), apparent D log-normal around 0.3) and immobile (no motion) particles,
1500 tracks per dataset, dt = exposure = 20 ms (the box-shutter blur modelled), per-frame
localization SDs 15-30 nm, and short tracks as in GEM movies: lengths 3 + geometric, so
most tracks have 3-6 frames. In the length-correlated scenarios caged and immobile tracks
are longer than mobile ones, as particles that stay in focus are. Tracks come from the
exact blurred-fBm covariance (`likelihood.fgn_motion_covariance`, which
tests/test_gridpost_likelihood.py checks against simulated fine-step fBm) plus noise.

Per replicate it records the fraction of tracks in the low-alpha population (truth: the
realized fraction caged), the joint read's mass at alpha < 0.4 with its 90% interval from
`samples`, the same mass from the pooled 1D alpha read (`deconvolve` on the alpha
posteriors), the joint read's mass of alpha < 0.4 below the localization floor, the
evidence's lam and the time. A simulation check under the model, not a calibration claim
on experimental tracks.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.gridpost import GridPostOptions, analyze_tracks
from diffusionkit.gridpost.deconvolve import deconvolve, deconvolve_joint
from diffusionkit.gridpost.likelihood import fgn_motion_covariance

DT = EXPOSURE = .02
N_TRACKS = 1500
LOW_ALPHA = .4
# name: ({class: fraction}, {class: geometric p of track length - 3})
SCENARIOS = {
    "all mobile": ({"mobile": 1.}, {"mobile": .3}),
    "25% caged, same lengths": ({"caged": .25, "mobile": .75}, {"caged": .3, "mobile": .3}),
    "25% caged, caged tracks longer": ({"caged": .25, "mobile": .75}, {"caged": .08, "mobile": .4}),
    "30% immobile, immobile tracks longer": ({"immobile": .3, "mobile": .7}, {"immobile": .08, "mobile": .4}),
}


def simulate(fractions, length_p, rng):
    classes = rng.choice(list(fractions), N_TRACKS, p=list(fractions.values()))
    tables, chol = [], {}
    for i, c in enumerate(classes):
        n = min(3 + rng.geometric(length_p[c]) - 1, 60)
        sd = rng.uniform(.015, .03, (n, 2))
        if c == "immobile":
            pos = np.zeros((n, 2))
        else:
            alpha = .1 if c == "caged" else float(np.clip(rng.normal(.9, .1), .5, 1.4))
            D = (.03 if c == "caged" else .3) * np.exp((.4 if c == "caged" else .8) * rng.standard_normal())
            A = fgn_motion_covariance(n - 1, DT, alpha, EXPOSURE)
            K = D * 2 * (DT - EXPOSURE / 3) / A[0, 0]
            pos = np.vstack([np.zeros(2), np.cumsum(np.linalg.cholesky(K * A) @ rng.standard_normal((n - 1, 2)),
                                                    axis=0)])
        pos = pos + sd * rng.standard_normal((n, 2))
        tables.append(pl.DataFrame({"track_id": i, "frame": np.arange(n), "x_um": pos[:, 0], "y_um": pos[:, 1],
                                    "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1]}))
    return pl.concat(tables), classes


def run(name, rng, pool):
    fractions, length_p = SCENARIOS[name]
    table, classes = simulate(fractions, length_p, rng)
    options, acquisition = GridPostOptions(), Acquisition(DT, EXPOSURE)
    result = analyze_tracks(table, acquisition, options, keep_posteriors=True, map_fn=pool.map)
    post, alphas = result.posteriors, options.alphas()
    low = alphas < LOW_ALPHA
    floor = float(np.median(result.fits.filter(pl.col("model") == "posterior_D")["D_floor_um2_s"].drop_nulls()))
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joint = deconvolve_joint(post.log_post_joint, alphas, options.u_joint_D(), n_samples=1000,
                                 rng=np.random.default_rng(rng.integers(2**32)))
        seconds = time.perf_counter() - t0
        pooled = deconvolve(post.log_post_alpha, alphas, n_samples=200)
    mass = joint.samples[:, low, :].sum(axis=(1, 2))
    lo, hi = np.quantile(mass, [.05, .95])
    truth = float(np.mean(classes == "caged"))
    below = options.u_joint_D() < np.log(floor)
    return {"scenario": name, "truth_caged": truth, "frac_immobile": float(np.mean(classes == "immobile")),
            "median_track_frames": float(np.median(table.group_by("track_id").len()["len"])),
            "joint_low": float(joint.weights[low].sum()), "joint_low_lo90": float(lo), "joint_low_hi90": float(hi),
            "joint_covers_90": bool(lo <= truth <= hi),
            "joint_low_below_floor": float(joint.weights[np.ix_(low, below)].sum()),
            "pooled_1d_low": float(pooled.weights[low].sum()),
            "log10_lam": [float(x) for x in np.log10(joint.lam)], "seconds": seconds,
            "warnings": sorted({str(w.message)[:80] for w in caught})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    with ThreadPoolExecutor(8) as pool:
        for name in SCENARIOS:
            for _ in range(args.replicates):
                rows.append(run(name, rng, pool))
                r = rows[-1]
                print(f"{name:38s} truth {r['truth_caged']:.3f}  joint {r['joint_low']:.3f} "
                      f"[{r['joint_low_lo90']:.3f}, {r['joint_low_hi90']:.3f}]  below floor {r['joint_low_below_floor']:.3f}  "
                      f"pooled 1D {r['pooled_1d_low']:.3f}  {r['seconds']:.0f} s {r['warnings']}", flush=True)
    summary = {}
    for name in SCENARIOS:
        rs = [r for r in rows if r["scenario"] == name]
        summary[name] = {
            "mean_truth_caged": float(np.mean([r["truth_caged"] for r in rs])),
            "mean_joint_error": float(np.mean([r["joint_low"] - r["truth_caged"] for r in rs])),
            "mean_pooled_1d_error": float(np.mean([r["pooled_1d_low"] - r["truth_caged"] for r in rs])),
            "joint_coverage_90": float(np.mean([r["joint_covers_90"] for r in rs])),
            "mean_joint_low_below_floor": float(np.mean([r["joint_low_below_floor"] for r in rs])),
            "median_seconds": float(np.median([r["seconds"] for r in rs])),
        }
        print(name, summary[name])
    if args.output:
        args.output.write_text(json.dumps({"dt_s": DT, "exposure_s": EXPOSURE, "n_tracks": N_TRACKS,
                                           "low_alpha": LOW_ALPHA, "replicates": args.replicates,
                                           "seed": args.seed, "summary": summary, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
