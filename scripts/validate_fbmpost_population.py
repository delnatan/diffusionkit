"""Recovery of a population of (alpha, D_ref) from simulated tracks (diffusionkit.fbmpost.population).

Gelatin-like acquisition (dt 44 ms, exposure 20 ms, SD 13 nm, tau_ref 0.1 s), tracks of 5-80 frames
(log-uniform), D_ref straddling the localization floor. Truths (`--truth`):

- two_modes: alpha ~ 0.6 N(0.45, 0.1) + 0.4 N(1.0, 0.1), log10 D_ref = floor + 0.25 + (alpha - 0.67) + N(0, 0.4):
  two modes, alpha correlated with D.
- homogeneous: alpha = 0.6 for every track, log10 D_ref = floor + 0.25 + N(0, 0.5).

Prints the truth of the simulated sample, the 2D population's 5/50/95% over its draws, and, for comparison, a
1D deconvolution of the tracks' alpha curves (posteriors, which the amplitude prior tilts).
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.fbmpost import FbmPostOptions, analyze_tracks, deconvolve_tracks, simulate_tracks
from diffusionkit.gridpost.deconvolve import deconvolve

ACQ, SIG, OPT = Acquisition(.044, .02), .013, FbmPostOptions()
FLOOR = np.log10(SIG ** 2 / OPT.tau_ref_s)
CUT = .75


def draw_truth(kind: str, n: int, rng: np.random.Generator):
    if kind == "two_modes":
        low = rng.random(n) < .6
        alpha = np.clip(np.where(low, rng.normal(.45, .1, n), rng.normal(1., .1, n)), .05, 1.9)
        log10_D = FLOOR + .25 + (alpha - .67) + rng.normal(0, .4, n)
    elif kind == "homogeneous":
        alpha = np.full(n, .6)
        log10_D = FLOOR + .25 + rng.normal(0, .5, n)
    else:
        raise ValueError(kind)
    frames = np.round(np.exp(rng.uniform(np.log(5), np.log(80), n))).astype(int)
    return alpha, log10_D, frames


def q(x):
    return "  ".join(f"{v:6.3f}" for v in np.quantile(x, [.05, .5, .95]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", default="two_modes", choices=["two_modes", "homogeneous"])
    ap.add_argument("--n", type=int, default=1500, help="tracks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    alpha, log10_D, frames = draw_truth(args.truth, args.n, rng)
    table = pl.concat([simulate_tracks(1, int(f), float(a), float(10 ** d), SIG, ACQ, OPT.tau_ref_s, rng,
                                       first_track_id=i)
                       for i, (a, d, f) in enumerate(zip(alpha, log10_D, frames))])
    t0 = time.perf_counter()
    with ThreadPoolExecutor() as pool:
        result = analyze_tracks(table, ACQ, OPT, keep_posteriors=True, keep_likelihoods=True, map_fn=pool.map)
    t1 = time.perf_counter()
    pop = deconvolve_tracks(result.likelihoods)
    t2 = time.perf_counter()
    print(f"{args.truth}, {args.n} tracks, seed {args.seed}: posteriors {t1 - t0:.0f} s, population {t2 - t1:.0f} s, "
          f"lam {pop.lam[0]:.3g}, {pop.lam[1]:.3g}, log10 D_ref cells {pop.log10_D_ref[0]:.2f} to {pop.log10_D_ref[-1]:.2f}")
    above = log10_D > pop.log10_floor
    print(f"{'':32s}{'truth':>8s}   2D population 5/50/95%")
    rows = [
        (f"P(alpha < {CUT})", np.mean(alpha < CUT), pop.p_alpha_below(CUT)),
        (f"P(alpha < {CUT} | above floor)", np.mean(alpha[above] < CUT), pop.p_alpha_below(CUT, above_floor=True)),
        ("mean alpha", alpha.mean(), pop.alpha_mean()),
        ("SD of alpha", alpha.std(), pop.alpha_sd()),
        ("corr(alpha, log10 D_ref)", np.corrcoef(alpha, log10_D)[0, 1] if alpha.std() else 0., pop.corr()),
        ("corr, above floor", np.corrcoef(alpha[above], log10_D[above])[0, 1] if alpha.std() else 0.,
         pop.corr(above_floor=True)),
        ("share below floor", 1 - above.mean(), pop.mass_below_floor()),
    ]
    for name, truth, draws in rows:
        print(f"  {name:30s}{truth:8.3f}   {q(draws)}")
    a = OPT.alphas()
    d1 = deconvolve(result.posteriors.log_post_alpha, a)
    share = np.clip((CUT - (a - .01)) / .02, 0, 1)
    print(f"  {'1D on alpha curves: P(a<cut)':30s}{np.mean(alpha < CUT):8.3f}   {q(d1.samples @ share)}")


if __name__ == "__main__":
    main()
