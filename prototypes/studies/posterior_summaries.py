"""Posterior summaries: median vs mean of D vs mean of ln D. Accuracy and sensitivity to the prior's upper limit.

Usage: uv run python prototypes/studies/posterior_summaries.py [tracks_per_cell]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import posterior_1d as P

DT, TAU, SIG = .035, .020, .035
PRIORS = {"hi=1": P.log_uniform(1e-3, 1.), "hi=5": P.log_uniform(1e-3, 5.),
          "lo=3e-4": P.log_uniform(3e-4, 1.)}
DGRID = np.exp(P.U)


def summaries(p):
    return {"median": P.quantile(p, .5), "mean D": p @ DGRID, "mean lnD": np.exp(p @ P.U)}


def run(N, D, n_tracks, rng):
    sd = np.full(N, SIG)
    rec = {(pn, k): np.empty(n_tracks) for pn in PRIORS for k in ("median", "mean D", "mean lnD")}
    for t in range(n_tracks):
        ll = P.track_loglik(P.simulate(D, sd, DT, TAU, rng), sd, DT, TAU)
        for pn, lp in PRIORS.items():
            for k, v in summaries(P.posterior(ll, lp)).items():
                rec[pn, k][t] = v
    return rec


if __name__ == "__main__":
    n_tracks = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    rng = np.random.default_rng(5)
    print(f"{n_tracks} tracks per cell, prior hi=1 (default); ratios are D_hat / D_true")
    print(f"{'':16}{'summary':<10}{'mean':>6}{'median':>8}{'relRMSE':>9}{'RMS ln':>8}{'shift hi 1->5':>15}{'shift lo 1e-3->3e-4':>22}")
    for D in (.01, .05, .3):
        for N in (5, 10, 20, 40):
            rec = run(N, D, n_tracks, rng)
            for k in ("median", "mean D", "mean lnD"):
                r = rec["hi=1", k] / D
                shift = np.median(np.abs(np.log(rec["hi=5", k] / rec["hi=1", k])))
                shift_lo = np.median(np.abs(np.log(rec["lo=3e-4", k] / rec["hi=1", k])))
                lead = f"D={D:<5} N={N:<3}" if k == "median" else ""
                print(f"{lead:<16}{k:<10}{r.mean():6.2f}{np.median(r):8.2f}{np.sqrt(np.mean((r - 1)**2)):9.2f}"
                      f"{np.sqrt(np.mean(np.log(r)**2)):8.2f}{shift:15.3f}{shift_lo:22.3f}")
