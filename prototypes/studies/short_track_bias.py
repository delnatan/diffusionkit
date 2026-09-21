"""Short-track bias/accuracy: MLE (sigma known / fitted), CVE, posterior median. Simulated 2D tracks.

Usage: uv run python prototypes/studies/short_track_bias.py [tracks_per_cell]
"""
import sys
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import posterior_1d as P

DT, TAU, SIG = .035, .020, .035
R = TAU / (6 * DT)
UX = np.concatenate([[-np.inf], P.U])  # D = 0 plus the grid
PRIOR = P.log_uniform(1e-3, 1.)


def cve(x):
    d = np.diff(x, axis=0)
    return (np.mean(d**2) + 2 * np.mean(d[:-1] * d[1:])) / (2 * DT)


def mle_fit_sigma(x):
    """Joint MLE of (D >= 0, common sigma). B = sigma^2 B0 keeps the eigenvectors fixed."""
    d = np.diff(x, axis=0)
    lam, V = eigh(P.motion_cov(len(d), DT, TAU), P.localization_cov(np.ones(len(x))))
    ys = (V.T @ d).T

    def f(D, s):
        v = D[:, None, None] * lam + s[None, :, None]**2
        return -.5 * sum(np.sum(np.log(v) + y**2 / v, axis=2) for y in ys)

    D, s = np.concatenate([[0.], np.geomspace(1e-4, 10, 200)]), np.geomspace(.005, .3, 30)
    for _ in range(4):
        i, j = np.unravel_index(f(D, s).argmax(), (len(D), len(s)))
        D = np.linspace(D[max(i - 1, 0)], D[min(i + 1, len(D) - 1)], 15)
        s = np.geomspace(s[max(j - 1, 0)], s[min(j + 1, len(s) - 1)], 15)
    i, j = np.unravel_index(f(D, s).argmax(), (len(D), len(s)))
    return D[i], s[j]


def run(N, D, n_tracks, rng):
    sd = np.full(N, SIG)
    rec = {k: np.empty(n_tracks) for k in ("mle_known", "mle_fit", "cve", "post_med", "lo", "hi", "sig_fit")}
    for t in range(n_tracks):
        x = P.simulate(D, sd, DT, TAU, rng)
        ll = P.track_loglik(x, sd, DT, TAU, UX)
        rec["mle_known"][t] = np.exp(UX[np.argmax(ll)])
        s = P.summary(P.posterior(ll[1:], PRIOR))
        rec["post_med"][t], rec["lo"][t], rec["hi"][t] = s["median"], s["lo"], s["hi"]
        rec["cve"][t] = cve(x)
        rec["mle_fit"][t], rec["sig_fit"][t] = mle_fit_sigma(x)
    return rec


def row(name, a, D):
    r = a / D
    return (f"  {name:<10} mean {r.mean():5.2f} +-{r.std() / np.sqrt(len(r)):4.2f}  median {np.median(r):5.2f}  "
            f"relRMSE {np.sqrt(np.mean((r - 1)**2)):5.2f}  P(D^<=0) {np.mean(a <= 0):5.1%}")


if __name__ == "__main__":
    n_tracks = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rng = np.random.default_rng(11)
    t0 = time.time()
    print(f"dt={DT} on_time={TAU} sigma={SIG}  R={R:.3f}  {n_tracks} tracks per cell; ratios are D_hat / D_true")
    for D in (.01, .05, .3):
        print(f"\n=== D = {D}   reduced localization error x = sigma^2/(D dt) - 2R = {SIG**2 / (D * DT) - 2 * R:+.2f}")
        for N in (5, 10, 20, 40):
            rec = run(N, D, n_tracks, rng)
            print(f"N = {N} frames ({N - 1} steps)")
            for k, name in (("mle_known", "MLE sig known"), ("mle_fit", "MLE sig fit"), ("cve", "CVE"), ("post_med", "post median")):
                print(row(name, rec[k], D))
            cover = np.mean((rec["lo"] <= D) & (D <= rec["hi"]))
            print(f"  posterior 90% interval covers true D: {cover:.2f};  median hi/lo {np.median(rec['hi'] / rec['lo']):.1f};  "
                  f"fitted sigma median {np.median(rec['sig_fit']) / SIG:.2f} x true")
    print(f"\n{time.time() - t0:.0f} s")
