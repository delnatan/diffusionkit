"""NPMLE vs Gaussian-smoothed EM vs MaxEnt+evidence, on the same synthetic datasets.

Usage: uv run python prototypes/demo_maxent.py [out.png]

Reuses demo_population.py's exact two 1000-track datasets (truth known):
  A. homogeneous: every track has D = 0.05.
  B. two subpopulations, D = 0.02 and 0.2 (SD 0.15 in ln D within each).
Per dataset, three distributions of D across tracks on the same grid:
  - unregularized NPMLE (posterior_1d.deconvolve, smooth=0): expected spiky.
  - the current default (posterior_1d.deconvolve, smooth=0.5): hand-tuned blur.
  - MaxEnt with alpha chosen by the Gull & Skilling evidence (maxent_deconvolve):
    no hand-tuned width; its bootstrap band fixes alpha_hat from the full
    dataset and only refits w per resample -- cheaper than rescanning evidence
    per replicate, at the cost of not propagating uncertainty in alpha itself.
A second row per dataset shows the evidence curve log P(data|alpha) vs alpha,
with the chosen alpha_hat marked, so the interior peak is visible directly.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import maxent_deconvolve as M
import posterior_1d as P
from demo_population import N_TRACKS, make_dataset

N_BOOT_G = 50
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
PURPLE, BLUE, AQUA = "#8b3fd6", "#2a78d6", "#1baf7a"  # NPMLE, smoothed EM, MaxEnt
PRIOR = P.log_uniform(1e-3, 1.)
DU = P.U[1] - P.U[0]
SUPPORT = np.isfinite(PRIOR)


def analyse(name, D, rng):
    _, _, lls = make_dataset(D, rng)
    n = len(lls)

    g_npmle = P.deconvolve(lls, PRIOR, smooth=0, iters=2000)
    g_em = P.deconvolve(lls, PRIOR)
    fit = M.maxent_deconvolve(lls, PRIOR)

    m = M.default_measure(int(SUPPORT.sum()))
    boot = [rng.integers(0, n, n) for _ in range(N_BOOT_G)]
    g_boot = np.zeros((N_BOOT_G, len(P.U)))  # zero outside SUPPORT, not empty -- garbage there blows out plot autoscale
    for i, b in enumerate(boot):
        Lb = np.exp(lls[b][:, SUPPORT] - lls[b][:, SUPPORT].max(axis=1, keepdims=True))
        w_b, _ = M.fit_maxent(Lb, m, fit.alpha)
        g_boot[i, SUPPORT] = w_b

    mid = np.log(np.sqrt(.02 * .2))
    print(f"\n{name}: true D mean {D.mean():.3f}")
    print(f"  alpha_hat {fit.alpha:.4g}  n_good {fit.n_good:.1f}  log evidence {fit.log_evidence:.1f}")
    print(f"  mass above D={np.exp(mid):.3f}  NPMLE {g_npmle[P.U > mid].sum():.3f}"
          f"  smoothed EM {g_em[P.U > mid].sum():.3f}  MaxEnt {fit.w[P.U > mid].sum():.3f}"
          f"  truth {np.mean(D > np.exp(mid)):.3f}")
    return dict(g_npmle=g_npmle, g_em=g_em, fit=fit, g_band=np.percentile(g_boot, [15.87, 84.13], axis=0))


def main(out: Path) -> None:
    rng = np.random.default_rng(21)
    homog = np.full(N_TRACKS, .05)
    modes = np.exp(np.log(rng.choice((.02, .2), N_TRACKS)) + .15 * rng.standard_normal(N_TRACKS))
    results = {"A. all tracks D = 0.05": analyse("A homogeneous", homog, rng),
               "B. two subpopulations (0.02, 0.2)": analyse("B two-mode", modes, rng)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), facecolor=SURFACE, layout="constrained",
                             gridspec_kw={"height_ratios": [2, 1.15]})
    Dg = np.exp(P.U)

    def style(a, logx=False):
        a.set_facecolor(SURFACE)
        a.tick_params(colors=INK2)
        if logx:
            a.set_xscale("log")
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            a.spines[side].set_color(GRID)
        a.grid(True, axis="x", color=GRID, lw=.8)

    for col, (title, r) in enumerate(results.items()):
        top, ev = axes[:, col]
        style(top, logx=True)
        style(ev, logx=True)
        truths = (.05,) if col == 0 else (.02, .2)
        for d in truths:
            top.axvline(d, color=INK, ls=(0, (4, 3)), lw=1.1, zorder=1)
        top.set_title(title, loc="left", color=INK, fontsize=12)
        top.plot(Dg, r["g_npmle"] / DU, color=PURPLE, lw=1.3, alpha=.8)
        top.plot(Dg, r["g_em"] / DU, color=BLUE, lw=1.6)
        top.fill_between(Dg, *(r["g_band"] / DU), color=AQUA, alpha=.25, lw=0)
        top.plot(Dg, r["fit"].w / DU, color=AQUA, lw=2)
        top.set_xlim(3e-3, 1)
        top.set_ylabel("distribution of D across tracks", color=INK2)
        top.set_xlabel("D (µm²/s)", color=INK2)
        if col == 0:
            top.legend(
                labels=["unregularized NPMLE", "smoothed EM (current default)",
                        "MaxEnt, evidence-chosen alpha (68% bootstrap band)"],
                handles=[plt.Line2D([], [], color=PURPLE, lw=1.3),
                         plt.Line2D([], [], color=BLUE, lw=1.6),
                         plt.Line2D([], [], color=AQUA, lw=2)],
                frameon=False, labelcolor=INK2, fontsize=8.5, loc="upper left")

        fit = r["fit"]
        ev.plot(fit.alpha_grid, fit.evidence_curve, color=AQUA, lw=1.6)
        ev.axvline(fit.alpha, color=INK, ls=":", lw=1.2)
        ev.set_ylabel("log evidence", color=INK2)
        ev.set_xlabel("alpha (regularization weight)", color=INK2)
        ev.text(.02, .05, f"alpha_hat = {fit.alpha:.3g}\nn_good = {fit.n_good:.1f}",
                transform=ev.transAxes, fontsize=8.5, color=INK2, va="bottom")

    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("maxent_view.png"))
