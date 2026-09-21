"""Watch one track's posterior over D sharpen as displacements are added.

Usage: uv run python prototypes/demo_sharpening.py [out.png]

A simulated 2D track (fine-step path, on_time blur, per-frame localization SDs)
starts from a prior set by the microscope's limits. Each row is the exact
posterior after the first k displacements; the right panel compares two priors
with the same limits but different edges.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import posterior_1d as P

D_TRUE, DT, ON_TIME, N_FRAMES = 0.05, 0.035, 0.020, 21  # um^2/s, s, s
D_LO, D_HI = 1e-3, 1.  # below: motion hides under localization noise; above: blurs past tracking at ~100 Hz
SHOW = (1, 2, 3, 5, 8, 12, 20)  # displacements to draw as rows

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2


def main(out: Path) -> None:
    rng = np.random.default_rng(7)
    sd = rng.uniform(.030, .045, N_FRAMES)  # um, per frame, from the Gaussian fit
    x = P.simulate(D_TRUE, sd, DT, ON_TIME, rng)

    priors = {"log-uniform": P.log_uniform(D_LO, D_HI), "log-normal": P.log_normal(D_LO, D_HI)}
    seqs = {name: P.sequence(x, sd, DT, lp, ON_TIME) for name, lp in priors.items()}
    sums = {name: [P.summary(p) for p in seq] for name, seq in seqs.items()}

    print(f"true D = {D_TRUE} um^2/s, dt = {DT} s, on_time = {ON_TIME} s, prior limits {D_LO}-{D_HI}")
    print(f"{'k':>3}  {'median':>8}  {'90% interval (log-uniform)':>28}  {'ratio hi/lo':>11}")
    for k, s in enumerate(sums["log-uniform"], 1):
        print(f"{k:>3}  {s['median']:8.4f}  {s['lo']:12.4f} - {s['hi']:<12.4f}  {s['hi'] / s['lo']:11.1f}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.25, 1]},
                                  facecolor=SURFACE, layout="constrained")
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        a.set_xscale("log") if a is ax else a.set_yscale("log")
        a.grid(True, color=GRID, lw=.8)
        a.tick_params(colors=INK2)
        for side in ("top", "right", "left" if a is ax else "bottom"):
            a.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            a.spines[side].set_color(GRID)

    # Left: ridgeline of density in ln D (common scale, so taller = sharper).
    du = P.U[1] - P.U[0]
    dens = [priors_p / du for priors_p in [P.posterior(np.zeros_like(P.U), priors["log-uniform"])]]
    dens += [seqs["log-uniform"][k - 1] / du for k in SHOW]
    gap = 1.
    scale = 2.2 * gap / max(d.max() for d in dens)
    D = np.exp(P.U)
    for i, d in enumerate(dens):
        base = -i * gap
        ax.fill_between(D, base, base + d * scale, color=BLUE, alpha=.28, lw=0, zorder=2 + i * .01)
        ax.plot(D, base + d * scale, color=BLUE, lw=1.6, zorder=3 + i * .01, solid_capstyle="round")
        ax.hlines(base, D[0], D[-1], color=GRID, lw=.8, zorder=1)
        if i:
            s = sums["log-uniform"][SHOW[i - 1] - 1]
            ax.text(1.01, base + .1, f"{s['median']:.3f}  [{s['lo']:.3f}, {s['hi']:.3f}]", transform=ax.get_yaxis_transform(),
                    color=INK2, fontsize=9, va="bottom")
    ax.axvline(D_TRUE, color=INK, ls=(0, (4, 3)), lw=1.4, zorder=5)
    ax.text(D_TRUE * 1.08, gap * .75, f"true D = {D_TRUE}", color=INK, fontsize=10, va="center")
    ax.set_yticks([-i * gap for i in range(len(dens))],
                  ["prior"] + [f"{k} displacement{'s' * (k > 1)}" for k in SHOW], color=INK)
    ax.set_xlim(3e-4, 3)
    ax.set_ylim(-(len(dens) - 1) * gap - .1, 2.6 * gap)
    ax.set_xlabel("D (µm²/s)", color=INK2)
    ax.set_title("Posterior of D as displacements accumulate", loc="left", color=INK, fontsize=12)
    ax.text(1.01, 1.0, "median  [90% interval]", transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")

    # Right: median and 90% interval versus k, for two priors with the same limits.
    ks = np.arange(1, N_FRAMES)
    for (name, ss), color in zip(sums.items(), (BLUE, ORANGE)):
        lo, hi, med = (np.array([s[key] for s in ss]) for key in ("lo", "hi", "median"))
        ax2.fill_between(ks, lo, hi, color=color, alpha=.16, lw=0)
        ax2.plot(ks, lo, color=color, lw=1, alpha=.7)
        ax2.plot(ks, hi, color=color, lw=1, alpha=.7)
        ax2.plot(ks, med, color=color, lw=2, label=f"{name} prior", solid_capstyle="round")
    ax2.hlines(D_TRUE, 1, N_FRAMES - 1, color=INK, ls=(0, (4, 3)), lw=1.4)
    ax2.text(N_FRAMES - .7, D_TRUE, "true D", color=INK, fontsize=10, ha="left", va="center")
    ax2.set_ylim(D_LO * .7, D_HI * 1.5)
    ax2.set_xlim(1, N_FRAMES + 1.6)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel("displacements used, k", color=INK2)
    ax2.set_ylabel("D (µm²/s): median and 90% interval", color=INK2)
    ax2.set_title("Two priors, same limits", loc="left", color=INK, fontsize=12)
    ax2.legend(frameon=False, loc="upper right", labelcolor=INK)

    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("posterior_sharpening.png"))
