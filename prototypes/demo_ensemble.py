"""Ensemble views of many per-track posteriors, against a known truth.

Usage: uv run python prototypes/demo_ensemble.py [out.png]

1000 simulated tracks (5-20 frames) from two subpopulations of D a decade
apart. Left: every track's posterior as one row of a heat map, sorted by its
median, so a track's information is visible as how narrow and bright its row
is. Right: what you get by summing the posteriors, and by histogramming the
per-track medians, next to the true distribution of D.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import posterior_1d as P

DT, ON_TIME, N_TRACKS = 0.035, 0.020, 1000
D_MODES, SPREAD = (0.02, 0.2), 0.15  # um^2/s, equal weights; SD of ln D within a mode
SPLIT = np.sqrt(D_MODES[0] * D_MODES[1])  # geometric midpoint, to count "fast" tracks

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f7fc", "#a9c9f0", "#2a78d6", "#0f3d80"])


def simulate_ensemble(rng):
    D = np.exp(np.log(rng.choice(D_MODES, N_TRACKS)) + SPREAD * rng.standard_normal(N_TRACKS))
    frames = rng.integers(5, 21, N_TRACKS)
    prior = P.log_uniform(1e-3, 1.)
    post = np.empty((N_TRACKS, len(P.U)))
    for i in range(N_TRACKS):
        sd = rng.uniform(.030, .045, frames[i])
        post[i] = P.posterior(P.track_loglik(P.simulate(D[i], sd, DT, ON_TIME, rng), sd, DT, ON_TIME), prior)
    return D, frames, post


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)


def main(out: Path) -> None:
    rng = np.random.default_rng(3)
    D, frames, post = simulate_ensemble(rng)
    du = P.U[1] - P.U[0]
    med = np.array([P.quantile(p, .5) for p in post])
    q16 = np.array([P.quantile(p, .1587) for p in post])
    q84 = np.array([P.quantile(p, .8413) for p in post])

    print(f"{N_TRACKS} tracks; true fraction with D > {SPLIT:.3f}: {np.mean(D > SPLIT):.3f}")
    print(f"{'group':<14}{'n':>5}{'true frac fast':>16}{'sum of posteriors':>19}{'medians':>9}{'median q84/q16':>16}")
    for name, m in (("all", frames > 0), ("5-7 frames", frames <= 7), ("8-14 frames", (frames > 7) & (frames < 15)),
                    ("15-20 frames", frames >= 15)):
        print(f"{name:<14}{m.sum():5d}{np.mean(D[m] > SPLIT):16.3f}{post[m][:, P.U > np.log(SPLIT)].sum(1).mean():19.3f}"
              f"{np.mean(med[m] > SPLIT):9.3f}{np.median(q84[m] / q16[m]):16.1f}")

    # Width of the fast mode in ln D: truth vs the summed posteriors (mass above the split only).
    dens = post.mean(0)
    fast = P.U > np.log(SPLIT)
    m = (dens[fast] * P.U[fast]).sum() / dens[fast].sum()
    sd_sum = np.sqrt((dens[fast] * (P.U[fast] - m)**2).sum() / dens[fast].sum())
    sd_med = np.std(np.log(med[med > SPLIT]))
    print(f"fast mode, SD of ln D: truth {SPREAD:.2f}; sum of posteriors {sd_sum:.2f}; medians {sd_med:.2f}; "
          f"true D of the tracks whose median is fast {np.std(np.log(D[med > SPLIT])):.2f}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.1]},
                                  facecolor=SURFACE, layout="constrained")
    for a in (ax, ax2):
        style(a)

    # Left: one row per track.
    order = np.argsort(med)
    im = ax.imshow(post[order] / du, aspect="auto", origin="lower", cmap=SEQ, interpolation="nearest",
                   extent=[P.U[0], P.U[-1], 0, N_TRACKS], vmin=0, vmax=np.percentile(post / du, 99.5))
    ticks = [1e-3, 1e-2, 1e-1, 1.]
    for d in D_MODES:
        ax.axvline(np.log(d), color=INK, ls=(0, (4, 3)), lw=1.2)
    ax.set_xticks(np.log(ticks), [f"{t:g}" for t in ticks])
    ax.set_xlim(np.log(3e-4), np.log(3))
    ax.set_xlabel("D (µm²/s)", color=INK2)
    ax.set_ylabel("tracks, sorted by posterior median", color=INK2)
    ax.set_title("Every track's posterior (one row each)", loc="left", color=INK, fontsize=12)
    cb = fig.colorbar(im, ax=ax, pad=.01, shrink=.8)
    cb.set_label("posterior density per ln D", color=INK2)
    cb.ax.tick_params(colors=INK2)
    cb.outline.set_edgecolor(GRID)

    # Right: summed posteriors, histogram of medians, and the truth, as densities in ln D.
    truth = sum(np.exp(-.5 * ((P.U - np.log(d)) / SPREAD)**2) / (SPREAD * np.sqrt(2 * np.pi)) for d in D_MODES) / 2
    ax2.plot(np.exp(P.U), post.mean(0) / du, color=BLUE, lw=2, label="sum of posteriors")
    edges = np.arange(np.log(1e-3), np.log(1.) + .01, .3)
    h, _ = np.histogram(np.log(med), bins=edges, density=True)
    ax2.stairs(h * np.mean((med >= 1e-3) & (med <= 1.)), np.exp(edges), color=ORANGE, lw=2, label="histogram of medians")
    ax2.plot(np.exp(P.U), truth, color=INK, lw=1.6, ls=(0, (4, 3)), label="true distribution")
    ax2.set_xscale("log")
    ax2.set_xlim(3e-4, 3)
    ax2.grid(True, color=GRID, lw=.8)
    ax2.set_xlabel("D (µm²/s)", color=INK2)
    ax2.set_ylabel("density per ln D", color=INK2)
    ax2.set_title("Summing vs the truth", loc="left", color=INK, fontsize=12)
    ax2.legend(frameon=False, labelcolor=INK, loc="upper left")

    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("ensemble_view.png"))
