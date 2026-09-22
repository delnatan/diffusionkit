"""Ensemble view of many per-track alpha posteriors, against a known truth.

Usage: uv run python prototypes/demo_alpha_ensemble.py [out.png]

1000 simulated tracks (5-20 frames) from two subpopulations of alpha: one
Brownian (alpha=1), one subdiffusive (alpha=0.6), K shared. Left: every
track's posterior over alpha as one row of a heat map, sorted by its
median -- the same "a track's information is visible as how narrow and
bright its row is" view demo_ensemble.py gives D, here for the question
"is this motion Brownian?" instead of the calibrated z-score
diffusionkit.gridpost.likelihood's non-Brownian score answers with a single
number per track.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import posterior_alpha as P

DT, N_TRACKS = 0.035, 1000
ALPHA_MODES, SPREAD = (1.0, 0.6), 0.05  # equal weights; SD of alpha within a mode
K_TRUE = 0.05
SPLIT = np.mean(ALPHA_MODES)  # midpoint, to count "subdiffusive" tracks

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f7fc", "#a9c9f0", "#2a78d6", "#0f3d80"])


def simulate_ensemble(rng):
    alpha_true = np.clip(
        rng.choice(ALPHA_MODES, N_TRACKS) + SPREAD * rng.standard_normal(N_TRACKS),
        P.ALPHA[0], P.ALPHA[-1],
    )
    frames = rng.integers(5, 21, N_TRACKS)
    k_prior = P.log_uniform_K(1e-3, 1.0)
    post = np.empty((N_TRACKS, len(P.ALPHA)))
    for i in range(N_TRACKS):
        sd = rng.uniform(0.030, 0.045, frames[i])
        x = P.simulate(K_TRUE, alpha_true[i], sd, DT, rng)
        post[i] = P.track_alpha_posterior(x, sd, DT, k_prior)
    return alpha_true, frames, post


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)


def main(out: Path) -> None:
    rng = np.random.default_rng(3)
    alpha_true, frames, post = simulate_ensemble(rng)
    da = P.ALPHA[1] - P.ALPHA[0]
    med = np.array([P.quantile(p, 0.5) for p in post])

    print(f"{N_TRACKS} tracks; true fraction subdiffusive (alpha < {SPLIT:.2f}): "
          f"{np.mean(alpha_true < SPLIT):.3f}")
    print(f"{'group':<14}{'n':>5}{'true frac sub':>15}{'sum of posteriors':>19}{'medians':>9}")
    for name, m in (("all", frames > 0), ("5-7 frames", frames <= 7),
                    ("8-14 frames", (frames > 7) & (frames < 15)), ("15-20 frames", frames >= 15)):
        print(f"{name:<14}{m.sum():5d}{np.mean(alpha_true[m] < SPLIT):15.3f}"
              f"{post[m][:, P.ALPHA < SPLIT].sum(1).mean():19.3f}{np.mean(med[m] < SPLIT):9.3f}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.1]},
                                  facecolor=SURFACE, layout="constrained")
    for a in (ax, ax2):
        style(a)

    order = np.argsort(med)
    im = ax.imshow(post[order] / da, aspect="auto", origin="lower", cmap=SEQ, interpolation="nearest",
                   extent=[P.ALPHA[0], P.ALPHA[-1], 0, N_TRACKS], vmin=0, vmax=np.percentile(post / da, 99.5))
    for a in ALPHA_MODES:
        ax.axvline(a, color=INK, ls=(0, (4, 3)), lw=1.2)
    ax.set_xlabel(r"$\alpha$", color=INK2)
    ax.set_ylabel("tracks, sorted by posterior median", color=INK2)
    ax.set_title("Every track's posterior over alpha (one row each)", loc="left", color=INK, fontsize=12)
    cb = fig.colorbar(im, ax=ax, pad=.01, shrink=.8)
    cb.set_label(r"posterior density per $\alpha$", color=INK2)
    cb.ax.tick_params(colors=INK2)
    cb.outline.set_edgecolor(GRID)

    truth = sum(np.exp(-.5 * ((P.ALPHA - a) / SPREAD) ** 2) / (SPREAD * np.sqrt(2 * np.pi))
                for a in ALPHA_MODES) / 2
    ax2.plot(P.ALPHA, post.mean(0) / da, color=BLUE, lw=2, label="sum of posteriors")
    edges = np.arange(P.ALPHA[0], P.ALPHA[-1] + .05, .05)
    h, _ = np.histogram(med, bins=edges, density=True)
    ax2.stairs(h, edges, color=ORANGE, lw=2, label="histogram of medians")
    ax2.plot(P.ALPHA, truth, color=INK, lw=1.6, ls=(0, (4, 3)), label="true distribution")
    ax2.grid(True, color=GRID, lw=.8)
    ax2.set_xlabel(r"$\alpha$", color=INK2)
    ax2.set_ylabel(r"density per $\alpha$", color=INK2)
    ax2.set_title("Summing vs the truth", loc="left", color=INK, fontsize=12)
    ax2.legend(frameon=False, labelcolor=INK, loc="upper left")

    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("alpha_ensemble_view.png"))
