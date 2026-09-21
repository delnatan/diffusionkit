"""Population-level comparators: one shared D (all data) and a distribution of D, vs an ensemble MSD fit.

Usage: uv run python prototypes/demo_population.py [out.png]

Two simulated 1000-track datasets (5-20 frames), truth known:
  A. homogeneous: every track has D = 0.05.
  B. two subpopulations, D = 0.02 and 0.2 (SD 0.15 in ln D within each).
Per dataset:
  - shared-D posterior from all tracks (`pooled`), with its own 68% interval and the 68%
    interval from resampling whole tracks (which sees between-track variation);
  - the distribution of D across tracks (`deconvolve`), with a track-bootstrap band;
  - a classic ensemble MSD fit: MSD(k dt) = 4 D k dt + c over lags 1-4, with bootstrap interval.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import posterior_1d as P

DT, ON_TIME, N_TRACKS = 0.035, 0.020, 1000
N_BOOT_POOLED, N_BOOT_MSD, N_BOOT_G, LAGS = 300, 300, 50, 4
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"  # shared-D posterior, distribution of D, MSD fit
PRIOR = P.log_uniform(1e-3, 1.)
DU = P.U[1] - P.U[0]


def make_dataset(D, rng):
    """Positions, SDs and per-track log-likelihoods (on P.U) for tracks with true diffusion coefficients D (n,)."""
    frames = rng.integers(5, 21, len(D))
    tracks, sds, lls = [], [], np.empty((len(D), len(P.U)))
    for i, (d, n) in enumerate(zip(D, frames)):
        sd = rng.uniform(.030, .045, n)
        x = P.simulate(d, sd, DT, ON_TIME, rng)
        tracks.append(x)
        sds.append(sd)
        lls[i] = P.track_loglik(x, sd, DT, ON_TIME)
    return tracks, sds, lls


def msd_fit(tracks):
    """Classic ensemble time-averaged MSD, linear fit of 4 D tau + c over lags 1..LAGS."""
    sums, counts = np.zeros(LAGS), np.zeros(LAGS)
    for x in tracks:
        for k in range(1, LAGS + 1):
            if len(x) > k:
                sums[k - 1] += np.sum((x[k:] - x[:-k])**2)
                counts[k - 1] += len(x) - k
    slope = np.polyfit(np.arange(1, LAGS + 1) * DT, sums / counts, 1)[0]
    return slope / 4


def analyse(name, D, rng):
    tracks, sds, lls = make_dataset(D, rng)
    n = len(tracks)
    boot = [rng.integers(0, n, n) for _ in range(max(N_BOOT_POOLED, N_BOOT_MSD))]

    # With many tracks the shared-D posterior is narrower than P.U's 2.3% step, so evaluate it
    # on a fine grid around the coarse peak. (The deconvolution is fine on the coarse grid.)
    centre = np.log(P.quantile(P.pooled(lls, PRIOR), .5))
    u_fine = centre + np.linspace(-.15, .15, 601)
    lls_fine = np.array([P.track_loglik(x, s, DT, ON_TIME, u_fine) for x, s in zip(tracks, sds)])
    prior_fine = P.log_uniform(1e-3, 1., u_fine)
    p = P.pooled(lls_fine, prior_fine)
    post = P.summary(p, u_fine, level=.6827)
    med_boot = np.array([P.quantile(P.pooled(lls_fine[b], prior_fine), .5, u_fine) for b in boot[:N_BOOT_POOLED]])
    msd = msd_fit(tracks)
    msd_boot = np.array([msd_fit([tracks[i] for i in b]) for b in boot[:N_BOOT_MSD]])
    g = P.deconvolve(lls, PRIOR)
    g_boot = np.array([P.deconvolve(lls[b], PRIOR) for b in boot[:N_BOOT_G]])

    q = lambda a: np.percentile(a, [15.87, 84.13])  # noqa: E731
    print(f"\n{name}: true D mean {D.mean():.3f}, geometric mean {np.exp(np.log(D).mean()):.3f}")
    print(f"  shared-D posterior : median {post['median']:.4f}  68% [{post['lo']:.4f}, {post['hi']:.4f}]  (posterior)")
    print(f"                       bootstrap over tracks: [{q(med_boot)[0]:.4f}, {q(med_boot)[1]:.4f}]")
    print(f"  ensemble MSD fit   : D {msd:.4f}  bootstrap 68% [{q(msd_boot)[0]:.4f}, {q(msd_boot)[1]:.4f}]")
    return dict(D=D, p=p, u_fine=u_fine, post=post, med_boot=q(med_boot), msd=msd, msd_ci=q(msd_boot), g=g,
                g_band=np.percentile(g_boot, [15.87, 84.13], axis=0))


def main(out: Path) -> None:
    rng = np.random.default_rng(21)
    homog = np.full(N_TRACKS, .05)
    modes = np.exp(np.log(rng.choice((.02, .2), N_TRACKS)) + .15 * rng.standard_normal(N_TRACKS))
    results = {"A. all tracks D = 0.05": analyse("A homogeneous", homog, rng),
               "B. two subpopulations (0.02, 0.2)": analyse("B two-mode", modes, rng)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), facecolor=SURFACE, layout="constrained",
                             gridspec_kw={"height_ratios": [2, 1.15]})
    Dg = np.exp(P.U)

    def style(a, log=False):
        a.set_facecolor(SURFACE)
        a.tick_params(colors=INK2)
        if log:
            a.set_xscale("log")
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            a.spines[side].set_color(GRID)
        a.grid(True, axis="x", color=GRID, lw=.8)

    for col, (title, r) in enumerate(results.items()):
        top, zoom = axes[:, col]
        style(top, log=True)
        style(zoom)
        truths = (.05,) if col == 0 else (.02, .2)
        for d in truths:
            top.axvline(d, color=INK, ls=(0, (4, 3)), lw=1.1, zorder=1)
        top.set_title(title, loc="left", color=INK, fontsize=12)
        top.fill_between(Dg, *(r["g_band"] / DU), color=AQUA, alpha=.28, lw=0)
        top.plot(Dg, r["g"] / DU, color=AQUA, lw=2)
        top.set_xlim(3e-3, 1)
        top.set_ylabel("distribution of D across tracks", color=INK2)
        top.set_xlabel("D (µm²/s)", color=INK2)
        top.legend(handles=[Patch(color=AQUA, alpha=.4, lw=0, label="deconvolved, 68% bootstrap band"),
                            Line2D([], [], marker="^", ls="", color=BLUE, label="shared-D posterior median"),
                            Line2D([], [], marker="o", ls="", color=ORANGE, label="ensemble MSD fit")],
                   frameon=False, labelcolor=INK2, fontsize=8.5, loc="upper left")
        # Where the single-D estimates sit on the same axis.
        top.plot([r["post"]["median"]], [0], "^", color=BLUE, ms=9, clip_on=False, zorder=4)
        top.plot([r["msd"]], [0], "o", color=ORANGE, ms=7, clip_on=False, zorder=5)

        # Zoom: shared-D posterior, its resampling interval, and the MSD fit, on a linear axis.
        ends = [r["post"]["lo"], r["post"]["hi"], *r["med_boot"], *r["msd_ci"], r["D"].mean()]
        pad = .5 * (max(ends) - min(ends))
        zoom.set_xlim(min(ends) - pad, max(ends) + pad)
        Df, dens = np.exp(r["u_fine"]), r["p"] / r["p"].max()
        zoom.fill_between(Df, dens, color=BLUE, alpha=.25, lw=0)
        zoom.plot(Df, dens, color=BLUE, lw=2)
        zoom.hlines(-.18, *r["med_boot"], color=BLUE, lw=5, clip_on=False)
        zoom.errorbar([r["msd"]], [-.5], xerr=[[r["msd"] - r["msd_ci"][0]], [r["msd_ci"][1] - r["msd"]]], fmt="o", color=ORANGE,
                      ms=7, capsize=5, lw=2, clip_on=False)
        zoom.axvline(r["D"].mean(), color=INK, ls=":", lw=1.4)
        zoom.set_ylim(-.7, 1.1)
        zoom.set_yticks([])
        zoom.set_xlabel("D (µm²/s), zoomed", color=INK2)
        kw = dict(fontsize=9, color=INK2, transform=zoom.get_yaxis_transform(), va="center")
        zoom.text(.012, .93, "curve: shared-D posterior", **kw)
        zoom.text(.012, -.36, "bar: its 68% interval by resampling tracks", **kw)
        zoom.text(.012, -.62, "point: ensemble MSD fit, 68% bootstrap", **kw)
        label = "dotted: true mean D" if col == 1 else "dotted: true D"
        zoom.text(.988, .93, label, ha="right", **kw)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("population_view.png"))
