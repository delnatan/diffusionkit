"""Plots for the fBm (alpha, D_ref) profile fits (needs the `plots` extra). Pure functions: data in, a matplotlib Figure out."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ..gridpost.viz import _floor_band


def plot_alpha_D(fits: pl.DataFrame, ax: plt.Axes | None = None, max_bars: int | None = 150,
                 color: str = "#2a78d6", seed: int = 0) -> plt.Figure:
    """alpha_hat against D_ref_hat, one point per "ok" track whose alpha is identified, with likelihood-ratio
    intervals as error bars.

    `max_bars` draws the bars for a random subset (None: every track) so a large movie stays readable. Points
    whose alpha interval spans the whole grid are grey: those tracks say nothing about alpha. Tracks with no
    motion above the noise have no point. The grey band is the localization floor's 10-90% range
    (`D_ref_floor_um2_s`).
    """
    ok = fits.filter((pl.col("status") == "ok") & pl.col("alpha_hat").is_not_null())
    own = ax is None
    fig, ax = plt.subplots(figsize=(4.4, 3.8)) if own else (ax.figure, ax)
    if ok.height == 0:
        return fig
    _floor_band(ax, ok["D_ref_floor_um2_s"].to_numpy())
    x, y = ok["log10_D_ref_hat"].to_numpy(), ok["alpha_hat"].to_numpy()
    blank = (ok["alpha_edge"] == "both").fill_null(False).to_numpy()
    ax.scatter(x[blank], y[blank], s=3, color="0.72", lw=0, zorder=2)
    ax.scatter(x[~blank], y[~blank], s=3, color=color, alpha=.45, lw=0, zorder=2)
    sub = ok.filter(~pl.Series(blank))
    if max_bars is not None and sub.height > max_bars:
        sub = sub.sample(max_bars, seed=seed)
    sx, sy = sub["log10_D_ref_hat"].to_numpy(), sub["alpha_hat"].to_numpy()
    with np.errstate(divide="ignore"):
        lo = np.log10(sub["D_ref_lo_um2_s"].to_numpy())
    hi = np.log10(sub["D_ref_hi_um2_s"].to_numpy())
    lo = np.where(np.isfinite(lo), lo, np.minimum(sx, hi) - 3)  # bounded above only: run the bar off the plot
    ax.errorbar(sx, sy, xerr=[sx - lo, hi - sx], yerr=[sy - sub["alpha_lo"].to_numpy(), sub["alpha_hi"].to_numpy() - sy],
                fmt="none", ecolor=color, elinewidth=.4, alpha=.35, zorder=3)
    ax.scatter(sx, sy, s=6, color=color, lw=0, zorder=4)
    ax.axhline(1, color="0.35", lw=.5, ls=(0, (2, 3)))
    lo_x, hi_x = np.floor(np.quantile(x, .005)), np.ceil(np.quantile(x, .995))
    ax.set_xlim(lo_x, hi_x)
    ticks = np.arange(lo_x, hi_x + 1)
    ax.set_xticks(ticks, [f"$10^{{{int(t)}}}$" for t in ticks])
    ax.set_ylim(0, 2)
    ax.set_xlabel(r"$D_{ref}$ ($\mu m^2/s$)")
    ax.set_ylabel(r"$\alpha$")
    if own:
        fig.tight_layout()
    return fig
