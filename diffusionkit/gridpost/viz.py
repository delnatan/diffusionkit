"""Plots for the D grid posteriors (needs the `plots` extra). Pure functions: data in, a matplotlib Figure out."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .composition import LengthComposition

FLOOR = dict(color="0.35", alpha=.13, lw=0)


def _floor_band(ax, floor_um2_s):
    """The localization floor's 10-90% band across tracks, and its median, on a log10 D axis."""
    lo, med, hi = np.log10(np.quantile(floor_um2_s, [.1, .5, .9]))
    ax.axvspan(lo, hi, **FLOOR)
    ax.axvline(med, color="0.35", lw=.8, ls=":")


def plot_by_track_length(comp: LengthComposition, floor_um2_s: np.ndarray | None = None,
                         cmap: str = "viridis") -> plt.Figure:
    """Top: the distribution of D per decade, stacked by track-length group. Bottom: each group's
    own distribution (normalized to its peak), with its tracks and detections at right.

    Columns are the pooled flat-prior posteriors and, when `comp` has a population, the
    deconvolved split (posterior mean over draws; `comp.band` and `comp.deconvolved` hold the
    uncertainty, best read as masses over a range rather than pointwise on the fine grid).
    `floor_um2_s` (e.g. the fits table's `D_floor_um2_s`) draws the localization floor's 10-90%
    band and median. Read below the floor as unresolved: those tracks only bound D from above.
    """
    x = comp.u / np.log(10)
    dx = x[1] - x[0]
    columns = [("pooled (flat-prior posteriors)", comp.pooled)]
    if comp.deconvolved is not None:
        columns.append(("deconvolved (posterior under the population)", comp.deconvolved.mean(0)))
    colors = plt.get_cmap(cmap)(np.linspace(.1, .9, len(comp.edges)))
    labels = comp.labels()
    fig, axes = plt.subplots(2, len(columns), figsize=(5.2 * len(columns), 6.4), squeeze=False,
                             sharex=True, gridspec_kw={"height_ratios": [1, 1.1]})
    for c, (title, contrib) in enumerate(columns):
        ax = axes[0, c]
        bottom = np.zeros(len(x))
        for j in range(len(comp.edges)):
            top = bottom + contrib[j] / dx
            ax.fill_between(x, bottom, top, color=colors[j], lw=0, label=labels[j])
            bottom = top
        if floor_um2_s is not None:
            _floor_band(ax, floor_um2_s)
        ax.set(ylabel=f"fraction of {comp.weight} per decade", title=title)
        ax.set_ylim(bottom=0)
        if c == len(columns) - 1:
            ax.legend(title="frames", fontsize=7, title_fontsize=7, loc="upper left", reverse=True)

        ax = axes[1, c]
        rows = contrib / np.maximum(contrib.max(1, keepdims=True), 1e-300)
        ax.imshow(rows, origin="lower", aspect="auto", cmap="Greys", interpolation="nearest",
                  extent=(x[0] - dx / 2, x[-1] + dx / 2, -.5, len(comp.edges) - .5))
        if floor_um2_s is not None:
            _floor_band(ax, floor_um2_s)
        ax.set_yticks(range(len(comp.edges)), labels)
        for j in range(len(comp.edges)):
            ax.text(1.01, j, f"{comp.n_tracks[j]} / {comp.n_detections[j]}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7)
        ax.set(xlabel="log10 D (um$^2$/s)", ylabel="track length (frames)",
               title="each group's own distribution (peak = 1); tracks / detections")
    fig.tight_layout()
    return fig
