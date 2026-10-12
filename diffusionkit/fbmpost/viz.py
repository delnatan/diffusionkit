"""Plots for the fBm (alpha, D) posteriors (needs the `plots` extra). Pure functions: data in, a matplotlib Figure out."""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import EllipseCollection

from ..gridpost.viz import _floor_band


def _ellipse_shapes(var_x, var_y, cov_xy):
    """Full width, height and angle (degrees) of the 1-SD ellipses of 2x2 covariances."""
    C = np.stack([np.stack([var_x, cov_xy], -1), np.stack([cov_xy, var_y], -1)], -2)
    ev, evec = np.linalg.eigh(C)
    ev = np.clip(ev, 0, None)
    return 2 * np.sqrt(ev[:, 1]), 2 * np.sqrt(ev[:, 0]), np.degrees(np.arctan2(evec[:, 1, 1], evec[:, 0, 1]))


def plot_alpha_D(fits: pl.DataFrame, ax: plt.Axes | None = None, max_ellipses: int | None = 150,
                 min_bits: float = .5, color: str = "#2a78d6", seed: int = 0) -> plt.Figure:
    """D_ref against alpha, one point per "ok" track, with posterior 1-SD ellipses as error bars.

    An ellipse's projections on the axes are +-1 marginal SD, like error bars, and its tilt is the
    track's own correlation between alpha and log10 D_ref, which crosses would hide. `max_ellipses`
    draws them for a random subset (None: every track) so a large movie stays readable; their
    opacity falls as 1/area, so every track puts about the same ink on the page. Points of tracks
    with less than `min_bits` of alpha information are grey: their alpha is the prior's. The grey
    band is the localization floor's 10-90% range (`D_ref_floor_um2_s`).
    """
    ok = fits.filter(pl.col("status") == "ok")
    own = ax is None
    fig, ax = plt.subplots(figsize=(4.4, 3.8)) if own else (ax.figure, ax)
    if ok.height == 0:
        return fig
    _floor_band(ax, ok["D_ref_floor_um2_s"].to_numpy())
    x, y = ok["log10_D_ref_post_mean"].to_numpy(), ok["alpha_post_mean"].to_numpy()
    informative = ok["alpha_post_info_bits"].to_numpy() >= min_bits
    ax.scatter(x[~informative], y[~informative], s=3, color="0.72", lw=0, zorder=2)
    ax.scatter(x[informative], y[informative], s=3, color=color, alpha=.45, lw=0, zorder=2)
    sub = ok if max_ellipses is None or ok.height <= max_ellipses else ok.sample(max_ellipses, seed=seed)
    sd_a, sd_d = sub["alpha_post_sd"].to_numpy(), sub["log10_D_ref_post_sd"].to_numpy()
    corr = np.nan_to_num(sub["alpha_log10_D_ref_corr"].to_numpy())
    w, h, angle = _ellipse_shapes(sd_d ** 2, sd_a ** 2, corr * sd_a * sd_d)
    opacity = .5 * np.minimum(1, .03 / np.maximum(np.pi / 4 * w * h, 1e-12))
    rgba = np.tile(mcolors.to_rgba(color), (sub.height, 1))
    face, edge = rgba.copy(), rgba.copy()
    face[:, 3], edge[:, 3] = .5 * opacity, np.minimum(1, 1.5 * opacity)
    ax.add_collection(EllipseCollection(w, h, angle, units="xy", offsets=np.c_[sub["log10_D_ref_post_mean"],
                                                                                 sub["alpha_post_mean"]],
                                        offset_transform=ax.transData, facecolors=face, edgecolors=edge,
                                        linewidths=.4, zorder=3))
    ax.scatter(sub["log10_D_ref_post_mean"], sub["alpha_post_mean"], s=6, color=color, lw=0, zorder=4)
    ax.axhline(1, color="0.35", lw=.5, ls=(0, (2, 3)))
    lo, hi = np.floor(np.quantile(x, .005)), np.ceil(np.quantile(x, .995))
    ax.set_xlim(lo, hi)
    ticks = np.arange(lo, hi + 1)
    ax.set_xticks(ticks, [f"$10^{{{int(t)}}}$" for t in ticks])
    ax.set_ylim(0, 2)
    ax.set_xlabel(r"$D_{ref}$ ($\mu m^2/s$)")
    ax.set_ylabel(r"$\alpha$")
    if own:
        fig.tight_layout()
    return fig
