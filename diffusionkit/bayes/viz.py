"""Diagnostic plots for the likelihood-based MLE/Bayesian pipeline.

Same conventions as `analysis.viz`: pure functions, polars/numpy data in, a
matplotlib Figure out, no shared state. A few plots here are deliberately
generic (column names as arguments) rather than hardcoded, since the same
comparison (fit vs. ground truth, estimator A vs. estimator B) gets reused
across the normal-diffusion and anomalous-diffusion tables.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import norm


def samples_dict_to_arrays(
    samples: dict[str, np.ndarray], param_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a numpyro `get_samples(group_by_chain=True)`-style dict
    (each value shape (n_chains, n_samples)) into the two array shapes
    `plot_posterior_corner`/`plot_mcmc_trace` expect: a flat
    (n_chains*n_samples, dim) array for the corner plot, and a
    (n_samples, n_chains, dim) array for the trace plot.
    """
    stacked = np.stack([samples[name] for name in param_names], axis=-1)  # (chains, samples, dim)
    flat = stacked.reshape(-1, stacked.shape[-1])
    trace = np.moveaxis(stacked, 0, 1)  # (samples, chains, dim)
    return flat, trace


def plot_posterior_corner(
    samples: np.ndarray,
    param_names: list[str],
    param_units: list[str] | None = None,
    truth: np.ndarray | None = None,
    laplace_mean: np.ndarray | None = None,
    laplace_cov: np.ndarray | None = None,
) -> plt.Figure:
    """Pairwise posterior scatter + 1D marginal histograms (a hand-rolled
    'corner plot', no `corner` package dependency).

    If `laplace_mean`/`laplace_cov` are given, overlays the Laplace
    (Gaussian) approximation used for the full per-track batch fit
    (`inference.fit_map`, `inference.fit_table_svi`) as a 1-2 sigma ellipse
    on each 2D panel and a Gaussian curve on each 1D panel -- the visual
    check that the fast approximation used everywhere else is trustworthy
    where it's applied. `samples` is a flat (n_samples, dim) array; use
    `samples_dict_to_arrays` to build one from numpyro's sample dict.
    """
    d = samples.shape[1]
    units = param_units or [""] * d
    fig, axes = plt.subplots(d, d, figsize=(2.6 * d, 2.6 * d))
    if d == 1:
        axes = np.array([[axes]])

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples[:, i], bins=40, color="steelblue", density=True, alpha=0.75)
                if laplace_mean is not None and laplace_cov is not None:
                    sd = np.sqrt(laplace_cov[i, i])
                    xs = np.linspace(*ax.get_xlim(), 200)
                    ax.plot(xs, norm.pdf(xs, laplace_mean[i], sd), color="crimson", lw=1.5,
                            label="Laplace approx.")
                    if i == 0:
                        ax.legend(frameon=False, fontsize=7)
                if truth is not None:
                    ax.axvline(truth[i], color="0.2", lw=1, ls="--")
            else:
                ax.scatter(samples[:, j], samples[:, i], s=3, alpha=0.15, color="0.3", rasterized=True)
                if laplace_mean is not None and laplace_cov is not None:
                    _plot_gaussian_ellipse(
                        ax, laplace_mean[[j, i]], laplace_cov[np.ix_([j, i], [j, i])], color="crimson"
                    )
                if truth is not None:
                    ax.scatter([truth[j]], [truth[i]], color="0.2", marker="+", s=80, zorder=5)
            if i == d - 1:
                lbl = param_names[j] + (f" ({units[j]})" if units[j] else "")
                ax.set_xlabel(lbl, fontsize=9)
            if j == 0 and i > 0:
                lbl = param_names[i] + (f" ({units[i]})" if units[i] else "")
                ax.set_ylabel(lbl, fontsize=9)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Posterior samples (n={samples.shape[0]})", fontsize=10)
    fig.tight_layout()
    return fig


def _plot_gaussian_ellipse(ax, mean, cov, color, n_std=(1, 2)):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for n in n_std:
        width, height = 2 * n * np.sqrt(np.clip(vals, 0, None))
        ellipse = Ellipse(
            mean, width, height, angle=theta, edgecolor=color, facecolor="none", lw=1.2, alpha=0.8
        )
        ax.add_patch(ellipse)


def plot_mcmc_trace(
    chain: np.ndarray, param_names: list[str], param_units: list[str] | None = None
) -> plt.Figure:
    """Per-parameter NUTS chain traces (chain: n_samples x n_chains x dim,
    see `samples_dict_to_arrays`), to visually confirm the chains have mixed
    rather than drifted or gotten stuck (complements numpyro's own r_hat/
    n_eff diagnostics from `mcmc.print_summary()` with a direct look)."""
    d = chain.shape[2]
    units = param_units or [""] * d
    fig, axes = plt.subplots(d, 1, figsize=(7, 2.2 * d), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(d):
        axes[i].plot(chain[:, :, i], alpha=0.7, lw=0.5)
        lbl = param_names[i] + (f" ({units[i]})" if units[i] else "")
        axes[i].set_ylabel(lbl, fontsize=9)
    axes[-1].set_xlabel("NUTS sample")
    fig.suptitle("NUTS chain traces", fontsize=10)
    fig.tight_layout()
    return fig


def plot_estimator_scatter(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    log: bool = False,
    unity: bool = True,
    display_quantiles: tuple[float, float] | None = (0.01, 0.99),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Generic x-vs-y scatter with an optional unity line -- the workhorse
    for estimator-vs-estimator and fit-vs-ground-truth comparisons.

    `r` is always computed on every point. `display_quantiles` only clips
    the axis *view* (not the data) to the given quantile range of x and y
    combined -- protects the plot from a couple of genuinely extreme points
    (e.g. a flat-prior fit occasionally collapsing to a near-zero D on a
    poorly-constrained short track) flattening the view of everything else.
    Pass None to autoscale to the full data range as usual.
    """
    fig = ax.figure if ax is not None else plt.figure(figsize=(5, 5))
    ax = ax or fig.gca()
    sub = df.select([x_col, y_col]).drop_nulls()
    x, y = sub[x_col].to_numpy(), sub[y_col].to_numpy()
    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else float("nan")
    ax.scatter(x, y, s=10, alpha=0.5, color="steelblue")

    n_dropped = 0
    if display_quantiles is not None:
        combined = np.concatenate([x, y])
        lo, hi = np.quantile(combined, display_quantiles)
        in_view = (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)
        n_dropped = int((~in_view).sum())
        pad = 0.05 * (hi - lo) if hi > lo else 0.05
        ax.set_xlim(lo - (0 if log else pad), hi + pad)
        ax.set_ylim(lo - (0 if log else pad), hi + pad)
    else:
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())

    if unity:
        pad = 0.05 * (hi - lo) if hi > lo else 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.3", lw=1, ls="--", label="unity")
        ax.legend(frameon=False, fontsize=9)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    subtitle = f"{title} (r={r:.3f}, n={len(x)}"
    subtitle += f", {n_dropped} outside display range)" if n_dropped else ")"
    ax.set_title(subtitle)
    fig.tight_layout()
    return fig


def plot_D_alpha_joint(
    df: pl.DataFrame,
    D_col: str,
    alpha_col: str,
    D_label: str,
    alpha_label: str,
    title: str,
    display_quantiles: tuple[float, float] = (0.01, 0.99),
) -> plt.Figure:
    """KDE + scatter + marginals of log10(D) vs alpha -- same diagnostic as
    `analysis.viz.plot_D_alpha_jointplot`, generalized to any (D, alpha)
    column pair so it can be pointed at the classic MSD fit, a flat-prior
    fit, or an informative-prior Bayes MAP table for a direct visual check
    of D-alpha correlation across tracks.

    The reported correlation `r` always uses every eligible track (D>0), no
    matter how extreme -- an unregularized (flat-prior) fit can genuinely
    send D to a near-zero value for a few of the shortest, most weakly-
    constrained tracks, a real optimizer degeneracy rather than a plotting
    artifact. But letting 1-2 such points set the axis range would flatten
    the view of everything else, so the *display* range (not the data or
    `r`) is clipped to `display_quantiles`, with the excluded count reported
    in the title.
    """
    sub = df.filter(pl.col(D_col) > 0).select(
        pl.col(D_col).log10().alias("log10_D"), pl.col(alpha_col).alias("alpha")
    ).drop_nulls()
    x_all, y_all = sub["log10_D"].to_numpy(), sub["alpha"].to_numpy()
    r = np.corrcoef(x_all, y_all)[0, 1]

    x_lo, x_hi = np.quantile(x_all, display_quantiles)
    y_lo, y_hi = np.quantile(y_all, display_quantiles)
    in_view = (x_all >= x_lo) & (x_all <= x_hi) & (y_all >= y_lo) & (y_all <= y_hi)
    n_dropped = int((~in_view).sum())
    pdf = pd.DataFrame({"log10_D": x_all[in_view], "alpha": y_all[in_view]})

    g = sns.JointGrid(data=pdf, x="log10_D", y="alpha", height=6, ratio=4)
    sns.kdeplot(data=pdf, x="log10_D", y="alpha", ax=g.ax_joint, fill=True, cmap="Blues",
                alpha=0.6, thresh=0.05, levels=12, zorder=0)
    sns.scatterplot(data=pdf, x="log10_D", y="alpha", ax=g.ax_joint, s=18, alpha=0.6,
                     color="0.15", edgecolor="none", zorder=1)
    g.ax_joint.axhline(1.0, color="crimson", lw=1, ls="--", label=r"$\alpha=1$ (Brownian)", zorder=2)
    g.ax_marg_x.hist(pdf["log10_D"], bins=30, color="steelblue", edgecolor="white")
    g.ax_marg_y.hist(pdf["alpha"], bins=30, color="steelblue", edgecolor="white", orientation="horizontal")

    g.ax_joint.set_xlabel(rf"$\log_{{10}}$ {D_label}")
    g.ax_joint.set_ylabel(alpha_label)
    g.ax_joint.legend(frameon=False, loc="upper left")
    subtitle = f"{title} (r={r:.2f}, n={len(x_all)}"
    if n_dropped:
        subtitle += f", {n_dropped} outside {int(100*(display_quantiles[1]-display_quantiles[0]))}% display range"
    subtitle += ")"
    g.ax_marg_x.set_title(subtitle, fontsize=10, loc="left")
    return g.figure


def _joint_panel(
    fig: plt.Figure,
    gs: object,
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    """One (marginal-x / joint / marginal-y) panel of `plot_classic_vs_bayes_joint`,
    drawn into a 2x2 sub-region of `fig` via a pre-positioned `GridSpec` `gs`
    (`gs[0,0]`/`gs[1,0]`/`gs[1,1]` = top marginal / joint / right marginal) --
    hand-built rather than `sns.JointGrid` (which always owns its own whole
    Figure) so two panels can share one Figure and, critically, identical
    `xlim`/`ylim` set explicitly on both rather than each panel auto-scaling
    to its own data.
    """
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    pdf = pd.DataFrame({"x": x, "y": y})
    sns.kdeplot(data=pdf, x="x", y="y", ax=ax_joint, fill=True, cmap="Blues",
                alpha=0.6, thresh=0.05, levels=12, zorder=0)
    sns.scatterplot(data=pdf, x="x", y="y", ax=ax_joint, s=18, alpha=0.6,
                     color="0.15", edgecolor="none", zorder=1)
    ax_joint.axhline(1.0, color="crimson", lw=1, ls="--", zorder=2,
                      label=r"$\alpha=1$ (Brownian)")
    ax_joint.set_xlim(xlim)
    ax_joint.set_ylim(ylim)
    ax_joint.set_xlabel(x_label)
    ax_joint.set_ylabel(y_label)
    ax_joint.legend(frameon=False, loc="upper left", fontsize=8)

    ax_marg_x.hist(pdf["x"], bins=30, range=xlim, color="steelblue", edgecolor="white")
    ax_marg_y.hist(pdf["y"], bins=30, range=ylim, color="steelblue", edgecolor="white",
                    orientation="horizontal")
    ax_marg_x.tick_params(labelbottom=False)
    ax_marg_y.tick_params(labelleft=False)
    ax_marg_x.set_ylabel("")
    ax_marg_y.set_xlabel("")

    r = np.corrcoef(x, y)[0, 1]
    n_out = int(((x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1])).sum())
    subtitle = f"{title}\nr={r:.2f}, n={len(x)}"
    if n_out:
        subtitle += f" ({n_out} outside shared display range)"
    ax_marg_x.set_title(subtitle, fontsize=9.5, loc="left")


def plot_classic_vs_bayes_joint(
    df: pl.DataFrame,
    D_classic_col: str = "D_classic_um2_s",
    alpha_classic_col: str = "alpha_classic",
    D_bayes_col: str = "D_median_um2_s",
    alpha_bayes_col: str = "alpha",
    display_quantiles: tuple[float, float] = (0.01, 0.99),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    """Classic MSD-fit vs. exact-likelihood Bayes MAP, log10(D) vs. alpha,
    side by side on identical axis limits -- the direct visual counterpart to
    `run_bayes_analysis.py`'s printed r(D_classic, D_bayes)/r(alpha_classic,
    alpha_bayes) agreement numbers, and to `plot_D_alpha_joint` /
    `analysis.viz.plot_D_alpha_jointplot` run individually per-estimator
    (each of which auto-scales its own axes, so those two are not
    visually comparable side by side without this).

    `df` is expected to already have both estimators on the same tracks,
    e.g. `results/tables/bayes/bayes_vs_classic_comparison.csv`
    (`run_bayes_analysis.py`'s inner join of its own summary against the
    classic pipeline's saved `per_track_msd_fits.csv`). Only rows with both
    D columns > 0 are used (needed for log10); the shared xlim/ylim are the
    `display_quantiles` of the *pooled* log10(D) and alpha values across both
    estimators together (not each estimator's own range), so a difference in
    spread between the two is visible rather than hidden by independent
    autoscaling -- points outside that shared window still count toward each
    panel's own r/n, just clipped from the view (count reported in the
    subtitle), same convention as `plot_D_alpha_joint`. Pass `xlim`/`ylim`
    explicitly (e.g. `ylim=(0, 2)` to match alpha's actual (0,2) prior
    support, `xlim=(-3, 1)` for a fixed log10(D) decade range) to override
    the data-driven quantile default with fixed physical bounds instead.
    """
    sub = df.filter((pl.col(D_classic_col) > 0) & (pl.col(D_bayes_col) > 0)).select(
        pl.col(D_classic_col).log10().alias("log10_D_classic"),
        pl.col(alpha_classic_col).alias("alpha_classic"),
        pl.col(D_bayes_col).log10().alias("log10_D_bayes"),
        pl.col(alpha_bayes_col).alias("alpha_bayes"),
    ).drop_nulls()

    x_classic, y_classic = sub["log10_D_classic"].to_numpy(), sub["alpha_classic"].to_numpy()
    x_bayes, y_bayes = sub["log10_D_bayes"].to_numpy(), sub["alpha_bayes"].to_numpy()

    if xlim is None:
        x_pool = np.concatenate([x_classic, x_bayes])
        xlim = tuple(np.quantile(x_pool, display_quantiles))
    if ylim is None:
        y_pool = np.concatenate([y_classic, y_bayes])
        ylim = tuple(np.quantile(y_pool, display_quantiles))

    fig = plt.figure(figsize=(13, 6))
    gs_left = fig.add_gridspec(
        nrows=2, ncols=2, left=0.06, right=0.47, height_ratios=[1, 4],
        width_ratios=[4, 1], hspace=0.06, wspace=0.06,
    )
    gs_right = fig.add_gridspec(
        nrows=2, ncols=2, left=0.57, right=0.98, height_ratios=[1, 4],
        width_ratios=[4, 1], hspace=0.06, wspace=0.06,
    )

    _joint_panel(
        fig, gs_left, x_classic, y_classic, xlim, ylim,
        r"$\log_{10} D_{\mathrm{linear}}$ ($\mu m^2/s$)", r"$\alpha$ (log-log fit)",
        "Classic MSD fit",
    )
    _joint_panel(
        fig, gs_right, x_bayes, y_bayes, xlim, ylim,
        r"$\log_{10} D$ ($\mu m^2/s$, normal-model MAP)", r"$\alpha$ (anomalous-model MAP)",
        "Exact-likelihood Bayes MAP",
    )
    fig.suptitle(
        f"Per-track $\\log_{{10}} D$ vs. $\\alpha$: classic MSD vs. Bayes (n={sub.height} tracks in common, "
        "shared axis limits)",
        fontsize=11,
    )
    return fig


def plot_D_recovery(
    df: pl.DataFrame, D_col: str, true_D_col: str, title: str = "D recovery"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sub = df.select([D_col, true_D_col]).drop_nulls()
    true_D, D_fit = sub[true_D_col].to_numpy(), sub[D_col].to_numpy()
    ax.scatter(true_D, D_fit, s=10, alpha=0.4, color="steelblue")
    lo, hi = true_D.min() * 0.8, true_D.max() * 1.2
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--", label="unity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"true D ($\mu m^2/s^\alpha$)")
    ax.set_ylabel(r"fitted D ($\mu m^2/s^\alpha$)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_alpha_recovery(
    df: pl.DataFrame, alpha_col: str, true_alpha_col: str, title: str = "alpha recovery"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sub = df.select([alpha_col, true_alpha_col]).drop_nulls()
    true_a, a_fit = sub[true_alpha_col].to_numpy(), sub[alpha_col].to_numpy()
    ax.scatter(true_a, a_fit, s=10, alpha=0.4, color="darkorange")
    lo, hi = min(true_a.min(), a_fit.min()) - 0.05, max(true_a.max(), a_fit.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--", label="unity")
    ax.set_xlabel(r"true $\alpha$")
    ax.set_ylabel(r"fitted $\alpha$")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_bias_vs_D_null(
    df: pl.DataFrame, alpha_col: str, true_D_col: str, title: str
) -> plt.Figure:
    """For a null simulation (true alpha=1 for every track, varying true D):
    fitted alpha vs. true D. Flat at alpha=1 means no D-alpha artifact;
    compare directly against `analysis.viz.plot_parameter_recovery_bias`'s
    right panel for the classic MSD estimator on the same kind of data."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sub = df.select([alpha_col, true_D_col]).drop_nulls()
    ax.scatter(sub[true_D_col], sub[alpha_col], s=10, alpha=0.4, color="darkorange")
    medians = (
        sub.group_by(true_D_col).agg(median_alpha=pl.col(alpha_col).median()).sort(true_D_col)
    )
    ax.plot(medians[true_D_col], medians["median_alpha"], color="crimson", marker="o", lw=1.5,
            label="median per D")
    ax.axhline(1.0, color="0.3", lw=1, ls="--", label=r"true $\alpha=1$")
    ax.set_xscale("log")
    ax.set_xlabel(r"true D ($\mu m^2/s$)")
    ax.set_ylabel(r"fitted $\alpha$")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_log_bf_distribution(
    per_track: pl.DataFrame, title: str, log_bf_col: str = "log_bf10"
) -> plt.Figure:
    """Histogram of per-track log BF10 (`bayes_factor.per_track_log_bayes_factor`'s
    output) against the Jeffreys-scale reference lines (>1.1 moderate, >2.3
    strong) -- the diagnostic for "does this population of tracks show any
    individual anisotropy signal," complementary to the ensemble sum
    (`bayes_factor.aggregate_log_bayes_factor`), which answers whether the
    *population as a whole* does."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals = per_track[log_bf_col].to_numpy()
    ax.hist(vals, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="0.2", lw=1, ls="-", label="logBF10=0 (no preference)")
    ax.axvline(1.1, color="darkorange", lw=1, ls="--", label="1.1 (moderate)")
    ax.axvline(2.3, color="crimson", lw=1, ls="--", label="2.3 (strong)")
    ax.set_xlabel("per-track log BF10 (>0 favors anisotropy)")
    ax.set_ylabel("number of tracks")
    ax.set_title(f"{title} (n={len(vals)}, median={np.median(vals):+.3f})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_trajectory_gallery(
    tracks: pl.DataFrame,
    track_ids: list[int],
    panel_labels: list[str],
    title: str,
    ncols: int = 3,
) -> plt.Figure:
    """Small multiples of actual (recentered) per-track (x,y) paths -- the
    direct "does this track's own inferred anisotropy score look plausible
    by eye" check: `panel_labels` (one string per `track_ids`, e.g.
    "logBF10=+1.23") is meant to carry whatever per-track attribute
    (`bayes_factor.per_track_log_bayes_factor`'s `log_bf10`,
    `inference.fit_table_nuts`'s `eps_median`, ...) motivated
    picking that track, so the plotted shape can be checked against the
    number that was computed from it.

    Every panel shares the same physical-distance axis limits (not
    autoscaled per panel) so visual "elongation" is comparable across
    panels, not an artifact of matplotlib rescaling each one differently.
    """
    n = len(track_ids)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows), squeeze=False)

    paths = []
    for pid in track_ids:
        g = tracks.filter(pl.col("track_id") == pid).sort("frame")
        x, y = g["x_um"].to_numpy(), g["y_um"].to_numpy()
        paths.append((x - x[0], y - y[0]))
    max_extent = max(max(np.abs(x).max(), np.abs(y).max()) for x, y in paths) * 1.15

    for i, (ax, pid, label, (x, y)) in enumerate(zip(axes.flat, track_ids, panel_labels, paths)):
        ax.plot(x, y, color="0.4", lw=1, zorder=1)
        ax.scatter(x, y, c=np.arange(len(x)), cmap="viridis", s=25, zorder=2)
        ax.scatter([0], [0], marker="+", color="crimson", s=60, zorder=3)
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        ax.set_aspect("equal")
        ax.set_title(f"track {pid}\n{label}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_spatial_map(
    master: pl.DataFrame,
    value_col: str,
    title: str,
    diverging: bool = True,
    x_col: str = "x_mean_um",
    y_col: str = "y_mean_um",
) -> plt.Figure:
    """Each track's mean field-of-view position, colored by `value_col` --
    checks whether a per-track quantity (e.g. `log_bf10`) clusters spatially
    rather than scattering uniformly, which would point at an
    instrument/acquisition-region effect rather than per-track physics
    (see FINDINGS.md, "Real-data anisotropy check").

    `diverging=True` (the right choice for a signed quantity like log_bf10,
    where 0 is a meaningful "no preference" midpoint) centers a red/blue
    diverging colormap on 0 via `TwoSlopeNorm`; `diverging=False` (for a
    non-negative magnitude like `eps_median`) uses a single-hue sequential
    colormap from its own min to max instead -- never a rainbow either way.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    sub = master.select([x_col, y_col, value_col]).drop_nulls()
    x, y, v = sub[x_col].to_numpy(), sub[y_col].to_numpy(), sub[value_col].to_numpy()

    if diverging:
        vmax = max(abs(v.min()), abs(v.max()), 1e-6)
        norm_ = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = "RdBu_r"
    else:
        norm_ = None
        cmap = "viridis"

    sc = ax.scatter(x, y, c=v, cmap=cmap, norm=norm_, s=28, edgecolor="0.3", linewidth=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(value_col)
    ax.set_xlabel(r"mean $x$ ($\mu m$)")
    ax.set_ylabel(r"mean $y$ ($\mu m$)")
    ax.set_aspect("equal")
    ax.set_title(f"{title} (n={len(v)})")
    fig.tight_layout()
    return fig


def plot_eps_vs_log_bf(master: pl.DataFrame, title: str) -> plt.Figure:
    """Per-track eps posterior median (with its HPDI as a horizontal error
    bar) against log_bf10 -- do the two anisotropy outputs (a per-track
    interval estimate vs. the population-level detector) agree in
    direction? See FINDINGS.md: expect a noisy, weak relationship at
    N=5-10, not a tight one -- eps alone has limited per-track power there,
    log_bf10 is the piece meant to be trusted at the individual-track
    level."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sub = master.select(["eps_median", "eps_lo", "eps_hi", "log_bf10"]).drop_nulls()
    eps, lo, hi, logbf = (sub[c].to_numpy() for c in ["eps_median", "eps_lo", "eps_hi", "log_bf10"])
    xerr = np.vstack([eps - lo, hi - eps])
    ax.errorbar(eps, logbf, xerr=xerr, fmt="o", ms=4, color="steelblue", ecolor="0.7",
                elinewidth=1, capsize=0, alpha=0.7)
    ax.axhline(0.0, color="0.3", lw=1, ls="--")
    ax.axvline(0.0, color="0.3", lw=1, ls="--")
    ax.set_xlabel("eps posterior median (90% HPDI)")
    ax.set_ylabel("log BF10")
    ax.set_title(f"{title} (n={len(eps)})")
    fig.tight_layout()
    return fig


def plot_eps_forest(
    master: pl.DataFrame, title: str, top_n: int = 30, sort_by: str = "log_bf10"
) -> plt.Figure:
    """Caterpillar plot: eps posterior median + 90% HPDI for the `top_n`
    tracks by `sort_by` -- the direct "how honest/wide are these intervals"
    view for the tracks that most drove an ensemble result, complementing
    the single aggregate number with the per-track uncertainty that number
    was built from."""
    sub = master.sort(sort_by, descending=True).head(top_n)
    eps, lo, hi = sub["eps_median"].to_numpy(), sub["eps_lo"].to_numpy(), sub["eps_hi"].to_numpy()
    particles = sub["track_id"].to_numpy()
    y = np.arange(len(sub))[::-1]

    fig, ax = plt.subplots(figsize=(6, 0.28 * len(sub) + 1.5))
    xerr = np.vstack([eps - lo, hi - eps])
    ax.errorbar(eps, y, xerr=xerr, fmt="o", ms=4, color="steelblue", ecolor="0.6",
                elinewidth=1.2, capsize=2)
    ax.axvline(0.0, color="0.3", lw=1, ls="--", label="eps=0 (isotropic)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"track {p}" for p in particles], fontsize=7)
    ax.set_xlabel("eps posterior median (90% HPDI)")
    ax.set_title(f"{title} (top {len(sub)} by {sort_by})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
