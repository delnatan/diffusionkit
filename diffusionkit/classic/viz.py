"""Diagnostic plots for the MSD-based diffusion analysis.

Each function takes tidy polars DataFrames (+ optional fit results) and an
optional matplotlib Axes to draw into, and returns the Figure. No function
holds state or mutates its inputs, so any plot can be regenerated from
intermediate results alone, independent of pipeline order.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from .fitting import AnomalousDiffusionFit, NormalDiffusionFit


def plot_tamsd_curves(
    tamsd: pl.DataFrame,
    ensemble: pl.DataFrame,
    n_tracks_to_show: int = 60,
    seed: int = 0,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Log-log spaghetti plot of individual per-track TAMSD curves, with the
    ensemble average overlaid."""
    fig = ax.figure if ax is not None else plt.figure(figsize=(6, 5))
    ax = ax or fig.gca()

    rng = np.random.default_rng(seed)
    particles = tamsd["track_id"].unique().to_numpy()
    chosen = rng.choice(particles, size=min(n_tracks_to_show, len(particles)), replace=False)

    for pid in chosen:
        sub = tamsd.filter(pl.col("track_id") == pid)
        ax.plot(sub["tau_s"], sub["msd_um2"], color="0.75", lw=0.6, alpha=0.7, zorder=1)

    ax.plot(
        ensemble["tau_s"], ensemble["msd_um2"], color="crimson", lw=2, zorder=2,
        label="ensemble average",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$ (s)")
    ax.set_ylabel(r"MSD ($\mu m^2$)")
    ax.set_title(f"Per-track TAMSD ({len(chosen)} of {len(particles)} tracks shown)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def plot_ensemble_fit(
    ensemble: pl.DataFrame,
    normal_fit: NormalDiffusionFit,
    anomalous_fit: AnomalousDiffusionFit,
) -> plt.Figure:
    """Two-panel plot: linear-scale MSD with the normal-diffusion fit, and
    log-log MSD with the power-law (anomalous) fit."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    tau = ensemble["tau_s"].to_numpy()
    msd = ensemble["msd_um2"].to_numpy()
    sem = ensemble["msd_sem"].to_numpy()

    ax1.errorbar(tau, msd, yerr=sem, fmt="o", ms=3, color="0.3", ecolor="0.75",
                 elinewidth=1, capsize=0, label="ensemble MSD")
    n = normal_fit.n_points
    tau_fit = tau[:n]
    pred = 4 * normal_fit.D_um2_s * tau_fit + normal_fit.intercept_um2
    ax1.plot(tau_fit, pred, color="crimson", lw=2,
              label=f"fit: D={normal_fit.D_um2_s:.4g}"
                    r" $\mu m^2/s$" f"\n(n={n} pts, "
                    r"$R^2$" f"={normal_fit.r_squared:.3f})")
    ax1.set_xlabel(r"$\tau$ (s)")
    ax1.set_ylabel(r"MSD ($\mu m^2$)")
    ax1.set_title("Normal-diffusion fit (linear space)")
    ax1.legend(frameon=False, fontsize=9)

    ax2.errorbar(tau, msd, yerr=sem, fmt="o", ms=3, color="0.3", ecolor="0.75",
                 elinewidth=1, capsize=0)
    n2 = anomalous_fit.n_points
    tau_fit2 = tau[:n2]
    pred2 = 4 * anomalous_fit.D_alpha_um2_s_alpha * tau_fit2 ** anomalous_fit.alpha
    ax2.plot(tau_fit2, pred2, color="steelblue", lw=2,
             label=r"fit: $\alpha$=" f"{anomalous_fit.alpha:.3f}"
                   f"\n(n={n2} pts, "
                   r"$R^2$" f"={anomalous_fit.r_squared:.3f})")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\tau$ (s)")
    ax2.set_ylabel(r"MSD ($\mu m^2$)")
    ax2.set_title("Anomalous-diffusion fit (log-log space)")
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    return fig


def plot_parameter_distributions(summary: pl.DataFrame) -> plt.Figure:
    """Histograms of per-track D and alpha estimates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    D = summary["D_um2_s"].to_numpy()
    D = D[D > 0]
    ax1.hist(np.log10(D), bins=40, color="steelblue", edgecolor="white")
    ax1.axvline(np.log10(np.median(D)), color="crimson", lw=1.5,
                label=f"median={np.median(D):.4g}" r" $\mu m^2/s$")
    ax1.set_xlabel(r"$\log_{10} D$  ($\mu m^2/s$)")
    ax1.set_ylabel("number of tracks")
    ax1.set_title("Per-track D (normal-diffusion fit)")
    ax1.legend(frameon=False, fontsize=9)

    alpha = summary["alpha"].to_numpy()
    bins = np.linspace(
        min(-0.3, np.nanmin(alpha)), max(1.6, np.nanmax(alpha)), 40
    )
    ax2.hist(alpha, bins=bins, color="darkorange", edgecolor="white", alpha=0.6,
             label=f"uncorrected (median={np.nanmedian(alpha):.3f})")
    if "alpha_corrected" in summary.columns:
        alpha_c = summary["alpha_corrected"].to_numpy()
        ax2.hist(alpha_c, bins=bins, color="steelblue", edgecolor="white", alpha=0.6,
                 label=f"offset-subtracted (median={np.nanmedian(alpha_c):.3f})")
    ax2.axvline(1.0, color="0.3", lw=1, ls="--", label=r"$\alpha=1$ (Brownian)")
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel("number of tracks")
    ax2.set_title(r"Per-track $\alpha$: uncorrected vs. offset-subtracted")
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    return fig


def plot_D_alpha_jointplot(
    summary: pl.DataFrame, alpha_col: str = "alpha"
) -> plt.Figure:
    """Joint distribution of log10(D) and alpha across real-data tracks.

    D here is `D_um2_s` (units um^2/s) from `fit_normal_diffusion`: the
    *linear* MSD = 4*D*tau + b fit, over the same short-lag window
    (n_points_used) as the alpha fit -- i.e. the short-time apparent D under
    a forced normal-diffusion assumption, deliberately paired against the
    independently-fit power-law exponent alpha from the *same* lag range.
    This is NOT `D_alpha_um2_s_alpha` (units um^2/s^alpha, from
    `fit_anomalous_diffusion`) -- the two only coincide numerically when
    alpha=1, and plotting D_alpha against alpha would be circular (alpha
    appears in D_alpha's own units) rather than a genuine two-parameter
    comparison. Axis labels spell this out to avoid the D vs D_alpha mixup
    that's an easy trap in the anomalous-diffusion literature.

    A 2D KDE sits *beneath* the scatter as a density guide, with marginal
    histograms on each axis. This is the diagnostic view for checking
    whether fitted D and alpha are spuriously correlated across tracks --
    real per-track fits only, no ground truth to compare against here; see
    `plot_alpha_correction_comparison` for the matched-simulation check of
    whether such a correlation is a localization-noise fitting artifact
    rather than physics.
    """
    df = (
        summary.filter(pl.col("D_um2_s") > 0)
        .select(
            pl.col("D_um2_s").log10().alias("log10_D"),
            pl.col(alpha_col).alias("alpha"),
        )
        .drop_nulls()
    )
    n_dropped = summary.height - df.height
    # polars.to_pandas() requires pyarrow, which isn't a project dependency;
    # seaborn needs a pandas/array-like frame, so build one directly instead.
    pdf = pd.DataFrame(
        {"log10_D": df["log10_D"].to_numpy(), "alpha": df["alpha"].to_numpy()}
    )

    g = sns.JointGrid(data=pdf, x="log10_D", y="alpha", height=6.5, ratio=4)

    sns.kdeplot(
        data=pdf, x="log10_D", y="alpha", ax=g.ax_joint,
        fill=True, cmap="Blues", alpha=0.6, thresh=0.05, levels=12, zorder=0,
    )
    sns.scatterplot(
        data=pdf, x="log10_D", y="alpha", ax=g.ax_joint,
        s=18, alpha=0.6, color="0.15", edgecolor="none", zorder=1,
    )
    g.ax_joint.axhline(1.0, color="crimson", lw=1, ls="--", zorder=2,
                        label=r"$\alpha=1$ (Brownian)")

    g.ax_marg_x.hist(pdf["log10_D"], bins=30, color="steelblue", edgecolor="white")
    g.ax_marg_y.hist(pdf["alpha"], bins=30, color="steelblue", edgecolor="white",
                      orientation="horizontal")

    r = np.corrcoef(pdf["log10_D"], pdf["alpha"])[0, 1]
    g.ax_joint.set_xlabel(r"$\log_{10} D_{\mathrm{linear}}$  ($\mu m^2/s$, short-time normal-diffusion fit)")
    ylabel = r"$\alpha$ (log-log fit)" if alpha_col == "alpha" else f"{alpha_col} (log-log fit)"
    g.ax_joint.set_ylabel(ylabel)
    g.ax_joint.legend(frameon=False, loc="upper left")
    g.ax_marg_x.set_title(
        f"Per-track D vs. alpha (r={r:.2f}, n={df.height}"
        + (f", {n_dropped} dropped: D<=0" if n_dropped else "") + ")\n"
        r"D is $D_{\mathrm{linear}}$ (MSD=$4D\tau$+b), NOT $D_\alpha$ (MSD=$4D_\alpha\tau^\alpha$, "
        r"units $\mu m^2/s^\alpha$) -- same short-lag window per track for both fits",
        fontsize=9.5, loc="left",
    )
    return g.figure


def plot_localization_diagnostic(summary_with_offset: pl.DataFrame) -> plt.Figure:
    """Fitted MSD intercept vs. the expected offset from measured sigma_x/sigma_y.

    Points on the unity line mean the fitted static-localization offset is
    consistent with the localization precision reported by the MLE
    localizer. Systematic deviation flags either dynamic (motion-blur)
    error or a fit-range artifact.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    x = summary_with_offset["offset_um2"].to_numpy()
    y = summary_with_offset["intercept_um2"].to_numpy()
    ax.scatter(x, y, s=10, alpha=0.5, color="steelblue")
    lo, hi = 0, max(x.max(), y.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--", label="unity")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"expected offset from $\sigma_x, \sigma_y$  ($\mu m^2$)")
    ax.set_ylabel(r"fitted MSD intercept  ($\mu m^2$)")
    ax.set_title("Localization-error self-consistency")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_parameter_recovery_bias(summary_with_truth: pl.DataFrame) -> plt.Figure:
    """Fitted D and alpha vs. known ground-truth D, for simulated tracks.

    Left panel: fitted D vs true D (should sit on the unity line). Right
    panel: fitted alpha vs true D, with a dashed line at alpha=1 (every
    simulated track is exactly Brownian) -- any upward trend here is a
    localization-noise fitting artifact, not a real D-alpha coupling, since
    the simulation has none by construction.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    true_D = summary_with_truth["true_D_um2_s"].to_numpy()
    D_fit = summary_with_truth["D_um2_s"].to_numpy()
    alpha_fit = summary_with_truth["alpha"].to_numpy()

    ax1.scatter(true_D, D_fit, s=10, alpha=0.4, color="steelblue")
    lo, hi = true_D.min() * 0.8, true_D.max() * 1.2
    ax1.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--", label="unity")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"true D ($\mu m^2/s$)")
    ax1.set_ylabel(r"fitted D ($\mu m^2/s$)")
    ax1.set_title("D recovery")
    ax1.legend(frameon=False)

    ax2.scatter(true_D, alpha_fit, s=10, alpha=0.4, color="darkorange")
    medians = (
        summary_with_truth.group_by("true_D_um2_s")
        .agg(median_alpha=pl.col("alpha").median())
        .sort("true_D_um2_s")
    )
    ax2.plot(medians["true_D_um2_s"], medians["median_alpha"], color="crimson",
             marker="o", lw=1.5, label="median per D")
    ax2.axhline(1.0, color="0.3", lw=1, ls="--", label=r"true $\alpha=1$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"true D ($\mu m^2/s$)")
    ax2.set_ylabel(r"fitted $\alpha$")
    ax2.set_title(r"$\alpha$ bias vs. true D (localization-noise artifact)")
    ax2.legend(frameon=False)

    fig.tight_layout()
    return fig


def plot_alpha_correction_comparison(summary_with_truth: pl.DataFrame) -> plt.Figure:
    """Median fitted alpha vs. true D, uncorrected vs. offset-subtracted.

    Requires `alpha` and `alpha_corrected` columns (from
    `fit_all_tracks(..., localization_offset=...)`) plus `true_D_um2_s`
    (from a `simulate_brownian_tracks` run). If the offset correction works,
    the corrected curve should sit flat at alpha=1 where the uncorrected one
    trends upward with D.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    medians = (
        summary_with_truth.group_by("true_D_um2_s")
        .agg(
            median_alpha=pl.col("alpha").median(),
            median_alpha_corrected=pl.col("alpha_corrected").median(),
        )
        .sort("true_D_um2_s")
    )

    ax.plot(medians["true_D_um2_s"], medians["median_alpha"], color="darkorange",
             marker="o", lw=1.5, label="uncorrected")
    ax.plot(medians["true_D_um2_s"], medians["median_alpha_corrected"], color="steelblue",
             marker="s", lw=1.5, label="offset-subtracted")
    ax.axhline(1.0, color="0.3", lw=1, ls="--", label=r"true $\alpha=1$")
    ax.set_xscale("log")
    ax.set_xlabel(r"true D ($\mu m^2/s$)")
    ax.set_ylabel(r"median fitted $\alpha$")
    ax.set_title("Does subtracting the known localization offset fix the bias?")
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig


def plot_D_vs_track_length(summary: pl.DataFrame) -> plt.Figure:
    """Check for length-dependent bias in per-track D estimates."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(summary["track_length"], summary["D_um2_s"], s=10, alpha=0.5, color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("track length (frames)")
    ax.set_ylabel(r"D ($\mu m^2/s$)")
    ax.set_title("D vs. track length")
    fig.tight_layout()
    return fig
