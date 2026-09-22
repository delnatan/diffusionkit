"""Diagnostic plots for the exact-likelihood Bayesian (NUTS) pipeline.

Same conventions as `analysis.viz`: pure functions, polars/numpy data in, a
matplotlib Figure out, no shared state.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import polars as pl
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

    If `laplace_mean`/`laplace_cov` are given, overlays a Gaussian
    approximation (e.g. from `inference.fit_batch_svi`'s mean-field guide)
    as a 1-2 sigma ellipse on each 2D panel and a Gaussian curve on each 1D
    panel -- the visual check that a fast approximation is trustworthy
    against the full NUTS posterior it's being compared to. `samples` is a
    flat (n_samples, dim) array; use
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


