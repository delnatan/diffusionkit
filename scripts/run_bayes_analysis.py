"""Exact-likelihood Bayesian diffusion analysis for the mobile-bead SPT
dataset -- the modern counterpart to `run_msd_analysis.py`.

No MSD curve is computed or fit anywhere in this script. D, alpha, and the
localization precision sigma are estimated directly from the per-frame
displacement sequence via the exact Gaussian likelihood in `bayes.likelihood`
(Michalet & Berglund 2012's framework, generalized to anomalous diffusion
via fractional Gaussian noise), through numpyro models in `bayes.model`.

Production inference is `bayes.fit_population(..., model="both")`: batched
exact MAP (L-BFGS-B on numpyro's own unconstrained potential, per
length-group sub-batches, `engine="map"` default) for both the
Brownian-constrained normal model (D, sigma) and the anomalous model
(D_alpha, sigma, alpha) at once -- see FINDINGS.md ("Inference-engine choice
for production") for why MAP over SVI. D and D_alpha are reported via their
log-space Laplace fit with an asymmetric back-transformed interval, not a
symmetric mean +/- stderr in linear units (FINDINGS.md, "D should be
reported in log-space, with an asymmetric interval") -- alpha keeps a
symmetric physical-space interval, which checks there found adequate. D
(normal model) and alpha (anomalous model) are the primary per-particle
diffusive-behavior metrics; D_alpha is kept as a secondary/diagnostic
quantity (FINDINGS.md: D_alpha's posterior degrades much faster than D's on
short tracks). See WORKFLOW.md for the low-data (`bayes.fit_track`) and bulk
(`bayes.fit_population`) API this script uses.

Pipeline: load -> per-track batched MAP (normal + anomalous models) -> full
NUTS posteriors on 3 representative tracks -> join against the classic MSD
results already on disk (results/tables/classic/per_track_msd_fits.csv, from
run_msd_analysis.py) -> save tables/figures.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpyro

numpyro.set_host_device_count(
    4
)  # must precede any jax device use -- for real parallel NUTS chains

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl
from analysis import AcquisitionParams, assert_contiguous_tracks, load_tracks
from bayes import (
    WEAK_ANOMALOUS_PRIOR,
    fit_population,
    fit_track,
    plot_D_alpha_joint,
    plot_classic_vs_bayes_joint,
    plot_estimator_scatter,
    plot_mcmc_trace,
    plot_posterior_corner,
    samples_dict_to_arrays,
)

DATA_CSV = REPO_ROOT / "mobile_beads_1to200.csv"
WORKFLOW = "bayes"
CLASSIC_TABLE = (
    REPO_ROOT / "results" / "tables" / "classic" / "per_track_msd_fits.csv"
)
FIG_DIR = REPO_ROOT / "results" / "figures" / WORKFLOW
TABLE_DIR = REPO_ROOT / "results" / "tables" / WORKFLOW

PARAMS = AcquisitionParams(pixel_size_um=0.1043, dt_s=0.033)
MIN_TRACK_LENGTH = (
    10  # same floor as the classic pipeline, for a fair comparison
)
MAX_BATCH_SIZE = (
    20  # fit_batch_map's per-call track cap -- see its docstring / FINDINGS.md
)

MCMC_NUM_WARMUP = 500
MCMC_NUM_SAMPLES = 1000
MCMC_NUM_CHAINS = 4
D_FLOOR, ALPHA_EPS = 1e-6, 1e-3


def degenerate_fraction(D: pl.Series, alpha: pl.Series) -> float:
    bad = (
        (D.to_numpy() < D_FLOOR)
        | (alpha.to_numpy() < ALPHA_EPS)
        | (alpha.to_numpy() > 2 - ALPHA_EPS)
    )
    return float(bad.mean())


def pick_representative_tracks(summary: pl.DataFrame, n: int = 3) -> list[int]:
    """Shortest, median-length, and longest eligible tracks -- chosen by
    track_length percentile so the illustrative full-NUTS examples span the
    same short/noisy-to-long/precise range the batch fits cover."""
    lengths = summary.select(["track_id", "track_length"]).sort("track_length")
    qs = [0.1, 0.5, 0.95][:n]
    chosen = [
        int(lengths["track_id"][int(round(q * (lengths.height - 1)))])
        for q in qs
    ]
    return sorted(set(chosen))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(DATA_CSV, PARAMS)
    assert_contiguous_tracks(tracks)
    print(
        f"Loaded {tracks.height} localizations, {tracks['track_id'].n_unique()} tracks"
    )

    # --- batch Bayesian fits: normal + anomalous models joined in one call,
    # no MSD, batched exact MAP per FINDINGS.md's production decision ---
    t0 = time.time()
    summary = fit_population(
        tracks,
        PARAMS.dt_s,
        model="both",
        min_track_length=MIN_TRACK_LENGTH,
        max_batch_size=MAX_BATCH_SIZE,
    )
    t1 = time.time()
    print(
        f"Batched Bayesian fit (normal + anomalous models): {summary.height} tracks in {t1 - t0:.1f}s"
    )

    D, alpha = summary["D_median_um2_s"], summary["alpha"]
    print(
        f"\n=== Per-track Bayesian fits (n={summary.height}, "
        f"track_length >= {MIN_TRACK_LENGTH}) ==="
    )
    print(
        f"  normal model (primary D):    log10(D) median={summary['log10_D'].median():.4g}  "
        f"D median={D.median():.4g}  IQR=[{D.quantile(0.25):.4g}, {D.quantile(0.75):.4g}] um^2/s"
    )
    print(
        f"  anomalous model (primary alpha): alpha median={alpha.median():.4f}  "
        f"IQR=[{alpha.quantile(0.25):.4f}, {alpha.quantile(0.75):.4f}]  "
        f"(D_alpha median={summary['D_alpha_median_um2_s_alpha'].median():.4g} um^2/s^a, secondary)"
    )

    n_not_converged = int(
        (~summary["normal_converged"]).sum()
        + (~summary["anomalous_converged"]).sum()
    )
    print(
        f"  L-BFGS-B non-convergence flags: {n_not_converged} out of "
        f"{2 * summary.height} fits (per sub-batch of <= {MAX_BATCH_SIZE} tracks, not per track)"
    )

    # Two distinct D-vs-alpha comparisons, not to be conflated (see
    # model.py's docstring and FINDINGS.md): pairing the *Brownian-
    # constrained* D (normal model) against alpha from the *separate*
    # anomalous fit is the metric comparable against the classic MSD
    # pipeline's own D-vs-alpha check. D_alpha and alpha *within* the same
    # joint anomalous fit answer a different question -- they're jointly
    # estimated from one likelihood surface with an expected Fisher-
    # information trade-off, not the cross-model artifact metric.
    r_bayes = np.corrcoef(D.to_numpy(), alpha.to_numpy())[0, 1]
    print(
        f"  pearson r(D_normal, alpha_anomalous) = {r_bayes:.3f}  (cross-model, comparable to the "
        "classic MSD pipeline's D-vs-alpha check -- see validate_bayes_recovery.py and FINDINGS.md)"
    )
    r_joint = np.corrcoef(
        summary["D_alpha_median_um2_s_alpha"].to_numpy(), alpha.to_numpy()
    )[0, 1]
    print(
        f"  pearson r(D_alpha, alpha) [same joint anomalous fit] = {r_joint:.3f}  "
        "(within-fit parameter trade-off, not the cross-model artifact metric above)"
    )

    summary.write_csv(TABLE_DIR / "per_track_bayes_fits.csv")

    # --- the one MLE-shaped comparison kept: does the informative prior fix
    # the short-track boundary-degeneracy a flat-prior fit shows? ---
    weak_fits = fit_population(
        tracks,
        PARAMS.dt_s,
        model="anomalous",
        prior=WEAK_ANOMALOUS_PRIOR,
        min_track_length=MIN_TRACK_LENGTH,
        max_batch_size=MAX_BATCH_SIZE,
    )
    frac_weak = degenerate_fraction(
        weak_fits["D_alpha_median_um2_s_alpha"], weak_fits["alpha"]
    )
    frac_bayes = degenerate_fraction(
        summary["D_alpha_median_um2_s_alpha"], alpha
    )
    print(
        f"\n  boundary-degenerate anomalous fits (D_alpha<{D_FLOOR:g} or alpha within "
        f"{ALPHA_EPS:g} of the [0,2] edge): flat-prior={int(frac_weak * summary.height)}/{summary.height} "
        f"({100 * frac_weak:.1f}%), informative-prior={int(frac_bayes * summary.height)}/{summary.height} "
        f"({100 * frac_bayes:.1f}%) -- almost all on the shortest, most weakly-constrained tracks; see FINDINGS.md"
    )

    # --- comparison against the classic MSD pipeline's saved results ---
    if CLASSIC_TABLE.exists():
        classic = pl.read_csv(CLASSIC_TABLE).select(
            "track_id",
            pl.col("D_um2_s").alias("D_classic_um2_s"),
            pl.col("alpha").alias("alpha_classic"),
        )
        joined = summary.join(classic, on="track_id", how="inner")
        joined.write_csv(TABLE_DIR / "bayes_vs_classic_comparison.csv")

        r_D = np.corrcoef(
            joined["D_classic_um2_s"], joined["D_median_um2_s"]
        )[0, 1]
        r_alpha = np.corrcoef(joined["alpha_classic"], joined["alpha"])[0, 1]
        print(
            f"\n=== vs. classic MSD fit (n={joined.height} tracks in common) ==="
        )
        print(
            f"  agreement: r(D_classic, D_bayes)={r_D:.3f}, "
            f"r(alpha_classic, alpha_bayes)={r_alpha:.3f}"
        )

        fig = plot_estimator_scatter(
            joined,
            "D_classic_um2_s",
            "D_median_um2_s",
            r"classic MSD fit $D$ ($\mu m^2/s$)",
            r"exact-likelihood Bayes MAP $D$ ($\mu m^2/s$)",
            "D: classic MSD vs. exact-likelihood Bayes",
            log=True,
        )
        fig.savefig(
            FIG_DIR / "bayes_D_vs_classic.png", dpi=150, bbox_inches="tight"
        )

        fig = plot_estimator_scatter(
            joined,
            "alpha_classic",
            "alpha",
            r"classic MSD fit $\alpha$",
            r"exact-likelihood Bayes MAP $\alpha$",
            r"$\alpha$: classic MSD vs. exact-likelihood Bayes",
        )
        fig.savefig(
            FIG_DIR / "bayes_alpha_vs_classic.png",
            dpi=150,
            bbox_inches="tight",
        )

        fig = plot_classic_vs_bayes_joint(joined)
        fig.savefig(
            FIG_DIR / "classic_vs_bayes_D_alpha_jointplot.png",
            dpi=150,
            bbox_inches="tight",
        )
    else:
        print(
            f"\n(skipping classic-pipeline comparison -- {CLASSIC_TABLE} not found; "
            "run scripts/run_msd_analysis.py first for a full comparison)"
        )

    fig = plot_D_alpha_joint(
        summary,
        "D_median_um2_s",
        "alpha",
        r"$D_{\mathrm{normal\ model}}$ ($\mu m^2/s$)",
        r"$\alpha_{\mathrm{anomalous\ model}}$",
        "Per-track D vs. alpha, exact-likelihood Bayesian MAP (no MSD) -- classic-MSD-comparable pairing",
    )
    fig.savefig(
        FIG_DIR / "bayes_D_alpha_jointplot.png", dpi=150, bbox_inches="tight"
    )

    fig = plot_D_alpha_joint(
        summary,
        "D_alpha_median_um2_s_alpha",
        "alpha",
        r"$D_{\alpha}$ (same anomalous fit, $\mu m^2/s^\alpha$)",
        r"$\alpha$ (same anomalous fit)",
        "Per-track D_alpha vs. alpha WITHIN the same joint fit -- NOT the classic-comparable artifact metric",
    )
    fig.savefig(
        FIG_DIR / "bayes_D_alpha_jointplot_within_fit.png",
        dpi=150,
        bbox_inches="tight",
    )

    # --- full NUTS posteriors on a few representative tracks, via
    # fit_track(method="nuts")/fit_track(method="map") -- illustrative
    # single-track diagnostic use, `.raw` is the escape hatch into the
    # underlying (samples, mcmc)/MAPFit these plots need ---
    print(
        f"\n=== Full NUTS posteriors on representative tracks "
        f"({MCMC_NUM_CHAINS} chains x {MCMC_NUM_WARMUP}+{MCMC_NUM_SAMPLES} steps) ==="
    )
    example_particles = pick_representative_tracks(summary, n=3)
    for pid in example_particles:
        grp = tracks.filter(pl.col("track_id") == pid).sort("frame")
        n_frames = grp.height
        row = summary.filter(pl.col("track_id") == pid).row(0, named=True)

        t0 = time.time()
        nuts_fit = fit_track(
            grp,
            PARAMS.dt_s,
            model="anomalous",
            method="nuts",
            num_warmup=MCMC_NUM_WARMUP,
            num_samples=MCMC_NUM_SAMPLES,
            num_chains=MCMC_NUM_CHAINS,
            seed=pid,
        )
        dt = time.time() - t0
        samples, _mcmc = nuts_fit.raw
        flat, trace = samples_dict_to_arrays(
            samples, ["D_alpha", "sigma", "alpha"]
        )
        print(
            f"  track {pid} (track_length={n_frames}): {dt:.1f}s, "
            f"posterior median D_alpha={nuts_fit.params['D_alpha']:.4g}, alpha={nuts_fit.params['alpha']:.3f}  "
            f"[batch fit: D_alpha={row['D_alpha_median_um2_s_alpha']:.4g} "
            f"({row['D_alpha_lo_um2_s_alpha']:.4g}, {row['D_alpha_hi_um2_s_alpha']:.4g}), alpha={row['alpha']:.3f}]"
        )

        # Single-track MAP (not the batched one above) purely for this
        # plot's linear-space Gaussian overlay -- its `.raw.cov` is the
        # exact delta-method pushforward, cleaner here than re-deriving a
        # linear-space stderr from the batch fit's log-space columns; the
        # batch fit is what's actually reported (see FINDINGS.md for why
        # that's log-space with an asymmetric interval, not this ellipse).
        map_fit = fit_track(grp, PARAMS.dt_s, model="anomalous", method="map")
        single_fit = map_fit.raw
        fit_mean = np.array(
            [single_fit.params[k] for k in ["D_alpha", "sigma", "alpha"]]
        )
        fit_cov = None
        if single_fit.cov is not None:
            fit_cov = np.array(
                [
                    [
                        single_fit.cov[a][b]
                        for b in ["D_alpha", "sigma", "alpha"]
                    ]
                    for a in ["D_alpha", "sigma", "alpha"]
                ]
            )

        fig = plot_posterior_corner(
            flat,
            ["D_alpha", "sigma_loc", "alpha"],
            ["um2/s^a", "um", ""],
            laplace_mean=fit_mean,
            laplace_cov=fit_cov,
        )
        fig.savefig(
            FIG_DIR / f"bayes_posterior_corner_particle{pid}.png", dpi=150
        )

        fig = plot_mcmc_trace(
            trace, ["D_alpha", "sigma_loc", "alpha"], ["um2/s^a", "um", ""]
        )
        fig.savefig(FIG_DIR / f"bayes_trace_particle{pid}.png", dpi=150)

    print(f"\nSaved tables to {TABLE_DIR}, figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
