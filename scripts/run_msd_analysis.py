"""Standard MSD-based diffusion analysis for the mobile-bead SPT dataset.

Pipeline: load -> `analysis.fit_population` (TAMSD -> ensemble fit ->
per-track fits) -> save tables/figures. See WORKFLOW.md for this call in
the wider low-data/bulk/anisotropy workflow picture.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run straight from a clone without installing: put the repo root ahead of
# sys.path so `import diffusionkit` resolves. Harmless once pip-installed.
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from diffusionkit.classic import (
    AcquisitionParams,
    assert_contiguous_tracks,
    fit_population,
    load_tracks,
)
from diffusionkit.classic.viz import (
    plot_K_jointplot,
    plot_D_vs_track_length,
    plot_ensemble_fit,
    plot_localization_diagnostic,
    plot_parameter_distributions,
    plot_tamsd_curves,
)

DATA_CSV = REPO_ROOT / "mobile_beads_1to200.csv"
WORKFLOW = "classic"
FIG_DIR = REPO_ROOT / "results" / "figures" / WORKFLOW
TABLE_DIR = REPO_ROOT / "results" / "tables" / WORKFLOW

PARAMS = AcquisitionParams(pixel_size_um=0.1043, dt_s=0.033)
MIN_TRACK_LENGTH = 10       # shortest track admitted into per-track fitting
FRAC_POINTS = 0.25          # fraction of each track's lags used in fits
MIN_FIT_POINTS = 3
MAX_FIT_POINTS = 10         # absolute cap; long-lag MSD is noisy/correlated (see fitting.n_fit_points)
MIN_TRACKS_FOR_ENSEMBLE = 10  # drop ensemble lags supported by fewer tracks


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(DATA_CSV, PARAMS)
    assert_contiguous_tracks(tracks)
    n_particles = tracks["track_id"].n_unique()
    print(f"Loaded {tracks.height} localizations, {n_particles} tracks")

    result = fit_population(
        tracks,
        PARAMS.dt_s,
        min_track_length=MIN_TRACK_LENGTH,
        frac_points=FRAC_POINTS,
        min_points=MIN_FIT_POINTS,
        max_points=MAX_FIT_POINTS,
        min_tracks_for_ensemble=MIN_TRACKS_FOR_ENSEMBLE,
    )
    tamsd, ensemble = result.tamsd, result.ensemble
    normal_fit, anomalous_fit = result.ensemble_normal_fit, result.ensemble_anomalous_fit
    summary = result.per_track
    print(f"Ensemble MSD curve: {ensemble.height} lags "
          f"(down to n_tracks={MIN_TRACKS_FOR_ENSEMBLE})")

    tau = ensemble["tau_s"].to_numpy()
    npts = result.ensemble_n_points_used

    print("\n=== Ensemble-averaged fit ===")
    print(f"  fit range: first {npts} lags (tau up to {tau[npts-1]:.3f} s)")
    print(f"  Normal diffusion:    D = {normal_fit.D_um2_s:.4g} +/- "
          f"{normal_fit.D_stderr_um2_s:.2g} um^2/s   "
          f"intercept = {normal_fit.intercept_um2:.4g} +/- "
          f"{normal_fit.intercept_stderr_um2:.2g} um^2  "
          f"(R^2={normal_fit.r_squared:.4f})")
    print(f"  Anomalous diffusion: alpha = {anomalous_fit.alpha:.4f} +/- "
          f"{anomalous_fit.alpha_stderr:.3g}   "
          f"K = {anomalous_fit.K_um2_s_alpha:.4g} um^2/s^alpha  "
          f"(R^2={anomalous_fit.r_squared:.4f})")
    print(f"  Mean expected localization offset (from sigma_x/sigma_y): "
          f"{result.mean_localization_offset_um2:.4g} um^2  (fitted intercept: "
          f"{normal_fit.intercept_um2:.4g} um^2)")

    # --- per-track fits ---
    D = summary["D_um2_s"]
    alpha = summary["alpha"]
    print(f"\n=== Per-track fits (n={summary.height} tracks, "
          f"track_length >= {MIN_TRACK_LENGTH}) ===")
    print(f"  D:     median={D.median():.4g}  "
          f"IQR=[{D.quantile(0.25):.4g}, {D.quantile(0.75):.4g}] um^2/s")
    print(f"  alpha: median={alpha.median():.4f}  "
          f"IQR=[{alpha.quantile(0.25):.4f}, {alpha.quantile(0.75):.4f}]")

    n_neg_D = summary["D_negative"].sum()
    n_neg_b = summary["intercept_negative"].sum()
    n_floor = summary["at_min_points"].sum()
    print(f"  quality flags: D<0 in {n_neg_D}/{summary.height} tracks "
          f"({100*n_neg_D/summary.height:.1f}%); "
          f"intercept<0 (unphysical) in {n_neg_b}/{summary.height} "
          f"({100*n_neg_b/summary.height:.1f}%); "
          f"at min_points floor in {n_floor}/{summary.height} "
          f"({100*n_floor/summary.height:.1f}%)")
    D_arr, a_arr = D.to_numpy(), alpha.to_numpy()
    r_uncorrected = np.corrcoef(D_arr, a_arr)[0, 1]
    print(f"  pearson r(D, alpha) across tracks = {r_uncorrected:.3f}"
          "  (see validate_localization_bias.py and FINDINGS.md for whether this reflects "
          "a localization-noise fitting artifact vs. real physics)")

    alpha_corr = summary["alpha_corrected"]
    n_failed = alpha_corr.is_nan().sum()
    valid = ~alpha_corr.is_nan().to_numpy()
    r_corrected = np.corrcoef(D_arr[valid], alpha_corr.to_numpy()[valid])[0, 1]
    print(f"  offset-subtracted alpha: median={alpha_corr.median():.4f}  "
          f"IQR=[{alpha_corr.quantile(0.25):.4f}, {alpha_corr.quantile(0.75):.4f}]  "
          f"({n_failed}/{summary.height} tracks had too few positive points left to fit)")
    print(f"  pearson r(D, alpha_corrected) = {r_corrected:.3f}  "
          "(compare to r(D, alpha) above -- correction is only as good as the per-track offset estimate)")

    alpha_nlgls_corr = summary["alpha_nlgls_corrected"]
    n_failed_nlgls = alpha_nlgls_corr.is_nan().sum()
    valid_nlgls = ~alpha_nlgls_corr.is_nan().to_numpy()
    r_nlgls_corrected = np.corrcoef(D_arr[valid_nlgls], alpha_nlgls_corr.to_numpy()[valid_nlgls])[0, 1]
    print(f"  nonlinear-GLS offset-corrected alpha: median={alpha_nlgls_corr.median():.4f}  "
          f"IQR=[{alpha_nlgls_corr.quantile(0.25):.4f}, {alpha_nlgls_corr.quantile(0.75):.4f}]  "
          f"({n_failed_nlgls}/{summary.height} tracks flagged singular)")
    print(f"  pearson r(D, alpha_nlgls_corrected) = {r_nlgls_corrected:.3f}  "
          "(see FINDINGS.md: this is the recommended classic alpha estimate -- covariance-weighted, "
          "offset-corrected, no log-transform delta-method approximation)")

    summary.write_csv(TABLE_DIR / "per_track_msd_fits.csv")
    ensemble.write_csv(TABLE_DIR / "ensemble_msd.csv")
    tamsd.write_parquet(TABLE_DIR / "per_track_tamsd.parquet")
    print(f"\nSaved tables to {TABLE_DIR}")

    fig1 = plot_tamsd_curves(tamsd, ensemble)
    fig1.savefig(FIG_DIR / "tamsd_curves.png", dpi=150)

    fig2 = plot_ensemble_fit(ensemble, normal_fit, anomalous_fit)
    fig2.savefig(FIG_DIR / "ensemble_fit.png", dpi=150)

    fig3 = plot_parameter_distributions(summary)
    fig3.savefig(FIG_DIR / "parameter_distributions.png", dpi=150)

    fig4 = plot_localization_diagnostic(summary)
    fig4.savefig(FIG_DIR / "localization_diagnostic.png", dpi=150)

    fig5 = plot_D_vs_track_length(summary)
    fig5.savefig(FIG_DIR / "D_vs_track_length.png", dpi=150)

    fig6 = plot_K_jointplot(summary)
    fig6.savefig(FIG_DIR / "K_jointplot.png", dpi=150, bbox_inches="tight")

    fig7 = plot_K_jointplot(summary, alpha_col="alpha_nlgls_corrected")
    fig7.savefig(FIG_DIR / "K_jointplot_nlgls_corrected.png", dpi=150, bbox_inches="tight")

    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
