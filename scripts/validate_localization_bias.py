"""Confirm the D-alpha correlation seen in per-track MSD fits is a
localization-noise fitting artifact, not real physics -- and check whether
subtracting the *known* localization offset before the log-log fit removes
it.

Simulates pure Brownian tracks (true alpha=1 for every track) spanning the
same D range seen in the real bead data, with constant localization noise
(no D-dependence by construction), then runs them through the exact same
compute_all_tamsd / fit_all_tracks pipeline used on the real data. Any trend
of fitted alpha with true D is necessarily an artifact of the fitting
procedure, since the simulation has no such coupling. The offset used here is
"perfectly known" (the exact sigma_loc used to generate the noise), so this
is a best-case test of the correction; real data only ever has a
finite-sample estimate of it (see `localization_offset_by_track`).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run straight from a clone without installing: put the repo root ahead of
# sys.path so `import diffusionkit` resolves. Harmless once pip-installed.
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl

from diffusionkit.classic import (
    compute_all_tamsd,
    fit_all_tracks,
    localization_offset_by_track,
    simulate_brownian_tracks,
)
from diffusionkit.classic.viz import (
    plot_alpha_correction_comparison,
    plot_parameter_recovery_bias,
)

EXPERIMENT = "validate_localization_bias"
FIG_DIR = REPO_ROOT / "results" / "figures" / EXPERIMENT
TABLE_DIR = REPO_ROOT / "results" / "tables" / EXPERIMENT

DT_S = 0.033
SIGMA_LOC_UM = 0.025   # representative of this dataset's localization precision
TRACK_LENGTH = 200
N_REPLICATES = 60
D_VALUES = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15]  # spans the real per-track D IQR


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    sim = simulate_brownian_tracks(
        D_values_um2_s=D_VALUES,
        n_replicates=N_REPLICATES,
        track_length=TRACK_LENGTH,
        dt_s=DT_S,
        sigma_loc_um=SIGMA_LOC_UM,
        seed=0,
    )
    truth = sim.select(["track_id", "true_D_um2_s"]).unique()
    loc_offset = localization_offset_by_track(sim)  # exact here: sigma_x_um/sigma_y_um are the true sigma_loc

    tamsd = compute_all_tamsd(
        sim.select(["track_id", "frame", "x_um", "y_um", "track_length"]), dt_s=DT_S
    )
    summary = fit_all_tracks(
        tamsd, min_track_length=10, localization_offset=loc_offset
    ).join(truth, on="track_id")

    report = (
        summary.group_by("true_D_um2_s")
        .agg(
            n=pl.len(),
            median_D_fit=pl.col("D_um2_s").median(),
            median_alpha=pl.col("alpha").median(),
            median_alpha_corrected=pl.col("alpha_corrected").median(),
            n_points_dropped=(pl.col("n_points_used_alpha") - pl.col("n_points_used_alpha_corrected")).mean(),
        )
        .sort("true_D_um2_s")
    )
    print(f"Simulated {sim['track_id'].n_unique()} Brownian tracks "
          f"(true alpha=1, sigma_loc={SIGMA_LOC_UM} um, track_length={TRACK_LENGTH})")
    print(report)

    D_fit = summary["D_um2_s"].to_numpy()
    alpha_fit = summary["alpha"].to_numpy()
    alpha_corr = summary["alpha_corrected"].to_numpy()
    r_uncorrected = np.corrcoef(D_fit, alpha_fit)[0, 1]
    valid = np.isfinite(alpha_corr)
    r_corrected = np.corrcoef(D_fit[valid], alpha_corr[valid])[0, 1]
    print(f"\npearson r(D_fit, alpha)           = {r_uncorrected:.3f}  (uncorrected)")
    print(f"pearson r(D_fit, alpha_corrected) = {r_corrected:.3f}  "
          f"(offset-subtracted, known offset; {valid.sum()}/{len(valid)} tracks had a usable corrected fit)")

    summary.write_csv(TABLE_DIR / "simulation_recovery.csv")

    fig1 = plot_parameter_recovery_bias(summary)
    fig1.savefig(FIG_DIR / "simulation_alpha_bias.png", dpi=150)

    fig2 = plot_alpha_correction_comparison(summary)
    fig2.savefig(FIG_DIR / "simulation_alpha_correction.png", dpi=150)

    print(f"\nSaved tables to {TABLE_DIR}, figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
