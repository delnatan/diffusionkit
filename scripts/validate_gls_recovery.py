"""Does GLS actually beat OLS-plus-truncation for D and alpha?

`fitting.fit_normal_diffusion_gls`/`fit_anomalous_diffusion_gls` replace
`n_fit_points`'s ad-hoc lag truncation with a covariance-weighted fit over
(essentially) every available lag (see README's "Classic, done properly"
section for why). That's a claim, not an assumption -- this script checks
it the same way `validate_localization_bias.py` checks the offset
correction: simulate pure Brownian tracks with known ground truth (D=truth,
alpha=1 for every track), run them through the exact production pipeline,
and compare OLS-truncated vs GLS recovery of D and alpha directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl

from diffusionkit.classic import (
    compute_all_tamsd,
    fit_all_tracks,
    localization_offset_by_track,
    simulate_brownian_tracks,
)
from diffusionkit.classic.viz import plot_ols_vs_gls_recovery

EXPERIMENT = "validate_gls_recovery"
FIG_DIR = REPO_ROOT / "results" / "figures" / EXPERIMENT
TABLE_DIR = REPO_ROOT / "results" / "tables" / EXPERIMENT

DT_S = 0.033
SIGMA_LOC_UM = 0.025
TRACK_LENGTH = 60  # shorter than validate_localization_bias.py's 200: GLS
                    # covers the whole curve, no truncation, so track length
                    # alone already stresses the tail-correlation case that
                    # matters here without needing as long a track
N_REPLICATES = 80
D_VALUES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.25]  # low-D (noisy,
    # high localization-noise-to-motion ratio) through high-D (clean) --
    # exactly where OLS-truncation's information loss should matter most vs
    # least


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
    loc_offset = localization_offset_by_track(sim)  # exact: sigma_x/y are the true sigma_loc

    tamsd = compute_all_tamsd(
        sim.select(["track_id", "frame", "x_um", "y_um", "track_length"]), dt_s=DT_S
    )
    summary = fit_all_tracks(
        tamsd, min_track_length=10, localization_offset=loc_offset
    ).join(truth, on="track_id")

    print(f"Simulated {sim['track_id'].n_unique()} Brownian tracks "
          f"(true alpha=1, sigma_loc={SIGMA_LOC_UM} um, track_length={TRACK_LENGTH})")
    print(f"gls_singular: {summary['gls_singular'].sum()}/{summary.height}   "
          f"alpha_gls_singular: {summary['alpha_gls_singular'].sum()}/{summary.height}")

    report = (
        summary.group_by("true_D_um2_s")
        .agg(
            n=pl.len(),
            n_pts_ols=pl.col("n_points_used").first(),
            n_pts_gls=pl.col("n_points_used_gls").median(),
            rmse_D_ols=((pl.col("D_um2_s") - pl.col("true_D_um2_s")) ** 2).mean().sqrt(),
            rmse_D_gls=((pl.col("D_gls_um2_s") - pl.col("true_D_um2_s")) ** 2).mean().sqrt(),
            bias_alpha_ols=(pl.col("alpha") - 1.0).mean(),
            bias_alpha_gls=(pl.col("alpha_gls") - 1.0).mean(),
            std_alpha_ols=pl.col("alpha").std(),
            std_alpha_gls=pl.col("alpha_gls").std(),
        )
        .sort("true_D_um2_s")
    )
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(report)

    summary.write_csv(TABLE_DIR / "simulation_recovery.csv")
    report.write_csv(TABLE_DIR / "recovery_summary_by_D.csv")

    fig = plot_ols_vs_gls_recovery(summary)
    fig.savefig(FIG_DIR / "ols_vs_gls_recovery.png", dpi=150)

    print(f"\nSaved tables to {TABLE_DIR}, figure to {FIG_DIR}")


if __name__ == "__main__":
    main()
