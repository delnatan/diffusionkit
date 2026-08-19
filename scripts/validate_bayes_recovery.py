"""Ground-truth recovery tests for the exact-likelihood Bayesian pipeline --
the direct counterpart to `validate_localization_bias.py`, run against the
same kind of question but with `bayes.simulate_fbm_tracks` ground truth and
no MSD anywhere in the estimator.

One inference engine (`bayes.inference.fit_all_tracks`) serves both the
informative Bayesian fit and, with `bayes.WEAK_ANOMALOUS_PRIOR`, what used
to be a separate MLE implementation -- used below only where a flat prior is
specifically the point (checks 2 and 3): check 2 as a prior-free test of
whether the *likelihood itself* recovers alpha (an informative prior mildly
pulling toward alpha=1 would make that claim less clean); check 3 to
reproduce the boundary-degeneracy failure mode the informative prior fixes.

Three checks, each simulating from the exact model the likelihood assumes
(so any bias found is about the *estimator*, not a mismatch between
generative and inference models):

  1. Null D-alpha bias: true alpha=1 for every track, true D swept across a
     representative range (same values and track length as
     `validate_localization_bias.py`, for an apples-to-apples comparison
     against the classic pipeline's own null test). Checks whether median
     fitted alpha stays flat across true D, with a flat or an informative
     prior. The classic-comparable pairing is D from the
     *Brownian-constrained* normal model against alpha from the *separate*
     anomalous fit (matching how the classic pipeline defines its own D-vs-
     alpha check -- see `analysis.viz.plot_D_alpha_jointplot`) -- D_alpha
     and alpha *within* the same joint anomalous fit are reported too,
     separately, since that's a different (and not artifact-comparable)
     quantity: jointly-estimated parameters with an expected Fisher-
     information trade-off, not the MSD-fitting bias this check targets.

  2. Genuine alpha recovery: true alpha swept from sub- to super-diffusive
     at fixed D, flat prior -- a check the classic pipeline's own validation
     can't easily run, since MSD-fitting anomalous tracks has no simple
     closed-form null. The exact-likelihood model can, because the same fGn
     covariance generates the ground truth and is what the likelihood
     assumes.

  3. Short-track boundary degeneracy rate: how often does a flat-prior fit
     run D or alpha to the edge of its support on short tracks, and does an
     informative prior fix it.

See FINDINGS.md for results from running these checks.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl

from bayes import (
    AnomalousModelPrior,
    NormalModelPrior,
    WEAK_ANOMALOUS_PRIOR,
    WEAK_NORMAL_PRIOR,
    batched_anomalous_diffusion_model,
    batched_normal_diffusion_model,
    sigma_prior_from_localization,
    fit_all_tracks,
    simulate_fbm_tracks,
    plot_bias_vs_D_null,
    plot_alpha_recovery,
    plot_D_recovery,
)

EXPERIMENT = "validate_bayes_recovery"
FIG_DIR = REPO_ROOT / "results" / "figures" / EXPERIMENT
TABLE_DIR = REPO_ROOT / "results" / "tables" / EXPERIMENT

DT_S = 0.033
SIGMA_LOC_UM = 0.025  # matches validate_localization_bias.py's representative value
D_FLOOR, ALPHA_EPS = 1e-6, 1e-3
PARAM_NAMES = ["D_alpha", "sigma", "alpha"]


def informative_prior(sigma_x_um: np.ndarray, sigma_y_um: np.ndarray) -> AnomalousModelPrior:
    mean, sd = sigma_prior_from_localization(sigma_x_um, sigma_y_um)
    return AnomalousModelPrior(log_sigma_mean=mean, log_sigma_sd=sd)


def weak_prior(sigma_x_um: np.ndarray, sigma_y_um: np.ndarray) -> AnomalousModelPrior:
    return WEAK_ANOMALOUS_PRIOR


def informative_normal_prior(sigma_x_um: np.ndarray, sigma_y_um: np.ndarray) -> NormalModelPrior:
    mean, sd = sigma_prior_from_localization(sigma_x_um, sigma_y_um)
    return NormalModelPrior(log_sigma_mean=mean, log_sigma_sd=sd)


def weak_normal_prior(sigma_x_um: np.ndarray, sigma_y_um: np.ndarray) -> NormalModelPrior:
    return WEAK_NORMAL_PRIOR


def fit_anomalous(sim: pl.DataFrame, prior_fn, min_track_length: int, suffix: str) -> pl.DataFrame:
    fit = fit_all_tracks(
        sim, batched_anomalous_diffusion_model, dt_s=DT_S, prior_fn=prior_fn,
        param_names=PARAM_NAMES, min_track_length=min_track_length,
    )
    rename = {}
    for name in PARAM_NAMES:
        rename[name] = f"{name}_{suffix}"
        rename[f"{name}_stderr"] = f"{name}_stderr_{suffix}"
    return fit.rename(rename)


def fit_normal(sim: pl.DataFrame, prior_fn, min_track_length: int, suffix: str) -> pl.DataFrame:
    fit = fit_all_tracks(
        sim, batched_normal_diffusion_model, dt_s=DT_S, prior_fn=prior_fn,
        param_names=["D", "sigma"], min_track_length=min_track_length,
    )
    return fit.rename({
        "D": f"D_{suffix}", "D_stderr": f"D_stderr_{suffix}",
        "sigma": f"sigma_normal_{suffix}", "sigma_stderr": f"sigma_stderr_normal_{suffix}",
    })


def degenerate_fraction(D: pl.Series, alpha: pl.Series) -> float:
    bad = (D.to_numpy() < D_FLOOR) | (alpha.to_numpy() < ALPHA_EPS) | (alpha.to_numpy() > 2 - ALPHA_EPS)
    return float(bad.mean())


def check_1_null_D_alpha_bias() -> None:
    print("\n" + "=" * 70)
    print("1. Null D-alpha bias (true alpha=1, true D swept) -- vs. classic MSD")
    print("=" * 70)
    D_VALUES = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15]  # identical to validate_localization_bias.py
    TRACK_LENGTH, N_REP = 200, 60

    sim = simulate_fbm_tracks(
        params_um2_s_alpha=[(D, 1.0) for D in D_VALUES],
        n_replicates=N_REP, track_length=TRACK_LENGTH, dt_s=DT_S,
        sigma_loc_um=SIGMA_LOC_UM, seed=100,
    )
    truth = sim.select(["track_id", "true_D_um2_s_alpha"]).unique()

    # The classic-MSD-comparable metric pairs the *Brownian-constrained* D
    # (normal model, alpha pinned to 1) against alpha from the *separate*
    # anomalous fit -- exactly how the classic pipeline defines its own
    # D-vs-alpha check (analysis.viz.plot_D_alpha_jointplot,
    # validate_localization_bias.py). D_alpha and alpha *within* the same
    # joint anomalous fit are a different question (an expected Fisher-
    # information trade-off between two jointly-estimated parameters), not
    # the cross-model artifact metric -- both reported below, separated.
    weak_n = fit_normal(sim, weak_normal_prior, min_track_length=10, suffix="weak")
    bay_n = fit_normal(sim, informative_normal_prior, min_track_length=10, suffix="bayes")
    weak_a = fit_anomalous(sim, weak_prior, min_track_length=10, suffix="weak")
    bay_a = fit_anomalous(sim, informative_prior, min_track_length=10, suffix="bayes")
    combined = (
        weak_n.join(bay_n, on=["track_id", "track_length", "n_disp"])
        .join(weak_a, on=["track_id", "track_length", "n_disp"])
        .join(bay_a, on=["track_id", "track_length", "n_disp"])
        .join(truth, on="track_id")
    )

    report = (
        combined.group_by("true_D_um2_s_alpha")
        .agg(
            n=pl.len(),
            median_alpha_weak=pl.col("alpha_weak").median(),
            median_alpha_bayes=pl.col("alpha_bayes").median(),
        )
        .sort("true_D_um2_s_alpha")
    )
    print(f"Simulated {sim['track_id'].n_unique()} Brownian tracks "
          f"(true alpha=1, sigma_loc={SIGMA_LOC_UM} um, track_length={TRACK_LENGTH})")
    print(report)

    r_weak = np.corrcoef(combined["D_weak"], combined["alpha_weak"])[0, 1]
    r_bayes = np.corrcoef(combined["D_bayes"], combined["alpha_bayes"])[0, 1]
    print(f"\npooled pearson r(D_normal, alpha_anomalous) [classic-comparable]: "
          f"flat-prior={r_weak:.3f}, informative-prior={r_bayes:.3f}  "
          "(compare against the classic MSD pipeline's own result on the same simulation design, "
          "see analysis/ and FINDINGS.md)")
    r_weak_joint = np.corrcoef(combined["D_alpha_weak"], combined["alpha_weak"])[0, 1]
    r_bayes_joint = np.corrcoef(combined["D_alpha_bayes"], combined["alpha_bayes"])[0, 1]
    print(f"pooled pearson r(D_alpha, alpha) [same joint anomalous fit, NOT the artifact metric]: "
          f"flat-prior={r_weak_joint:.3f}, informative-prior={r_bayes_joint:.3f}")
    print("median alpha range across the D sweep: "
          f"flat-prior=[{report['median_alpha_weak'].min():.3f}, {report['median_alpha_weak'].max():.3f}], "
          f"informative-prior=[{report['median_alpha_bayes'].min():.3f}, {report['median_alpha_bayes'].max():.3f}]  "
          "(flat here means no D-dependent bias in fitted alpha; compare against the classic MSD "
          "pipeline's own range from validate_localization_bias.py)")

    combined.write_csv(TABLE_DIR / "bayes_validate_null_D_alpha_bias.csv")

    fig = plot_bias_vs_D_null(
        combined, "alpha_weak", "true_D_um2_s_alpha",
        "Exact-likelihood, flat prior: alpha vs. true D (null test)",
    )
    fig.savefig(FIG_DIR / "bayes_validate_null_bias_weak.png", dpi=150, bbox_inches="tight")

    fig = plot_bias_vs_D_null(
        combined, "alpha_bayes", "true_D_um2_s_alpha",
        "Bayesian MAP (informative prior): alpha vs. true D (null test)",
    )
    fig.savefig(FIG_DIR / "bayes_validate_null_bias_bayes.png", dpi=150, bbox_inches="tight")


def check_2_alpha_recovery() -> None:
    print("\n" + "=" * 70)
    print("2. Genuine alpha recovery (true alpha swept, true D fixed, flat prior)")
    print("=" * 70)
    ALPHA_VALUES = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8]
    TRUE_D, TRACK_LENGTH, N_REP = 0.05, 200, 60

    sim = simulate_fbm_tracks(
        params_um2_s_alpha=[(TRUE_D, a) for a in ALPHA_VALUES],
        n_replicates=N_REP, track_length=TRACK_LENGTH, dt_s=DT_S,
        sigma_loc_um=SIGMA_LOC_UM, seed=200,
    )
    truth = sim.select(["track_id", "true_D_um2_s_alpha", "true_alpha"]).unique()
    fit = fit_all_tracks(
        sim, batched_anomalous_diffusion_model, dt_s=DT_S, prior_fn=weak_prior,
        param_names=PARAM_NAMES, min_track_length=10,
    ).join(truth, on="track_id")

    report = (
        fit.group_by("true_alpha")
        .agg(n=pl.len(), median_alpha_fit=pl.col("alpha").median(), median_D_fit=pl.col("D_alpha").median())
        .sort("true_alpha")
    )
    print(f"Simulated {sim['track_id'].n_unique()} fBm tracks "
          f"(true D={TRUE_D} um^2/s^alpha, sigma_loc={SIGMA_LOC_UM} um, track_length={TRACK_LENGTH})")
    print(report)
    bias = (report["median_alpha_fit"] - report["true_alpha"]).to_numpy()
    print(f"max |median alpha_fit - true alpha| = {np.max(np.abs(bias)):.3f} across "
          f"true alpha in [{min(ALPHA_VALUES)}, {max(ALPHA_VALUES)}]")

    fit.write_csv(TABLE_DIR / "bayes_validate_alpha_recovery.csv")

    fig = plot_alpha_recovery(
        fit, "alpha", "true_alpha",
        "Exact-likelihood, flat prior: alpha recovery across sub-/super-diffusive ground truth",
    )
    fig.savefig(FIG_DIR / "bayes_validate_alpha_recovery.png", dpi=150, bbox_inches="tight")

    fig = plot_D_recovery(
        fit, "D_alpha", "true_D_um2_s_alpha",
        "Exact-likelihood, flat prior: D_alpha recovery (fixed true D, alpha varies)",
    )
    fig.savefig(FIG_DIR / "bayes_validate_D_alpha_recovery.png", dpi=150, bbox_inches="tight")


def check_3_short_track_degeneracy() -> None:
    print("\n" + "=" * 70)
    print("3. Short-track boundary degeneracy: flat prior vs. informative prior")
    print("=" * 70)
    TRACK_LENGTHS = [10, 12, 15]
    TRUE_D, TRUE_ALPHA, N_REP = 0.05, 1.0, 150

    rows = []
    for track_length in TRACK_LENGTHS:
        sim = simulate_fbm_tracks(
            params_um2_s_alpha=[(TRUE_D, TRUE_ALPHA)],
            n_replicates=N_REP, track_length=track_length, dt_s=DT_S,
            sigma_loc_um=SIGMA_LOC_UM, seed=300 + track_length,
        )
        weak = fit_all_tracks(sim, batched_anomalous_diffusion_model, dt_s=DT_S, prior_fn=weak_prior,
                               param_names=PARAM_NAMES, min_track_length=track_length)
        bay = fit_all_tracks(sim, batched_anomalous_diffusion_model, dt_s=DT_S, prior_fn=informative_prior,
                              param_names=PARAM_NAMES, min_track_length=track_length)
        frac_weak = degenerate_fraction(weak["D_alpha"], weak["alpha"])
        frac_bayes = degenerate_fraction(bay["D_alpha"], bay["alpha"])
        rows.append((track_length, N_REP, frac_weak, frac_bayes))
        print(f"  track_length={track_length:3d} (n_disp={track_length - 1:2d}): "
              f"boundary-degenerate rate flat-prior={100 * frac_weak:5.1f}%  "
              f"informative-prior={100 * frac_bayes:5.1f}%  (n={N_REP} simulated tracks)")

    report = pl.DataFrame(
        rows, schema=["track_length", "n_replicates", "degenerate_frac_weak", "degenerate_frac_bayes"],
        orient="row",
    )
    report.write_csv(TABLE_DIR / "bayes_validate_short_track_degeneracy.csv")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    check_1_null_D_alpha_bias()
    check_2_alpha_recovery()
    check_3_short_track_degeneracy()
    print(f"\nDone in {time.time() - t0:.1f}s. Saved tables to {TABLE_DIR}, figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
