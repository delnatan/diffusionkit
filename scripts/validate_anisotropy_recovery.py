"""Ground-truth recovery / calibration checks for `bayes.anisotropy.anisotropic_diffusion_model`
-- the direct counterpart to `validate_bayes_recovery.py`, targeting the
N=5-10 information-starved regime the anisotropy-detection plan is for.

Full NUTS posteriors only (`sample_posterior` on the batched model), not
MAP/Laplace: the eps=0 boundary induces the same kind of ridge in the
posterior geometry that FINDINGS.md documents for alpha near its own
boundary at short track lengths (as eps -> 0, psi becomes unidentifiable),
where a Gaussian/delta-method approximation is known to be unreliable.

Two checks, both simulating from `bayes.anisotropy.simulate_anisotropic_tracks` (the
exact model `anisotropic_diffusion_model` assumes, so any miscalibration
found is about the estimator/prior, not a generative/inference mismatch):

  1. Null (false-positive) characterization at eps=0. NOTE on methodology:
     eps=0 sits at the boundary of a continuous, strictly-positive-support
     posterior, so "does the 95% interval exclude the point 0" is a
     degenerate test -- a continuous posterior's lower HPDI edge is a.s.
     never exactly 0, so that criterion flags ~100% of tracks regardless of
     calibration (confirmed by hand before writing this script). The
     meaningful, operational version is a *threshold* test: what fraction of
     truly-isotropic tracks would a practical anisotropy-flagging rule (e.g.
     posterior median eps > 0.3, or P(eps > 0.3 | data) > 0.5) incorrectly
     flag. Run at two sigma_loc regimes (low and high relative to step size)
     to directly test the Gemini draft's motivating false-positive scenario
     (large sigma inducing a spurious 180-degree turning-angle peak), and
     under both the default shrinkage prior and the flat (WEAK) prior to
     show *why* the shrinkage prior in `priors.AnisotropicModelPrior` is
     load-bearing, not cosmetic.

  2. eps recovery / interval calibration away from the boundary (true eps in
     {0.2, 0.4, 0.6, 0.8}, where the "does the X% interval contain the true
     value X% of the time" question is well-posed) plus the psi-ridge check
     (orientation should be poorly constrained at low eps and progressively
     better constrained as eps grows), at track lengths spanning the whole
     N=5-10 target range.

Checks 1-2 (a point/interval estimate of eps) turn out to have limited
per-track power at N=5-10 -- see FINDINGS.md. `bayes.bayes_factor` reframes
the question as model comparison ("is this more anisotropic than free
diffusion at this track length") instead of parameter estimation, which is
well-posed even when there isn't enough data to pin down eps's value.
Checks 3-4 validate that:

  3. Per-track log BF10 null calibration: at true eps=0, does the log BF10
     distribution stay centered at/below 0 (no systematic false-positive
     inflation) across track lengths and sigma regimes, the well-posed
     version of check 1 (no boundary-exclusion degeneracy here, since BF10
     is a single real number with no boundary of its own).

  4. Ensemble aggregation power: summing log BF10 across M independent
     tracks that share the same true (eps, psi) should accumulate real
     evidence for truly anisotropic populations while staying flat (not
     inflating) for isotropic or only-mildly-anisotropic ones as M grows --
     the concrete test of whether per-track "inconclusive" individually
     still adds up to decisive evidence at the population level, which is
     the practically useful regime at N=5-10 (see also the original Gemini
     draft's own sec. 5.3 "ensemble aggregation" idea, valid here because
     every track's BF is computed under the same fixed H0/H1 pair).

See FINDINGS.md for results from running these checks.
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

import numpyro

numpyro.set_host_device_count(4)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run straight from a clone without installing: put the repo root ahead of
# sys.path so `import diffusionkit` resolves. Harmless once pip-installed.
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np
import polars as pl
from numpyro.diagnostics import hpdi

from diffusionkit.bayes import (
    anisotropic_displacement_covariance,
    displacement_covariance,
    sample_posterior,
)
from diffusionkit.bayes.anisotropy import (
    AnisotropicModelPrior,
    WEAK_ANISOTROPIC_PRIOR,
    batched_anisotropic_diffusion_model,
    batched_log_bayes_factor_anisotropy,
    simulate_anisotropic_tracks,
)
from diffusionkit.bayes.viz import plot_estimator_scatter

EXPERIMENT = "validate_anisotropy_recovery"
FIG_DIR = REPO_ROOT / "results" / "figures" / EXPERIMENT
TABLE_DIR = REPO_ROOT / "results" / "tables" / EXPERIMENT

DT_S = 0.033
D_MEAN = 0.05  # um^2/s
FLAG_THRESHOLD = 0.3  # posterior-median eps above this counts as "flagged anisotropic"
NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 400, 800, 4


def _reduction_sanity_check() -> None:
    """anisotropic_displacement_covariance at eps=0 must exactly reduce to
    the isotropic displacement_covariance (any psi) -- the algebraic claim
    `model.py`'s docstring makes about H0 being nested in H1. A silent
    mismatch here would invalidate every other check in this script."""
    n_disp, sigma2 = 4, 0.02**2
    iso = np.asarray(displacement_covariance(n_disp, D_MEAN, DT_S, 1.0, sigma2))
    for psi in [0.0, 0.4, 1.2, 3.0]:
        full = np.asarray(
            anisotropic_displacement_covariance(n_disp, D_MEAN, 0.0, psi, DT_S, sigma2)
        )
        dxdx, dydy, dxdy = full[0::2, 0::2], full[1::2, 1::2], full[0::2, 1::2]
        assert np.allclose(dxdx, iso) and np.allclose(dydy, iso) and np.allclose(dxdy, 0.0), (
            f"eps=0 reduction failed at psi={psi}"
        )
    print("Sanity check passed: anisotropic_displacement_covariance(eps=0) == "
          "displacement_covariance(alpha=1) exactly (any psi).\n")


def _stack_displacements(sim: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(dx, dy), each shape (n_tracks, n_disp), sorted by track id --
    shared by every check below that needs raw displacement arrays (NUTS
    fitting or the Monte Carlo Bayes factor alike)."""
    particles = sim["track_id"].unique().sort().to_list()
    dx = np.stack([np.diff(sim.filter(pl.col("track_id") == p).sort("frame")["x_um"].to_numpy())
                    for p in particles])
    dy = np.stack([np.diff(sim.filter(pl.col("track_id") == p).sort("frame")["y_um"].to_numpy())
                    for p in particles])
    return dx, dy


def _fit_batch(sim: pl.DataFrame, prior: AnisotropicModelPrior, track_length: int, seed: int):
    dx, dy = _stack_displacements(sim)
    n_tracks, n_disp = dx.shape[0], track_length - 1
    samples, mcmc = sample_posterior(
        batched_anisotropic_diffusion_model,
        (jnp.asarray(dx), jnp.asarray(dy), DT_S, n_disp, prior, n_tracks),
        num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, num_chains=NUM_CHAINS, seed=seed,
    )
    eps = samples["eps"].reshape(-1, n_tracks)
    psi = samples["psi"].reshape(-1, n_tracks)
    return eps, psi


def check_1_null_false_positive_rate() -> None:
    print("=" * 70)
    print("1. Null false-positive characterization (true eps=0)")
    print("=" * 70)
    TRACK_LENGTHS = [5, 7, 10]
    SIGMA_REGIMES = {"low_sigma": 0.01, "high_sigma": 0.05}  # um; 0.05 ~ step-size scale at D=0.05
    N_REP = 60

    rows = []
    for track_length in TRACK_LENGTHS:
        for regime, sigma_loc in SIGMA_REGIMES.items():
            sim = simulate_anisotropic_tracks(
                params=[(D_MEAN, 0.0, 0.0)], n_replicates=N_REP, track_length=track_length,
                dt_s=DT_S, sigma_loc_um=sigma_loc, seed=1000 + track_length,
            )
            for prior_label, prior in [
                ("shrinkage_Beta(1,3)", AnisotropicModelPrior()),
                ("flat_WEAK", WEAK_ANISOTROPIC_PRIOR),
            ]:
                eps, _ = _fit_batch(sim, prior, track_length, seed=0)
                median_eps = np.median(eps, axis=0)
                flagged = float(np.mean(median_eps > FLAG_THRESHOLD))
                hi90 = np.median(hpdi(eps, 0.90, axis=0)[1])
                rows.append((track_length, regime, prior_label, N_REP, float(np.mean(median_eps)),
                             flagged, hi90))
                print(f"  track_length={track_length:2d} {regime:10s} {prior_label:20s}: "
                      f"mean(median eps)={np.mean(median_eps):.3f}  "
                      f"flagged (median eps>{FLAG_THRESHOLD})={100*flagged:5.1f}%  "
                      f"median(90% upper bound)={hi90:.3f}")

    report = pl.DataFrame(
        rows,
        schema=["track_length", "sigma_regime", "prior", "n_replicates", "mean_median_eps",
                "flagged_fraction", "median_hpdi90_upper"],
        orient="row",
    )
    report.write_csv(TABLE_DIR / "null_false_positive_rate.csv")
    print()


def check_2_eps_recovery_and_psi_ridge() -> None:
    print("=" * 70)
    print("2. eps recovery / interval calibration + psi-ridge check")
    print("=" * 70)
    EPS_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8]
    TRACK_LENGTHS = [5, 10]
    TRUE_PSI, N_REP, SIGMA_LOC = 0.9, 100, 0.02

    all_rows = []
    coverage_rows = []
    for track_length in TRACK_LENGTHS:
        eps_medians, eps_truth_all, psi_iqr_all, psi_truth_eps_all = [], [], [], []
        coverage_hits = {level: [] for level in (0.5, 0.8, 0.95)}
        for true_eps in EPS_VALUES:
            sim = simulate_anisotropic_tracks(
                params=[(D_MEAN, true_eps, TRUE_PSI)], n_replicates=N_REP,
                track_length=track_length, dt_s=DT_S, sigma_loc_um=SIGMA_LOC,
                seed=2000 + track_length + int(100 * true_eps),
            )
            eps, psi = _fit_batch(sim, AnisotropicModelPrior(), track_length, seed=1)
            median_eps = np.median(eps, axis=0)
            eps_medians.append(median_eps)
            eps_truth_all.append(np.full_like(median_eps, true_eps))

            psi_iqr = np.percentile(psi, 75, axis=0) - np.percentile(psi, 25, axis=0)
            psi_iqr_all.append(psi_iqr)
            psi_truth_eps_all.append(np.full_like(psi_iqr, true_eps))

            for level in coverage_hits:
                lo, hi = hpdi(eps, level, axis=0)
                coverage_hits[level].append((lo <= true_eps) & (true_eps <= hi))

            print(f"  track_length={track_length:2d} true_eps={true_eps:.1f}: "
                  f"median(fitted eps)={np.median(median_eps):.3f}  "
                  f"median(psi IQR, rad)={np.median(psi_iqr):.2f}")

        eps_medians = np.concatenate(eps_medians)
        eps_truth_all = np.concatenate(eps_truth_all)
        for p, t in zip(eps_medians, eps_truth_all):
            all_rows.append((track_length, float(t), float(p)))

        for level, hits in coverage_hits.items():
            observed = float(np.mean(np.concatenate(hits)))
            coverage_rows.append((track_length, level, observed))
            print(f"  track_length={track_length:2d} nominal={100*level:.0f}% HPDI "
                  f"-> observed coverage={100*observed:.1f}%")
        print()

    recovery = pl.DataFrame(all_rows, schema=["track_length", "true_eps", "fitted_eps_median"], orient="row")
    recovery.write_csv(TABLE_DIR / "eps_recovery.csv")
    coverage = pl.DataFrame(coverage_rows, schema=["track_length", "nominal_level", "observed_coverage"], orient="row")
    coverage.write_csv(TABLE_DIR / "eps_interval_coverage.csv")

    for track_length in TRACK_LENGTHS:
        sub = recovery.filter(pl.col("track_length") == track_length)
        fig = plot_estimator_scatter(
            sub, "true_eps", "fitted_eps_median", "true eps", "posterior median eps",
            f"eps recovery, track_length={track_length}", unity=True,
        )
        fig.savefig(FIG_DIR / f"eps_recovery_tl{track_length}.png", dpi=150, bbox_inches="tight")


N_MC_BF = 20000  # prior-predictive Monte Carlo draws per marginal-likelihood estimate


def check_3_bayes_factor_null_calibration() -> None:
    print("=" * 70)
    print("3. Per-track log BF10 null calibration (true eps=0)")
    print("=" * 70)
    TRACK_LENGTHS = [5, 7, 10]
    SIGMA_REGIMES = {"low_sigma": 0.01, "high_sigma": 0.05}
    N_REP = 200
    prior = AnisotropicModelPrior()

    rows = []
    for track_length in TRACK_LENGTHS:
        n_disp = track_length - 1
        for regime, sigma_loc in SIGMA_REGIMES.items():
            sim = simulate_anisotropic_tracks(
                params=[(D_MEAN, 0.0, 0.0)], n_replicates=N_REP, track_length=track_length,
                dt_s=DT_S, sigma_loc_um=sigma_loc, seed=6000 + track_length,
            )
            dx, dy = _stack_displacements(sim)
            logbf = np.asarray(batched_log_bayes_factor_anisotropy(
                jnp.asarray(dx), jnp.asarray(dy), DT_S, n_disp, prior, n_mc=N_MC_BF, seed=0
            ))
            frac_moderate = float(np.mean(logbf > 1.1))
            rows.append((track_length, regime, N_REP, float(np.median(logbf)), float(np.mean(logbf)),
                         frac_moderate))
            print(f"  track_length={track_length:2d} {regime:10s}: median logBF10={np.median(logbf):+.3f}  "
                  f"mean logBF10={np.mean(logbf):+.3f}  frac(logBF10>1.1 'moderate')={100*frac_moderate:5.1f}%")

    report = pl.DataFrame(
        rows, schema=["track_length", "sigma_regime", "n_replicates", "median_logbf10", "mean_logbf10",
                       "frac_moderate_or_above"],
        orient="row",
    )
    report.write_csv(TABLE_DIR / "bayes_factor_null_calibration.csv")
    print()


def check_4_ensemble_aggregation_power() -> None:
    print("=" * 70)
    print("4. Ensemble aggregation power (sum of log BF10 across M tracks)")
    print("=" * 70)
    TRACK_LENGTH, N_DISP = 5, 4
    EPS_VALUES = [0.0, 0.3, 0.6, 0.8]
    M_VALUES = [10, 30, 100]
    N_REP, N_BOOTSTRAP = 500, 200
    TRUE_PSI = 0.9
    prior = AnisotropicModelPrior()
    rng = np.random.default_rng(0)

    rows = []
    for true_eps in EPS_VALUES:
        sim = simulate_anisotropic_tracks(
            params=[(D_MEAN, true_eps, TRUE_PSI)], n_replicates=N_REP, track_length=TRACK_LENGTH,
            dt_s=DT_S, sigma_loc_um=0.02, seed=7000 + int(100 * true_eps),
        )
        dx, dy = _stack_displacements(sim)
        logbf = np.asarray(batched_log_bayes_factor_anisotropy(
            jnp.asarray(dx), jnp.asarray(dy), DT_S, N_DISP, prior, n_mc=N_MC_BF, seed=0
        ))
        line = f"  true_eps={true_eps:.1f}:"
        for M in M_VALUES:
            sums = np.array([
                logbf[rng.choice(len(logbf), size=M, replace=False)].sum() for _ in range(N_BOOTSTRAP)
            ])
            rows.append((true_eps, M, float(np.median(sums)), float(np.percentile(sums, 10)),
                         float(np.percentile(sums, 90))))
            line += f"  M={M:3d} median={np.median(sums):+6.2f} (p10={np.percentile(sums,10):+6.2f}, p90={np.percentile(sums,90):+6.2f})"
        print(line)

    report = pl.DataFrame(
        rows, schema=["true_eps", "M", "median_ensemble_logbf10", "p10_ensemble_logbf10", "p90_ensemble_logbf10"],
        orient="row",
    )
    report.write_csv(TABLE_DIR / "bayes_factor_ensemble_aggregation.csv")
    print()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _reduction_sanity_check()
    check_1_null_false_positive_rate()
    check_2_eps_recovery_and_psi_ridge()
    check_3_bayes_factor_null_calibration()
    check_4_ensemble_aggregation_power()
    print(f"Done in {time.time() - t0:.1f}s. Saved tables to {TABLE_DIR}, figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
