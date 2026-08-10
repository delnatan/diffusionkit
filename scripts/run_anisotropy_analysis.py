"""Anisotropy-detection Bayes factor for the real mobile-bead SPT dataset,
restricted to its short (track_length 5-10) tracks -- the regime
`bayes.anisotropic_diffusion_model`/`bayes.bayes_factor` targets.

These beads are freely diffusing (no known structure/channel to confine
them), so this run doubles as a real-data negative control: FINDINGS.md's
simulated null-calibration check (checks 3-4 of
`validate_anisotropy_recovery.py`) predicts individual-track log BF10 should
stay centered at/below 0 with ~0% of tracks reaching even "moderate"
evidence. This script checks whether that holds on data the method was
never fit to, and additionally calibrates the *ensemble* sum against its own
null distribution (simulated at this exact track-length composition, not
just compared to the generic Jeffreys-scale buckets, which say nothing about
how much a *specific* M=185-track sample of a true null can fluctuate by
chance) -- see FINDINGS.md ("Real-data anisotropy check") for the result and
why the generic buckets alone would have been misleading here.

Also fits the per-track `eps`/`psi` posterior (NUTS, `sample_posterior_table`)
alongside the Bayes factor and joins both onto one `per_track_master.csv`
with each quantity kept under its own explicit name (`log_bf10` vs.
`eps_median`/`eps_lo`/`eps_hi` vs. `psi_median_rad`) -- see FINDINGS.md for
why a per-track `eps` point/interval has limited power on its own at this N
(it's included here as a secondary, honestly-wide per-track descriptive
column, not as a second detector); `log_bf10` remains the only quantity this
script's verdicts are based on. The master table also carries each track's
mean field-of-view position, so every quantity can be mapped back onto the
real trajectories and inspected visually (`plot_trajectory_gallery`,
`plot_spatial_map`) rather than trusted as a bare number.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np
import polars as pl

from analysis import AcquisitionParams, assert_contiguous_tracks, load_tracks
from bayes import (
    AnisotropicModelPrior,
    aggregate_log_bayes_factor,
    batched_anisotropic_diffusion_model,
    batched_log_bayes_factor_anisotropy,
    per_track_log_bayes_factor,
    plot_eps_forest,
    plot_eps_vs_log_bf,
    plot_log_bf_distribution,
    plot_spatial_map,
    plot_trajectory_gallery,
    sample_posterior_table,
    simulate_anisotropic_tracks,
)

DATA_CSV = REPO_ROOT / "mobile_beads_1to200.csv"
WORKFLOW = "anisotropy"
FIG_DIR = REPO_ROOT / "results" / "figures" / WORKFLOW
TABLE_DIR = REPO_ROOT / "results" / "tables" / WORKFLOW

PARAMS = AcquisitionParams(pixel_size_um=0.1043, dt_s=0.033)
MIN_TRACK_LENGTH, MAX_TRACK_LENGTH = 5, 10  # the N=5-10 regime this method targets
N_MC = 20000
N_NULL = 200  # matched-composition null-calibration replicates
EPS_PARAM_NAMES = ["D_mean", "eps", "psi", "D_par", "D_perp"]
N_GALLERY = 9  # tracks shown per trajectory-gallery panel


def _fixed_prior(x_std_um: np.ndarray, y_std_um: np.ndarray) -> AnisotropicModelPrior:
    """`sample_posterior_table`'s `prior_fn` slot: the anisotropic model
    uses one shared prior across tracks (see `bayes_factor.py`'s docstring
    on why a genuinely per-track prior isn't supported), so this ignores
    its arguments rather than deriving a per-track prior the way
    `sigma_prior_from_localization`-based `prior_fn`s elsewhere do."""
    return AnisotropicModelPrior()


def null_calibration(
    composition: dict[int, int], dt_s: float, prior: AnisotropicModelPrior,
    n_null: int, n_mc: int, seed: int = 0,
) -> np.ndarray:
    """Empirical null distribution of the ensemble sum, simulated at the
    *exact* track-length composition of the real short-track sample (not
    just a fixed M) -- D_mean/sigma_loc are taken from the prior's own
    central values (not fit from the real data) so this stays a generic,
    reusable calibration rather than one tuned to a single dataset's
    numbers. `n_null` replicate isotropic datasets, each with the same
    number of tracks per track_length as `composition`; returns the
    ensemble sum (across all tracks in the replicate, all lengths pooled)
    for each replicate.
    """
    D_mean = float(np.exp(prior.log_D_mean))
    sigma_loc = float(np.exp(prior.log_sigma_mean))
    rng = np.random.default_rng(seed)

    null_sums = np.empty(n_null)
    for rep in range(n_null):
        total = 0.0
        for track_length, n_tracks in composition.items():
            n_disp = track_length - 1
            sim = simulate_anisotropic_tracks(
                params=[(D_mean, 0.0, 0.0)], n_replicates=n_tracks, track_length=track_length,
                dt_s=dt_s, sigma_loc_um=sigma_loc, seed=int(rng.integers(0, 1_000_000)),
            )
            particles = sim["particle"].unique().sort().to_list()
            dx = np.stack([np.diff(sim.filter(pl.col("particle") == p).sort("frame")["x_um"].to_numpy())
                            for p in particles])
            dy = np.stack([np.diff(sim.filter(pl.col("particle") == p).sort("frame")["y_um"].to_numpy())
                            for p in particles])
            logbf = np.asarray(batched_log_bayes_factor_anisotropy(
                jnp.asarray(dx), jnp.asarray(dy), dt_s, n_disp, prior, n_mc=n_mc, seed=0
            ))
            total += float(logbf.sum())
        null_sums[rep] = total
    return null_sums


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(DATA_CSV, PARAMS)
    assert_contiguous_tracks(tracks)
    short_tracks = tracks.filter(
        (tracks["track_length"] >= MIN_TRACK_LENGTH) & (tracks["track_length"] <= MAX_TRACK_LENGTH)
    )
    print(f"Loaded {tracks['particle'].n_unique()} tracks total; "
          f"{short_tracks['particle'].n_unique()} have track_length in "
          f"[{MIN_TRACK_LENGTH}, {MAX_TRACK_LENGTH}]")

    prior = AnisotropicModelPrior()
    t0 = time.time()
    per_track = per_track_log_bayes_factor(
        short_tracks, PARAMS.dt_s, prior, min_track_length=MIN_TRACK_LENGTH, n_mc=N_MC, seed=0,
    )
    print(f"Computed log BF10 for {per_track.height} tracks in {time.time() - t0:.1f}s")
    per_track.write_csv(TABLE_DIR / "per_track_log_bf.csv")

    logbf = per_track["log_bf10"].to_numpy()
    frac_moderate = float(np.mean(logbf > 1.1))
    frac_strong = float(np.mean(logbf > 2.3))
    print(f"\n=== Per-track log BF10 (n={len(logbf)}) ===")
    print(f"  median={np.median(logbf):+.4f}  mean={np.mean(logbf):+.4f}  "
          f"IQR=[{np.percentile(logbf,25):+.4f}, {np.percentile(logbf,75):+.4f}]")
    print(f"  fraction reaching 'moderate' evidence (>1.1): {100*frac_moderate:.1f}%")
    print(f"  fraction reaching 'strong' evidence   (>2.3): {100*frac_strong:.1f}%")

    fig = plot_log_bf_distribution(
        per_track, "Mobile beads, track_length 5-10: per-track log BF10 (anisotropy)"
    )
    fig.savefig(FIG_DIR / "per_track_log_bf_distribution.png", dpi=150, bbox_inches="tight")

    # Ensemble-level verdict: (a) the whole short-track population as one
    # group, and (b) broken out by track_length, to see whether the small
    # per-track information gain from n_disp=4 to n_disp=9 (FINDINGS.md's
    # sampling-noise-floor numbers) is visible in the real data too.
    labeled = per_track.with_columns(all=pl.lit("all_short_tracks"))
    ensemble_all = aggregate_log_bayes_factor(labeled, "all")
    ensemble_by_length = aggregate_log_bayes_factor(per_track, "track_length")
    observed_sum = float(ensemble_all["sum_log_bf10"][0])

    print(f"\n=== Ensemble log BF10 (sum across tracks) ===")
    print(ensemble_all)
    print(ensemble_by_length)
    ensemble_all.write_csv(TABLE_DIR / "ensemble_log_bf_all.csv")
    ensemble_by_length.write_csv(TABLE_DIR / "ensemble_log_bf_by_track_length.csv")

    # --- per-track eps/psi posterior (NUTS) + master table + plots ---
    # Secondary, descriptive per-track output -- log_bf10 above remains the
    # only quantity the verdict is based on (see module docstring / FINDINGS.md).
    t0 = time.time()
    eps_posterior = sample_posterior_table(
        short_tracks, batched_anisotropic_diffusion_model, PARAMS.dt_s, _fixed_prior,
        param_names=EPS_PARAM_NAMES, min_track_length=MIN_TRACK_LENGTH, hpdi_prob=0.9,
    )
    # eps_median/eps_lo/eps_hi (dimensionless) are already unambiguous;
    # explicitly unit-suffix everything else so this table's columns follow
    # the same _um2_s/_rad convention as the rest of the project (see
    # README's "Results tables" column reference) rather than the bare
    # `{name}_median` sample_posterior_table returns generically.
    eps_posterior = eps_posterior.rename({
        "D_mean_median": "D_mean_median_um2_s", "D_mean_lo": "D_mean_lo_um2_s", "D_mean_hi": "D_mean_hi_um2_s",
        "D_par_median": "D_par_median_um2_s", "D_par_lo": "D_par_lo_um2_s", "D_par_hi": "D_par_hi_um2_s",
        "D_perp_median": "D_perp_median_um2_s", "D_perp_lo": "D_perp_lo_um2_s", "D_perp_hi": "D_perp_hi_um2_s",
        "psi_median": "psi_median_rad", "psi_lo": "psi_lo_rad", "psi_hi": "psi_hi_rad",
    })
    print(f"\nFit eps/psi posterior for {eps_posterior.height} tracks in {time.time() - t0:.1f}s")
    eps_posterior.write_csv(TABLE_DIR / "per_track_eps_posterior.csv")

    geometry = (
        short_tracks.group_by("particle")
        .agg(x_mean_um=pl.col("x_um").mean(), y_mean_um=pl.col("y_um").mean())
    )
    # Explicit, non-colliding names for every quantity: log_bf10 is the
    # Bayes-factor detector; eps_*/psi_*/D_*_median are the (weaker, at this
    # N) posterior-interval estimate from the *same* anisotropic model --
    # never conflate the two when reading this table (see FINDINGS.md).
    master = (
        per_track.join(eps_posterior, on=["particle", "track_length", "n_disp"])
        .join(geometry, on="particle")
        .sort("particle")
    )
    master.write_csv(TABLE_DIR / "per_track_master.csv")
    print(f"Saved per-track master table ({master.height} tracks, {len(master.columns)} columns) "
          f"to {TABLE_DIR / 'per_track_master.csv'}")

    # Trajectory gallery: the highest-log_bf10 tracks (the ones driving the
    # ensemble result) vs. a random sample, so the log_bf10 number can be
    # checked against what the track actually looks like.
    top = master.sort("log_bf10", descending=True).head(N_GALLERY)
    fig = plot_trajectory_gallery(
        short_tracks, top["particle"].to_list(),
        [f"logBF10={v:+.2f}, eps={e:.2f}" for v, e in zip(top["log_bf10"], top["eps_median"])],
        f"Highest log BF10 tracks (top {N_GALLERY} of {master.height})",
    )
    fig.savefig(FIG_DIR / "trajectory_gallery_top_logbf.png", dpi=150, bbox_inches="tight")

    random_sample = master.sample(n=N_GALLERY, seed=0)
    fig = plot_trajectory_gallery(
        short_tracks, random_sample["particle"].to_list(),
        [f"logBF10={v:+.2f}, eps={e:.2f}" for v, e in
         zip(random_sample["log_bf10"], random_sample["eps_median"])],
        f"Random sample of tracks (n={N_GALLERY})",
    )
    fig.savefig(FIG_DIR / "trajectory_gallery_random.png", dpi=150, bbox_inches="tight")

    # Spatial maps: does log_bf10 (or eps) cluster in the field of view?
    fig = plot_spatial_map(master, "log_bf10", "Mobile beads: log BF10 by track position", diverging=True)
    fig.savefig(FIG_DIR / "spatial_map_log_bf.png", dpi=150, bbox_inches="tight")
    fig = plot_spatial_map(master, "eps_median", "Mobile beads: eps posterior median by track position",
                            diverging=False)
    fig.savefig(FIG_DIR / "spatial_map_eps.png", dpi=150, bbox_inches="tight")

    # Cross-method agreement + interval honesty for the tracks driving the result.
    fig = plot_eps_vs_log_bf(master, "Mobile beads: eps posterior vs. log BF10")
    fig.savefig(FIG_DIR / "eps_vs_log_bf_scatter.png", dpi=150, bbox_inches="tight")
    fig = plot_eps_forest(master, "Mobile beads: eps posterior intervals", top_n=30, sort_by="log_bf10")
    fig.savefig(FIG_DIR / "eps_forest_top30.png", dpi=150, bbox_inches="tight")

    # The generic Jeffreys buckets say nothing about how much a *specific*
    # M=185-track ensemble sum can fluctuate under a true null by chance --
    # calibrate directly against a matched-composition null instead.
    composition = {
        int(tl): int(n) for tl, n in
        short_tracks.group_by("track_length").agg(n=pl.col("particle").n_unique()).sort("track_length").iter_rows()
    }
    print(f"\nRunning {N_NULL}-replicate matched-composition null calibration "
          f"(composition={composition}) ...")
    t0 = time.time()
    null_sums = null_calibration(composition, PARAMS.dt_s, prior, n_null=N_NULL, n_mc=N_MC)
    p_value = float(np.mean(null_sums >= observed_sum))
    print(f"Null calibration done in {time.time() - t0:.1f}s")
    print(f"  null ensemble-sum distribution: median={np.median(null_sums):+.2f}  "
          f"p90={np.percentile(null_sums,90):+.2f}  p99={np.percentile(null_sums,99):+.2f}  "
          f"max={null_sums.max():+.2f}")
    print(f"  observed = {observed_sum:+.3f}  ->  empirical null p-value = {p_value:.4f} "
          f"(fraction of {N_NULL} matched-null replicates >= observed)")
    pl.DataFrame({"null_sum_log_bf10": null_sums}).write_csv(TABLE_DIR / "null_calibration_ensemble_sums.csv")

    verdict = "consistent with isotropic diffusion" if p_value > 0.05 else "unexpected excess vs. matched null -- see FINDINGS.md"
    print(f"\nVerdict on the whole short-track population: {verdict}")


if __name__ == "__main__":
    main()
