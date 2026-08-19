"""Anisotropy-detection Bayes factor for the real mobile-bead SPT dataset,
restricted to its short (track_length 5-10) tracks -- the regime
`bayes.anisotropy` targets. See WORKFLOW.md for how this workflow relates to
the low-data single-track and bulk-population workflows; this script is the
one demonstrating `bayes.anisotropy.analyze`/`null_calibration`.

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

`bayes.anisotropy.analyze` also fits the per-track `eps`/`psi` posterior
(NUTS) alongside the Bayes factor and joins both onto one
`per_track_master.csv`, each quantity kept under its own explicit name
(`log_bf10` vs. `eps_median`/`eps_lo`/`eps_hi` vs. `psi_median_rad`) -- see
FINDINGS.md for why a per-track `eps` point/interval has limited power on
its own at this N (it's included here as a secondary, honestly-wide
per-track descriptive column, not as a second detector); `log_bf10` remains
the only quantity this script's verdicts are based on. The master table also
carries each track's mean field-of-view position, so every quantity can be
mapped back onto the real trajectories and inspected visually
(`plot_trajectory_gallery`, `plot_spatial_map`) rather than trusted as a
bare number.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl

from analysis import AcquisitionParams, assert_contiguous_tracks, load_tracks
from bayes import anisotropy

DATA_CSV = REPO_ROOT / "mobile_beads_1to200.csv"
WORKFLOW = "anisotropy"
FIG_DIR = REPO_ROOT / "results" / "figures" / WORKFLOW
TABLE_DIR = REPO_ROOT / "results" / "tables" / WORKFLOW

PARAMS = AcquisitionParams(pixel_size_um=0.1043, dt_s=0.033)
MIN_TRACK_LENGTH, MAX_TRACK_LENGTH = 5, 10  # the N=5-10 regime this method targets
N_MC = 20000
N_NULL = 200  # matched-composition null-calibration replicates
N_GALLERY = 9  # tracks shown per trajectory-gallery panel


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(DATA_CSV, PARAMS)
    assert_contiguous_tracks(tracks)
    short_tracks = tracks.filter(
        (tracks["track_length"] >= MIN_TRACK_LENGTH) & (tracks["track_length"] <= MAX_TRACK_LENGTH)
    )
    print(f"Loaded {tracks['track_id'].n_unique()} tracks total; "
          f"{short_tracks['track_id'].n_unique()} have track_length in "
          f"[{MIN_TRACK_LENGTH}, {MAX_TRACK_LENGTH}]")

    prior = anisotropy.AnisotropicModelPrior()
    t0 = time.time()
    result = anisotropy.analyze(
        tracks, PARAMS.dt_s, prior=prior,
        min_track_length=MIN_TRACK_LENGTH, max_track_length=MAX_TRACK_LENGTH,
        n_mc=N_MC, seed=0,
    )
    per_track = result.per_track
    print(f"Computed log BF10 + eps/psi posterior for {per_track.height} tracks in {time.time() - t0:.1f}s")

    # per_track_log_bf.csv / per_track_eps_posterior.csv: the two halves
    # analyze() already joined into per_track_master.csv, kept as their own
    # files too (README's documented schema) -- plain column selects, no
    # recomputation.
    per_track.select(["track_id", "track_length", "n_disp", "log_bf10"]).write_csv(
        TABLE_DIR / "per_track_log_bf.csv"
    )
    eps_cols = [c for c in per_track.columns if c not in ("log_bf10", "x_mean_um", "y_mean_um")]
    per_track.select(eps_cols).write_csv(TABLE_DIR / "per_track_eps_posterior.csv")
    per_track.write_csv(TABLE_DIR / "per_track_master.csv")
    print(f"Saved per-track master table ({per_track.height} tracks, {len(per_track.columns)} columns) "
          f"to {TABLE_DIR / 'per_track_master.csv'}")

    logbf = per_track["log_bf10"].to_numpy()
    frac_moderate = float(np.mean(logbf > 1.1))
    frac_strong = float(np.mean(logbf > 2.3))
    print(f"\n=== Per-track log BF10 (n={len(logbf)}) ===")
    print(f"  median={np.median(logbf):+.4f}  mean={np.mean(logbf):+.4f}  "
          f"IQR=[{np.percentile(logbf,25):+.4f}, {np.percentile(logbf,75):+.4f}]")
    print(f"  fraction reaching 'moderate' evidence (>1.1): {100*frac_moderate:.1f}%")
    print(f"  fraction reaching 'strong' evidence   (>2.3): {100*frac_strong:.1f}%")

    fig = anisotropy.plot_log_bf_distribution(
        per_track, "Mobile beads, track_length 5-10: per-track log BF10 (anisotropy)"
    )
    fig.savefig(FIG_DIR / "per_track_log_bf_distribution.png", dpi=150, bbox_inches="tight")

    # Ensemble-level verdict: (a) the whole short-track population as one
    # group -- result.ensemble is already that sum, analyze()'s default
    # grouping -- and (b) broken out by track_length, reusing the same
    # per_track table (aggregate_log_bayes_factor is a plain group-by-sum,
    # no extra Monte Carlo cost).
    ensemble_all = result.ensemble
    ensemble_by_length = anisotropy.aggregate_log_bayes_factor(per_track, "track_length")
    observed_sum = float(ensemble_all["sum_log_bf10"][0])

    print(f"\n=== Ensemble log BF10 (sum across tracks) ===")
    print(ensemble_all)
    print(ensemble_by_length)
    ensemble_all.write_csv(TABLE_DIR / "ensemble_log_bf_all.csv")
    ensemble_by_length.write_csv(TABLE_DIR / "ensemble_log_bf_by_track_length.csv")

    # Trajectory gallery: the highest-log_bf10 tracks (the ones driving the
    # ensemble result) vs. a random sample, so the log_bf10 number can be
    # checked against what the track actually looks like.
    top = per_track.sort("log_bf10", descending=True).head(N_GALLERY)
    fig = anisotropy.plot_trajectory_gallery(
        short_tracks, top["track_id"].to_list(),
        [f"logBF10={v:+.2f}, eps={e:.2f}" for v, e in zip(top["log_bf10"], top["eps_median"])],
        f"Highest log BF10 tracks (top {N_GALLERY} of {per_track.height})",
    )
    fig.savefig(FIG_DIR / "trajectory_gallery_top_logbf.png", dpi=150, bbox_inches="tight")

    random_sample = per_track.sample(n=N_GALLERY, seed=0)
    fig = anisotropy.plot_trajectory_gallery(
        short_tracks, random_sample["track_id"].to_list(),
        [f"logBF10={v:+.2f}, eps={e:.2f}" for v, e in
         zip(random_sample["log_bf10"], random_sample["eps_median"])],
        f"Random sample of tracks (n={N_GALLERY})",
    )
    fig.savefig(FIG_DIR / "trajectory_gallery_random.png", dpi=150, bbox_inches="tight")

    # Spatial maps: does log_bf10 (or eps) cluster in the field of view?
    fig = anisotropy.plot_spatial_map(per_track, "log_bf10", "Mobile beads: log BF10 by track position", diverging=True)
    fig.savefig(FIG_DIR / "spatial_map_log_bf.png", dpi=150, bbox_inches="tight")
    fig = anisotropy.plot_spatial_map(per_track, "eps_median", "Mobile beads: eps posterior median by track position",
                                       diverging=False)
    fig.savefig(FIG_DIR / "spatial_map_eps.png", dpi=150, bbox_inches="tight")

    # Cross-method agreement + interval honesty for the tracks driving the result.
    fig = anisotropy.plot_eps_vs_log_bf(per_track, "Mobile beads: eps posterior vs. log BF10")
    fig.savefig(FIG_DIR / "eps_vs_log_bf_scatter.png", dpi=150, bbox_inches="tight")
    fig = anisotropy.plot_eps_forest(per_track, "Mobile beads: eps posterior intervals", top_n=30, sort_by="log_bf10")
    fig.savefig(FIG_DIR / "eps_forest_top30.png", dpi=150, bbox_inches="tight")

    # The generic Jeffreys buckets say nothing about how much a *specific*
    # M=185-track ensemble sum can fluctuate under a true null by chance --
    # calibrate directly against a matched-composition null instead.
    composition = {
        int(tl): int(n) for tl, n in
        short_tracks.group_by("track_length").agg(n=pl.col("track_id").n_unique()).sort("track_length").iter_rows()
    }
    print(f"\nRunning {N_NULL}-replicate matched-composition null calibration "
          f"(composition={composition}) ...")
    t0 = time.time()
    null_sums = anisotropy.null_calibration(composition, PARAMS.dt_s, prior, n_null=N_NULL, n_mc=N_MC)
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
