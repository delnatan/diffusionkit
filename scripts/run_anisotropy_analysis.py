"""Per-track anisotropy evidence for the real mobile-bead SPT dataset.

One row per trajectory, computed from that trajectory's own displacements.
Nothing here is pooled, grouped, or ensemble-averaged: the question this
script answers is "is *this* track diffusing anisotropically", asked
separately of every track. See WORKFLOW.md for how it relates to the other
workflows, and `diffusionkit.bayes.anisotropy` for why the earlier Monte
Carlo Bayes factor and its ensemble sum were removed.

These beads are freely diffusing (no known structure or channel to confine
them), so this doubles as a real-data negative control: log BF10 should sit
at or below zero, and the tracks long enough to actually resolve the
question should say so decisively.

There is no upper track-length cap. The old one (10) existed because the
Monte Carlo estimator broke above it, and it had the side effect of
restricting the analysis to exactly the tracks that hold too little
information to answer anything. Nested sampling stays valid as tracks get
long, which is where per-track anisotropy becomes answerable at all -- so
this run is dominated, correctly, by the long tracks.

Runtime is roughly 3-5 s per track (two nested-sampling runs each), so a
full pass over this dataset takes ~25-40 minutes. `--limit N` runs the N
longest tracks instead, for a quick look.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run straight from a clone without installing: put the repo root ahead of
# sys.path so `import diffusionkit` resolves. Harmless once pip-installed.
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from diffusionkit.bayes import anisotropy
from diffusionkit.classic import AcquisitionParams, assert_contiguous_tracks, load_tracks

DATA_CSV = REPO_ROOT / "mobile_beads_1to200.csv"
WORKFLOW = "anisotropy"
FIG_DIR = REPO_ROOT / "results" / "figures" / WORKFLOW
TABLE_DIR = REPO_ROOT / "results" / "tables" / WORKFLOW

PARAMS = AcquisitionParams(pixel_size_um=0.1043, dt_s=0.033)
MIN_TRACK_LENGTH = 5
N_GALLERY = 9


def main(limit: int | None = None) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(DATA_CSV, PARAMS)
    assert_contiguous_tracks(tracks)
    tracks = tracks.filter(pl.col("track_length") >= MIN_TRACK_LENGTH)

    if limit is not None:
        keep = (
            tracks.group_by("track_id").agg(n=pl.len())
            .sort("n", descending=True).head(limit)["track_id"].to_list()
        )
        tracks = tracks.filter(pl.col("track_id").is_in(keep))

    n_tracks = tracks["track_id"].n_unique()
    print(f"{n_tracks} tracks with track_length >= {MIN_TRACK_LENGTH}")

    prior = anisotropy.LogEuclideanAnisotropicPrior()
    print(f"prior: tau_log_ratio={prior.tau_log_ratio} "
          f"(prior sd of 0.5*log(D_par/D_perp)) -- report this with any Bayes factor")

    t0 = time.perf_counter()
    per_track = anisotropy.analyze(tracks, PARAMS.dt_s, prior)
    print(f"fit {n_tracks} tracks in {time.perf_counter() - t0:.0f}s")

    per_track.write_csv(TABLE_DIR / "per_track_anisotropy.csv")

    # --- what the evidence actually says, per track ---
    print("\nEvidence label counts (each row is one trajectory's own answer):")
    counts = per_track.group_by("evidence").agg(n=pl.len()).sort("n", descending=True)
    for row in counts.iter_rows(named=True):
        print(f"   {row['n']:>4}  {row['evidence']}")

    print("\nlog BF10 by track-length band -- longer tracks can resolve the question,")
    print("short ones honestly cannot, and their evidence sits near zero:")
    banded = (
        per_track.with_columns(
            band=pl.when(pl.col("track_length") < 10).then(pl.lit("5-9"))
            .when(pl.col("track_length") < 20).then(pl.lit("10-19"))
            .when(pl.col("track_length") < 50).then(pl.lit("20-49"))
            .when(pl.col("track_length") < 100).then(pl.lit("50-99"))
            .otherwise(pl.lit("100+"))
        )
        .group_by("band")
        .agg(n=pl.len(), median_log_bf10=pl.col("log_bf10").median(),
             n_strong=(pl.col("log_bf10") > 2.3).sum(),
             median_stderr=pl.col("log_bf10_stderr").median())
        .sort("median_log_bf10")
    )
    print(banded)
    banded.write_csv(TABLE_DIR / "log_bf10_by_track_length_band.csv")

    # --- figures ---
    fig = anisotropy.plot_log_bf_distribution(per_track, "Mobile beads: per-track log BF10")
    fig.savefig(FIG_DIR / "log_bf10_distribution.png", dpi=150, bbox_inches="tight")

    ranked = per_track.sort("log_bf10", descending=True)
    for name, sub in (("most_anisotropic", ranked.head(N_GALLERY)),
                      ("most_isotropic", ranked.tail(N_GALLERY))):
        fig = anisotropy.plot_trajectory_gallery(
            tracks, sub["track_id"].to_list(),
            [f"N={n}  logBF10={v:+.2f}" for n, v in
             zip(sub["track_length"], sub["log_bf10"])],
            f"Mobile beads: {name.replace('_', ' ')} tracks",
        )
        fig.savefig(FIG_DIR / f"gallery_{name}.png", dpi=150, bbox_inches="tight")

    fig = anisotropy.plot_spatial_map(per_track, "log_bf10",
                                      "Mobile beads: log BF10 by track position")
    fig.savefig(FIG_DIR / "spatial_map_log_bf10.png", dpi=150, bbox_inches="tight")

    fig = anisotropy.plot_eps_vs_log_bf(per_track, "Mobile beads: eps posterior vs. log BF10")
    fig.savefig(FIG_DIR / "eps_vs_log_bf10.png", dpi=150, bbox_inches="tight")

    fig = anisotropy.plot_eps_forest(per_track, "Mobile beads: eps posterior intervals",
                                     top_n=30, sort_by="log_bf10")
    fig.savefig(FIG_DIR / "eps_forest.png", dpi=150, bbox_inches="tight")

    print(f"\nWrote tables to {TABLE_DIR} and figures to {FIG_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="fit only the N longest tracks (quick look)")
    main(**vars(ap.parse_args()))
