"""Exact-likelihood Bayesian NUTS diagnostics for the mobile-bead SPT dataset.

No MSD curve is computed or fit anywhere in this script. K, alpha, and the
localization precision sigma are estimated directly from the per-frame
displacement sequence via the exact Gaussian likelihood in `bayes.likelihood`
(Michalet & Berglund 2012's framework, generalized to anomalous diffusion
via fractional Gaussian noise), through numpyro models in `bayes.model`.

This is a per-track diagnostic tool, not a bulk production pipeline: bulk
per-track diffusivity estimation is `diffusionkit.classic.posterior`'s job
(the grid posterior over D). Here, `bayes.fit_track` (always full NUTS) runs
on a handful of representative tracks (by track_length percentile) to
inspect posterior shape directly -- corner plots and chain traces -- where a
short or weakly-identified track makes that shape worth looking at.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpyro

numpyro.set_host_device_count(
    4
)  # must precede any jax device use -- for real parallel NUTS chains

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run straight from a clone without installing: put the repo root ahead of
# sys.path so `import diffusionkit` resolves. Harmless once pip-installed.
sys.path.insert(0, str(REPO_ROOT))

import polars as pl
from diffusionkit.classic import AcquisitionParams, assert_contiguous_tracks, load_tracks
from diffusionkit.bayes import fit_track
from diffusionkit.bayes.viz import plot_mcmc_trace, plot_posterior_corner, samples_dict_to_arrays

DATA_CSV = REPO_ROOT / "mobile_beads_1to200.csv"
FIG_DIR = REPO_ROOT / "results" / "figures" / "bayes"

PARAMS = AcquisitionParams(pixel_size_um=0.1043, dt_s=0.033)
MCMC_NUM_WARMUP = 500
MCMC_NUM_SAMPLES = 1000
MCMC_NUM_CHAINS = 4


def pick_representative_tracks(tracks: pl.DataFrame, n: int = 3) -> list[int]:
    """Shortest, median-length, and longest tracks -- chosen by track_length
    percentile so the NUTS examples span the short/noisy-to-long/precise
    range a real dataset covers."""
    lengths = (
        tracks.select(["track_id", "track_length"]).unique().sort("track_length")
    )
    qs = [0.1, 0.5, 0.95][:n]
    chosen = [
        int(lengths["track_id"][int(round(q * (lengths.height - 1)))])
        for q in qs
    ]
    return sorted(set(chosen))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(DATA_CSV, PARAMS)
    assert_contiguous_tracks(tracks)
    print(f"Loaded {tracks.height} localizations, {tracks['track_id'].n_unique()} tracks")

    example_particles = pick_representative_tracks(tracks, n=3)
    print(f"\n=== Full NUTS posteriors on representative tracks "
          f"({MCMC_NUM_CHAINS} chains x {MCMC_NUM_WARMUP}+{MCMC_NUM_SAMPLES} steps) ===")
    for pid in example_particles:
        grp = tracks.filter(pl.col("track_id") == pid).sort("frame")
        n_frames = grp.height

        t0 = time.time()
        nuts_fit = fit_track(
            grp, PARAMS.dt_s, model="anomalous",
            num_warmup=MCMC_NUM_WARMUP, num_samples=MCMC_NUM_SAMPLES,
            num_chains=MCMC_NUM_CHAINS, seed=pid,
        )
        dt = time.time() - t0
        samples, _mcmc = nuts_fit.raw
        flat, trace = samples_dict_to_arrays(samples, ["K", "sigma", "alpha"])
        print(f"  track {pid} (track_length={n_frames}): {dt:.1f}s, "
              f"posterior median K={nuts_fit.params['K']:.4g}, alpha={nuts_fit.params['alpha']:.3f}, "
              f"sigma={nuts_fit.params['sigma']:.4g}")

        fig = plot_posterior_corner(flat, ["K", "sigma_loc", "alpha"], ["um2/s^a", "um", ""])
        fig.savefig(FIG_DIR / f"bayes_posterior_corner_particle{pid}.png", dpi=150)

        fig = plot_mcmc_trace(trace, ["K", "sigma_loc", "alpha"], ["um2/s^a", "um", ""])
        fig.savefig(FIG_DIR / f"bayes_trace_particle{pid}.png", dpi=150)

    print(f"\nSaved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
