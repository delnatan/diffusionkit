"""Minimal low-data-regime example: one track in, one Bayesian fit out.

The concrete demonstration that the low-data path (`bayes.fit_track`) is
actually a one-liner -- see WORKFLOW.md for the fuller picture (this
workflow vs. the bulk `fit_population` workflow vs. `bayes.anisotropy`).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from analysis import AcquisitionParams, load_tracks
from bayes import fit_track

DATA_CSV = REPO_ROOT / "mobile_beads_1to200.csv"
PARAMS = AcquisitionParams(pixel_size_um=0.1043, dt_s=0.033)


def main() -> None:
    tracks = load_tracks(DATA_CSV, PARAMS)

    # Pick one track -- e.g. the first track in the file. In practice
    # this is "the one track a user just clicked on" or "today's handful of
    # tracks," not a bulk table.
    track_id = tracks["track_id"][0]
    track = tracks.filter(pl.col("track_id") == track_id)

    # prior=None -> informative default built from this track's own
    # measured localization precision (the honest low-data default);
    # method="map" (default) is fast MAP + Laplace interval.
    fit = fit_track(track, PARAMS.dt_s, model="anomalous")

    print(f"track {fit.track_id} (track_length={fit.track_length}, n_disp={fit.n_disp})")
    print(f"  D_alpha = {fit.params['D_alpha']:.4g} "
          f"({fit.lo['D_alpha']:.4g}, {fit.hi['D_alpha']:.4g}) um^2/s^alpha")
    print(f"  alpha   = {fit.params['alpha']:.3f} "
          f"({fit.lo['alpha']:.3f}, {fit.hi['alpha']:.3f})")
    print(f"  sigma   = {fit.params['sigma']:.4g} "
          f"({fit.lo['sigma']:.4g}, {fit.hi['sigma']:.4g}) um")

    # Want the full posterior shape, not just a Laplace interval? Same call,
    # method="nuts" -- useful when the track is short/noisy enough that a
    # Gaussian approximation is suspect (FINDINGS.md).
    nuts_fit = fit_track(track, PARAMS.dt_s, model="anomalous", method="nuts")
    print(f"\n  (NUTS check) D_alpha median = {nuts_fit.params['D_alpha']:.4g}, "
          f"alpha median = {nuts_fit.params['alpha']:.3f}")


if __name__ == "__main__":
    main()
