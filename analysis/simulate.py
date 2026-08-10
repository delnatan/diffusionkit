"""Ground-truth Brownian-motion track generator for validating fit pipelines.

Produces tracks in the same schema `io.load_tracks` produces (particle,
frame, t_s, x_um, y_um, x_std_um, y_std_um, track_length), so simulated data
flows through `compute_all_tamsd` / `fit_all_tracks` / `viz.*` unmodified.
This lets a suspected fitting bias be checked against a known ground truth
rather than just inferred from real data: simulate, run the exact
production pipeline over it, compare recovered parameters to the truth.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def simulate_brownian_tracks(
    D_values_um2_s: list[float],
    n_replicates: int,
    track_length: int,
    dt_s: float,
    sigma_loc_um: float,
    seed: int = 0,
) -> pl.DataFrame:
    """Simulate pure 2D Brownian tracks (true alpha=1) with static localization noise.

    Each of `D_values_um2_s` gets `n_replicates` independent tracks of
    `track_length` frames. Localization noise is i.i.d. Gaussian with std
    `sigma_loc_um`, identical across all tracks regardless of D -- the same
    setup implied by a roughly constant MLE localization precision across
    similarly bright beads.

    Returns a DataFrame with the same core columns as `io.load_tracks`, plus
    a `true_D_um2_s` column carrying the ground truth for validation.
    """
    rng = np.random.default_rng(seed)
    frame = np.arange(track_length)

    rows = {
        "particle": [],
        "frame": [],
        "t_s": [],
        "x_um": [],
        "y_um": [],
        "x_std_um": [],
        "y_std_um": [],
        "track_length": [],
        "true_D_um2_s": [],
    }

    particle_id = 0
    for D in D_values_um2_s:
        step_std = np.sqrt(2 * D * dt_s)
        for _ in range(n_replicates):
            x_true = np.cumsum(rng.normal(0.0, step_std, size=track_length))
            y_true = np.cumsum(rng.normal(0.0, step_std, size=track_length))
            x_obs = x_true + rng.normal(0.0, sigma_loc_um, size=track_length)
            y_obs = y_true + rng.normal(0.0, sigma_loc_um, size=track_length)

            rows["particle"].extend([particle_id] * track_length)
            rows["frame"].extend(frame.tolist())
            rows["t_s"].extend((frame * dt_s).tolist())
            rows["x_um"].extend(x_obs.tolist())
            rows["y_um"].extend(y_obs.tolist())
            rows["x_std_um"].extend([sigma_loc_um] * track_length)
            rows["y_std_um"].extend([sigma_loc_um] * track_length)
            rows["track_length"].extend([track_length] * track_length)
            rows["true_D_um2_s"].extend([D] * track_length)
            particle_id += 1

    return pl.DataFrame(rows)
