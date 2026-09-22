"""Ground-truth fBm (+ static localization noise) track generator.

Generalizes `analysis.simulate.simulate_brownian_tracks` from pure Brownian
motion to arbitrary anomalous exponent alpha, by drawing displacements
directly from the same fGn covariance matrix (`likelihood.fgn_covariance`)
the likelihood model assumes -- so the generative model and the inference
model are exactly the same process, not merely similar. At alpha=1 this is
mathematically identical to `simulate_brownian_tracks` (see
`likelihood.fgn_gamma`'s alpha=1 reduction), which is itself a useful
cross-check between the two simulators.

Output uses the same core schema as `analysis.io.load_tracks` (track_id,
frame, t_s, x_um, y_um, sigma_x_um, sigma_y_um, track_length), plus
true_K_um2_s_alpha and true_alpha ground-truth columns, so it flows through
either `analysis`'s MSD pipeline or this package's likelihood-based one
unmodified.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .likelihood import fgn_covariance


def simulate_fbm_tracks(
    params_um2_s_alpha: list[tuple[float, float]],
    n_replicates: int,
    track_length: int,
    dt_s: float,
    sigma_loc_um: float,
    seed: int = 0,
) -> pl.DataFrame:
    """Simulate 2D fBm tracks (+ iid localization noise) for each (K,
    alpha) pair in `params_um2_s_alpha`, `n_replicates` tracks each.

    The Cholesky factor of the fGn covariance depends only on (K,
    alpha, track_length), not on the random draw, so it's built once per
    parameter pair and reused across all `n_replicates` tracks -- avoids
    n_replicates redundant O(track_length^3) factorizations.
    """
    rng = np.random.default_rng(seed)
    frame = np.arange(track_length)
    n_disp = track_length - 1

    rows: dict[str, list] = {
        "track_id": [], "frame": [], "t_s": [], "x_um": [], "y_um": [],
        "sigma_x_um": [], "sigma_y_um": [], "track_length": [],
        "true_K_um2_s_alpha": [], "true_alpha": [],
    }

    track_id = 0
    for K, alpha in params_um2_s_alpha:
        cov = np.asarray(fgn_covariance(n_disp, K, dt_s, alpha))
        L = np.linalg.cholesky(cov)
        for _ in range(n_replicates):
            dx = L @ rng.standard_normal(n_disp)
            dy = L @ rng.standard_normal(n_disp)
            x_true = np.concatenate([[0.0], np.cumsum(dx)])
            y_true = np.concatenate([[0.0], np.cumsum(dy)])
            x_obs = x_true + rng.normal(0.0, sigma_loc_um, size=track_length)
            y_obs = y_true + rng.normal(0.0, sigma_loc_um, size=track_length)

            rows["track_id"].extend([track_id] * track_length)
            rows["frame"].extend(frame.tolist())
            rows["t_s"].extend((frame * dt_s).tolist())
            rows["x_um"].extend(x_obs.tolist())
            rows["y_um"].extend(y_obs.tolist())
            rows["sigma_x_um"].extend([sigma_loc_um] * track_length)
            rows["sigma_y_um"].extend([sigma_loc_um] * track_length)
            rows["track_length"].extend([track_length] * track_length)
            rows["true_K_um2_s_alpha"].extend([K] * track_length)
            rows["true_alpha"].extend([alpha] * track_length)
            track_id += 1

    return pl.DataFrame(rows)
