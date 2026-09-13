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

`simulate_anisotropic_tracks` is the same idea for the anisotropy
comparison in `nested.py`: draws from
`likelihood.anisotropic_step_covariance` instead of `fgn_covariance`, same
generative/inference-model-share-one-implementation principle, same output
schema plus true_D_mean_um2_s/true_eps/true_psi in place of the fBm
(true_D_mean_um2_s is the *arithmetic* mean (D_par+D_perp)/2, since
D_par = D_mean*(1+eps) -- it pairs with `nested`'s
`D_arith_mean_median_um2_s`, not `D_geom_mean_median_um2_s`)
ground-truth columns.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .likelihood import anisotropic_step_covariance, fgn_covariance


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


def simulate_anisotropic_tracks(
    params: list[tuple[float, float, float]],
    n_replicates: int,
    track_length: int,
    dt_s: float,
    sigma_loc_um: float,
    seed: int = 0,
) -> pl.DataFrame:
    """Simulate 2D anisotropic Brownian tracks (+ iid localization noise)
    for each (D_mean, eps, psi) triple in `params`, `n_replicates` tracks
    each -- the anisotropic counterpart to `simulate_fbm_tracks`. alpha=1
    only (anisotropy + anomalous exponent jointly is out of scope for v1,
    see the project's anisotropy-detection plan).

    Draws true-motion steps from `likelihood.anisotropic_step_covariance`
    (motion only, no noise -- every step is iid at alpha=1, so one 2x2
    Cholesky factor is reused for all `n_disp` steps and all
    `n_replicates`), cumulative-sums to true positions, then adds iid
    per-frame position noise -- same x_true + eps_i construction as
    `simulate_fbm_tracks`, not a direct draw from
    `likelihood.anisotropic_displacement_covariance` (which is the
    *displacement*-space covariance already including noise, i.e. what the
    inference model needs, not what a position-level simulator should
    Cholesky-factor).
    """
    rng = np.random.default_rng(seed)
    frame = np.arange(track_length)
    n_disp = track_length - 1

    rows: dict[str, list] = {
        "track_id": [], "frame": [], "t_s": [], "x_um": [], "y_um": [],
        "sigma_x_um": [], "sigma_y_um": [], "track_length": [],
        "true_D_mean_um2_s": [], "true_eps": [], "true_psi": [],
    }

    track_id = 0
    for D_mean, eps, psi in params:
        step_cov = np.asarray(anisotropic_step_covariance(D_mean, eps, psi, dt_s))
        L = np.linalg.cholesky(step_cov)
        for _ in range(n_replicates):
            steps = (L @ rng.standard_normal((2, n_disp))).T  # (n_disp, 2)
            true_xy = np.concatenate([np.zeros((1, 2)), np.cumsum(steps, axis=0)], axis=0)
            obs_xy = true_xy + rng.normal(0.0, sigma_loc_um, size=(track_length, 2))

            rows["track_id"].extend([track_id] * track_length)
            rows["frame"].extend(frame.tolist())
            rows["t_s"].extend((frame * dt_s).tolist())
            rows["x_um"].extend(obs_xy[:, 0].tolist())
            rows["y_um"].extend(obs_xy[:, 1].tolist())
            rows["sigma_x_um"].extend([sigma_loc_um] * track_length)
            rows["sigma_y_um"].extend([sigma_loc_um] * track_length)
            rows["track_length"].extend([track_length] * track_length)
            rows["true_D_mean_um2_s"].extend([D_mean] * track_length)
            rows["true_eps"].extend([eps] * track_length)
            rows["true_psi"].extend([psi] * track_length)
            track_id += 1

    return pl.DataFrame(rows)
