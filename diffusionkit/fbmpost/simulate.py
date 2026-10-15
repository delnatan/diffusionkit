"""Blurred fBm + static localization noise tracks, simulated independently of the likelihood's blur formula.

The true motion is drawn by Cholesky of the fBm covariance K (t^alpha + s^alpha - |t - s|^alpha) per
axis (MSD_1D(tau) = 2 K tau^alpha) at `n_sub` points spread over each frame's exposure, and each frame
records their mean; with exposure 0 it is the instantaneous position. The cost is one Cholesky of a
(n_frames n_sub) square matrix per call. The mean of n_sub points approaches the exposure average slowly at
small alpha, where the path is rough: at alpha = 0.4 n_sub = 10 reads as alpha 0.03 lower, 40 as unbiased.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..data import Acquisition


def simulate_tracks(n_tracks: int, n_frames: int, alpha: float, D_ref_um2_s: float, sigma_um: float,
                    acquisition: Acquisition, tau_ref_s: float = .1, rng: np.random.Generator | None = None,
                    n_sub: int = 40, first_track_id: int = 0) -> pl.DataFrame:
    """A track table (track_id, frame, x_um, y_um, sigma_x_um, sigma_y_um) of `n_tracks` 2D fBm tracks.

    The motion's D at `tau_ref_s` is `D_ref_um2_s` (`FbmPostOptions.tau_ref_s`), so K = D_ref
    tau_ref^(1 - alpha); D_ref = 0 is an immobile emitter. `sigma_um` is the true and reported SD.
    """
    if not 0 < alpha < 2:
        raise ValueError(f"alpha must be in (0, 2), got {alpha}")
    rng = np.random.default_rng() if rng is None else rng
    dt, te = float(acquisition.dt_s), float(acquisition.exposure_s)
    n_sub = n_sub if te > 0 else 1
    # Times start one frame in, so no sample sits at the fBm's pinned origin.
    t = (dt * (np.arange(n_frames)[:, None] + 1) + te * (np.arange(n_sub)[None, :] + .5) / n_sub).ravel()
    K = D_ref_um2_s * tau_ref_s ** (1 - alpha)
    frames = np.zeros((n_tracks, n_frames, 2))
    if K > 0:
        C = t[:, None] ** alpha + t[None, :] ** alpha - np.abs(t[:, None] - t[None, :]) ** alpha
        L = np.linalg.cholesky(C + 1e-12 * C.diagonal().mean() * np.eye(len(t)))
        fine = np.sqrt(K) * np.einsum("ij,abj->abi", L, rng.standard_normal((n_tracks, 2, len(t))))
        frames = fine.reshape(n_tracks, 2, n_frames, n_sub).mean(-1).transpose(0, 2, 1)
    frames = frames + sigma_um * rng.standard_normal(frames.shape)
    return pl.DataFrame({
        "track_id": np.repeat(np.arange(n_tracks) + first_track_id, n_frames),
        "frame": np.tile(np.arange(n_frames), n_tracks),
        "x_um": frames[..., 0].ravel(), "y_um": frames[..., 1].ravel(),
        "sigma_x_um": np.full(n_tracks * n_frames, sigma_um), "sigma_y_um": np.full(n_tracks * n_frames, sigma_um),
    })
