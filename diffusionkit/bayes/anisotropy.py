"""Anisotropy detection: is *this* track diffusing anisotropically?

The question is per-track and it is a model comparison, not a point
estimate. `nested.py` holds the implementation (nested sampling over a
log-Euclidean diffusion tensor); this module is the workflow layer -- table
in, table out, plus the plots -- kept out of the bulk `bayes` namespace
(`api.fit_track`/`fit_population`) because it answers a different question
than a per-track D and because importing it pulls in matplotlib.

What replaced what, and why
---------------------------
Earlier versions estimated log BF10 by prior-predictive Monte Carlo
(`bayes_factor.py`, removed) and, separately, fit an eps/psi posterior by
NUTS, then summed per-track evidence into an ensemble score. All three are
gone:

  * The Monte Carlo estimator was only valid while the likelihood stayed
    weak relative to the prior, i.e. exactly on the short tracks that cannot
    answer the question anyway. It degraded silently on the long tracks that
    can. Nested sampling is valid across the whole range.
  * The separate NUTS pass is redundant -- one nested-sampling run yields
    the evidence *and* the posterior, so `analyze` now runs one engine
    instead of two.
  * Ensemble aggregation summed log BF10 across tracks. That is a form of
    ensemble averaging, which is the thing this package exists to avoid, and
    it was actively misleading: the sum answers "does each track have its
    own independent anisotropy", not "do these tracks share an axis", and a
    pooled fit with a shared D manufactures anisotropy out of ordinary
    D-heterogeneity once the spread reaches ~0.5 decades. Per-track
    inference is structurally immune to that, since every track carries its
    own D.

There is also no `max_track_length` any more. The old cap of 10 existed
because the Monte Carlo estimator broke above it; it had the effect of
restricting the analysis to precisely the tracks that hold too little
information to answer the question. See FINDINGS.md for the measured
detectability-vs-track-length table.

Reading the output honestly
---------------------------
`log_bf10` is evidence, not a decision. It is self-calibrating -- a proper
Bayes factor already accounts for how much apparent elongation sampling
noise produces at this track length, so no simulated null reference is
needed (and none is shipped). Positive favours anisotropy, negative favours
isotropy, and near zero means the track does not say. Short tracks return
near zero and that is the honest answer, not a defect: use
`evidence_label`, which also reports when the sampler's own uncertainty is
larger than the signal.
"""
from __future__ import annotations

from collections.abc import Callable

import polars as pl

from .nested import (
    LogEuclideanAnisotropicPrior,
    NestedFit,
    fit_track_nested,
    per_track_nested,
)
from .simulate import simulate_anisotropic_tracks
from .viz import (
    plot_eps_forest,
    plot_eps_vs_log_bf,
    plot_log_bf_distribution,
    plot_spatial_map,
    plot_trajectory_gallery,
)

__all__ = [
    "LogEuclideanAnisotropicPrior",
    "NestedFit",
    "fit_track_nested",
    "per_track_nested",
    "simulate_anisotropic_tracks",
    "analyze",
    "evidence_label",
    "plot_log_bf_distribution",
    "plot_trajectory_gallery",
    "plot_spatial_map",
    "plot_eps_vs_log_bf",
    "plot_eps_forest",
]

# Jeffreys' buckets on log BF10 (natural log): 1.1 ~ 3:1, 2.3 ~ 10:1,
# 4.6 ~ 100:1. Reported as labels rather than a thresholded boolean because
# the useful per-track answer at short track lengths is "this track does not
# say", which a flag cannot express.
_BUCKETS = ((4.6, "decisive"), (2.3, "strong"), (1.1, "moderate"), (0.0, "weak"))


def evidence_label(log_bf10: float, log_bf10_stderr: float = 0.0) -> str:
    """Jeffreys-scale label for one track's `log_bf10`, direction included.

    Returns "inconclusive (below sampler noise)" whenever the evidence is
    smaller than the nested sampler's own uncertainty on it -- the case that
    matters most on short tracks, where a bare number invites reading
    structure into what is really Monte Carlo scatter.
    """
    if abs(log_bf10) <= max(log_bf10_stderr, 1e-12):
        return "inconclusive (below sampler noise)"
    direction = "anisotropic" if log_bf10 > 0 else "isotropic"
    for threshold, name in _BUCKETS:
        if abs(log_bf10) >= threshold:
            return f"{name} evidence for {direction}"
    return f"weak evidence for {direction}"


def analyze(
    tracks: pl.DataFrame,
    dt_s: float,
    prior: LogEuclideanAnisotropicPrior | None = None,
    min_track_length: int = 5,
    seed: int = 0,
    show_progress: bool = True,
    progress: Callable[[int, int], None] | None = None,
    **fit_kwargs,
) -> pl.DataFrame:
    """Per-track anisotropy evidence for every track in `tracks`, one row each.

    Adds the Jeffreys-scale `evidence` label and each track's mean position
    (`x_mean_um`/`y_mean_um`, for `plot_spatial_map`) to what
    `nested.per_track_nested` returns. See TABLES.md for the columns.

    No grouping, pooling, or ensemble sum: every row is one trajectory's own
    answer, computed from that trajectory's own displacements.
    """
    p = prior if prior is not None else LogEuclideanAnisotropicPrior()
    per_track = per_track_nested(
        tracks, dt_s, p, min_track_length=min_track_length, seed=seed,
        show_progress=show_progress, progress=progress, **fit_kwargs,
    )
    geometry = tracks.group_by("track_id").agg(
        x_mean_um=pl.col("x_um").mean(), y_mean_um=pl.col("y_um").mean()
    )
    return (
        per_track.join(geometry, on="track_id")
        .with_columns(
            evidence=pl.struct("log_bf10", "log_bf10_stderr").map_elements(
                lambda r: evidence_label(r["log_bf10"], r["log_bf10_stderr"]),
                return_dtype=pl.String,
            )
        )
        .sort("track_id")
    )
