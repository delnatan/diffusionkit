"""Spatial-class inputs and outputs. The model is `potts`, the run `workflow`."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .potts import PottsFit, conditional


@dataclass(frozen=True)
class SpatialOptions:
    """Everything a spatial run depends on besides the likelihoods and the positions.

    - `k`: neighbours per track in the graph.
    - `max_classes`: classes are fitted for C = 1 .. max_classes.
    - `n_classes`: the C to use; None takes the suggested one, the last C whose gain over C - 1 clears
      `gain_per_track` nats per track (the elbow of the score against C).
    - `n_shuffles`: position shuffles for beta's null. The p bound is (1 + shuffles at or above beta) /
      (1 + n_shuffles), so 99 shuffles resolve p down to 0.01.
    - `p_pattern`: the p bound at or below which the classes count as spatially organized.
    - `level`: credible-interval mass of the per-track spatial D columns (gridpost's default).
    """
    k: int = 8
    max_classes: int = 4
    n_classes: int | None = None
    gain_per_track: float = .02
    n_shuffles: int = 99
    p_pattern: float = .05
    level: float = .9
    seed: int = 0
    max_iter: int = 300
    tol: float = 1e-3

    def __post_init__(self):
        if self.k < 1:
            raise ValueError(f"k must be at least 1, got {self.k}")
        if self.max_classes < 1:
            raise ValueError(f"max_classes must be at least 1, got {self.max_classes}")
        if self.n_classes is not None and not 1 <= self.n_classes <= self.max_classes:
            raise ValueError(f"n_classes must be in 1..max_classes ({self.max_classes}), got {self.n_classes}")
        if self.n_shuffles < 0:
            raise ValueError(f"n_shuffles must be >= 0, got {self.n_shuffles}")
        if not 0 < self.level < 1:
            raise ValueError(f"level must be in (0, 1), got {self.level}")


@dataclass(frozen=True)
class ShuffleNull:
    """The neighbour gain against the same tracks with their positions shuffled.

    The neighbour gain is how many nats the neighbours add to predicting each track's data: the fit's
    neighbour-predictive score minus the plain mixture's (beta = 0), which no arrangement changes. Shuffling
    keeps every track's data and destroys only the arrangement, so the shuffled gains are its distribution with
    no spatial pattern. beta itself is a poor statistic: when neighbours carry nothing, beta and the class
    weights trade off along a flat ridge and beta wanders.
    """
    gain: float
    beta: float
    shuffled_gain: np.ndarray  # (n_shuffles,)
    shuffled_beta: np.ndarray
    p: float                   # (1 + shuffles with gain >= observed) / (1 + n_shuffles)


@dataclass(frozen=True)
class SpatialAnalysis:
    """One spatial run over the tracks `track_ids` (the gridpost "ok" tracks that have a position).

    - `fits[C]`: the fit with C classes (`potts.PottsFit`), classes numbered 1..C in increasing order of D.
    - `selection`: one row per C -- `n_classes`, `score` (neighbour-predictive log-likelihood), `gain` and
      `gain_per_track` over C - 1, `beta`, `suggested`.
    - `n_classes`: the C used; `null`: beta's shuffle null for it (None for C = 1 or no shuffles).
    - `D_spatial`: (n, 3) per-track mean, low and high of D (um^2/s) with the class mixture as the prior.
    """
    track_ids: np.ndarray
    n_frames: np.ndarray
    xy_um: np.ndarray
    neighbours: np.ndarray
    u_D: np.ndarray
    options: SpatialOptions
    fits: dict[int, PottsFit]
    selection: pl.DataFrame
    n_classes: int
    null: ShuffleNull | None
    D_spatial: np.ndarray = field(repr=False)

    @property
    def fit(self) -> PottsFit:
        return self.fits[self.n_classes]

    @property
    def patterned(self) -> bool:
        """Whether the classes are spatially organized: C > 1 and beta above its shuffle null."""
        return self.null is not None and self.null.p <= self.options.p_pattern

    def neighbour_q(self) -> np.ndarray:
        """(n, C) class probabilities from each track's neighbours alone."""
        return self.fit.neighbour_q(self.neighbours)

    def columns(self) -> pl.DataFrame:
        """One row per track: `spatial_class` (1..C, own data and neighbours) and `spatial_p`;
        `spatial_class_neighbours` (neighbours alone, the grouping for per-class populations) and
        `spatial_p_neighbours`; `D_spatial_mean/lo/hi_um2_s`."""
        q, qn = self.fit.q, self.neighbour_q()
        rows = np.arange(len(q))
        c, cn = q.argmax(axis=1), qn.argmax(axis=1)
        return pl.DataFrame({
            "track_id": self.track_ids,
            "spatial_class": (c + 1).astype(np.int64),
            "spatial_p": q[rows, c],
            "spatial_class_neighbours": (cn + 1).astype(np.int64),
            "spatial_p_neighbours": qn[rows, cn],
            "D_spatial_mean_um2_s": self.D_spatial[:, 0],
            "D_spatial_lo_um2_s": self.D_spatial[:, 1],
            "D_spatial_hi_um2_s": self.D_spatial[:, 2],
        })

    def groups(self) -> dict[int, np.ndarray]:
        """{class: its track ids} by `spatial_class_neighbours`: each track in the class its neighbours give it,
        so a class's tracks are not selected on their own data. Pass a group's ids to any population model."""
        cn = self.neighbour_q().argmax(axis=1) + 1
        return {c: self.track_ids[cn == c] for c in range(1, self.n_classes + 1)}

    def report(self) -> pl.DataFrame:
        """One row per class, counts only: `n_tracks` and `n_detections` in its group (`groups`),
        `n_assigned` (tracks whose `spatial_class` it is) and `n_confident` (of those, `spatial_p` > 0.9),
        and `share`, its weight. What the class's tracks look like is their population, not a number here."""
        q = self.fit.q
        c = q.argmax(axis=1) + 1
        cn = self.neighbour_q().argmax(axis=1) + 1
        confident = q.max(axis=1) > .9
        cls = np.arange(1, self.n_classes + 1)
        return pl.DataFrame({
            "spatial_class": cls,
            "n_tracks": [int((cn == k).sum()) for k in cls],
            "n_detections": [int(self.n_frames[cn == k].sum()) for k in cls],
            "n_assigned": [int((c == k).sum()) for k in cls],
            "n_confident": [int(((c == k) & confident).sum()) for k in cls],
            "share": self.fit.weight,
        })

    def field(self, xy_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """At query points (m, 2) in um: (m, C) class probabilities from the k nearest tracks (the Potts
        conditional) and (m,) the distance to the k-th of them, to blank points no track is near."""
        from scipy.spatial import cKDTree
        dist, idx = cKDTree(self.xy_um).query(np.asarray(xy_um, float), self.options.k)
        idx = idx.reshape(len(idx), -1)
        dist = dist.reshape(len(dist), -1)
        return conditional(self.fit.q[idx].mean(axis=1), self.fit.weight, self.fit.beta), dist[:, -1]

    def typical_spacing_um(self) -> float:
        """Median distance from a track to its k-th neighbour."""
        d = np.linalg.norm(self.xy_um[self.neighbours[:, -1]] - self.xy_um, axis=1)
        return float(np.median(d))

