"""Spatial-class inputs and outputs. The model is `potts`, the run `workflow`."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from .potts import PottsFit, conditional

MODEL = "potts-exact-beta"  # the model a record was saved by: no class weights, beta per edge by its exact M-step


@dataclass(frozen=True)
class SpatialOptions:
    """Everything a spatial run depends on besides the likelihoods and the positions.

    - `k`: neighbours per track in the graph, before it is made symmetric (i ~ j if either lists the other).
    - `max_classes`: classes are fitted for C = 1 .. max_classes.
    - `n_classes`: the C to use; None takes the suggested one, the last C whose gain over C - 1 clears
      `gain_per_track` nats per track (the elbow of the score against C).
    - `n_shuffles`: position shuffles for the pattern test's null. p is (1 + shuffles whose neighbour gain is
      at least the observed) / (1 + n_shuffles), so 99 shuffles resolve p down to 0.01.
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
    no spatial pattern. beta is kept with each shuffle; without class weights it is identified, and near 0 when
    the neighbours carry nothing.
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
    - `n_classes`: the C used; `null`: the shuffle null of its neighbour gain (None for C = 1 or no shuffles).
    - `adjacency`: (n, n) the symmetric k-nearest-neighbour graph the fits use.
    - `D_spatial`: (n, 3) per-track mean, low and high of D (um^2/s) with the class mixture as the prior.
    """
    track_ids: np.ndarray
    n_frames: np.ndarray
    xy_um: np.ndarray
    adjacency: csr_matrix
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
        """Whether the classes are spatially organized: C > 1 and the neighbour gain above its shuffle null."""
        return self.null is not None and self.null.p <= self.options.p_pattern

    def neighbour_q(self) -> np.ndarray:
        """(n, C) class probabilities from each track's neighbours alone."""
        return self.fit.neighbour_q(self.adjacency)

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
        `n_assigned` (tracks whose `spatial_class` it is) and `n_confident` (of those, `spatial_p` > 0.9). What
        the class's tracks look like is their population, not a number here."""
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
        })

    def field(self, xy_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """At query points (m, 2) in um: (m, C) class probabilities from the k nearest tracks (the Potts
        conditional, with those k as the point's neighbours) and (m,) the distance to the k-th of them, to blank
        points no track is near."""
        from scipy.spatial import cKDTree
        dist, idx = cKDTree(self.xy_um).query(np.asarray(xy_um, float), self.options.k)
        idx = idx.reshape(len(idx), -1)
        dist = dist.reshape(len(dist), -1)
        return conditional(self.fit.q[idx].sum(axis=1), self.fit.beta), dist[:, -1]

    def typical_spacing_um(self) -> float:
        """Median distance from a track to its k-th neighbour."""
        from scipy.spatial import cKDTree
        d, _ = cKDTree(self.xy_um).query(self.xy_um, self.options.k + 1)
        return float(np.median(d[:, -1]))


    def probabilities(self) -> pl.DataFrame:
        """One row per track: `track_id`, `x_um`, `y_um` and `q_1`..`q_C`, the fit's class probabilities. With
        `record` it is what `workflow.restore` rebuilds the analysis from."""
        q = self.fit.q
        return pl.DataFrame({"track_id": self.track_ids, "x_um": self.xy_um[:, 0], "y_um": self.xy_um[:, 1],
                             **{f"q_{c + 1}": q[:, c] for c in range(q.shape[1])}})

    def record(self) -> dict:
        """The run as plain JSON values: options, the selection table, the C used and its classes (in ln D),
        the shuffle null and the pattern verdict, the typical spacing and the report."""
        f, null = self.fit, self.null
        return {
            "model": MODEL,
            "options": asdict(self.options),
            "n_tracks": len(self.track_ids),
            "n_classes": self.n_classes,
            "selection": self.selection.to_dicts(),
            "classes": {"centre_ln_D": f.centre.tolist(), "sd_ln_D": f.sd.tolist(),
                        "beta": f.beta, "agreement": f.agreement, "n_iter": f.n_iter, "converged": f.converged},
            "null": None if null is None else {
                "gain": null.gain, "beta": null.beta, "p": null.p,
                "shuffled_gain": null.shuffled_gain.tolist(), "shuffled_beta": null.shuffled_beta.tolist()},
            "patterned": self.patterned,
            "typical_spacing_um": self.typical_spacing_um(),
            "report": self.report().to_dicts(),
        }
