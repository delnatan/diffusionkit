"""Where along D the tracks of each length sit: the population of D split by track length.

Track length is observed, not inferred, and it is tied to D: a fast particle crosses the focal
depth in a few frames, so it makes short tracks, while a slow one stays in focus and makes one
long track. Splitting the population of D by track length shows which tracks each part of it
comes from.

The tracks are split into a few length groups, and each group's population of D is deconvolved
from its own tracks' log-likelihood rows (`deconvolve.deconvolve`): a population of any shape per
group, with its own smoothness and posterior draws. The groups are stacked by their share of the
weight (tracks, or detections):

    contribution_b(D) = share_b g_b(D)

so the stack is a mixture of the groups' populations, and sums to 1. It is close to, but not the
same as, the whole table's deconvolution: each group's evidence picks its own smoothness. A group
with few tracks is mostly its smoothness prior, which its band shows. Within a group each track
counts once, also per detection, where only the groups' shares change.

This replaces the average of the tracks' flat-prior posteriors, which is the likelihood of
nothing: a short track's share of it was mostly the flat prior's shape.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .data import GridLikelihoods
from .deconvolve import Deconvolution, deconvolve

LENGTH_EDGES = (3, 5, 10, 20)  # lower edges, in frames; the last group is open-ended
WEIGHTS = ("tracks", "detections")


@dataclass(frozen=True)
class LengthComposition:
    """A population of D on `u` = ln D split into track-length groups.

    Group j holds tracks with edges[j] <= n_frames < edges[j + 1]; the last group has no upper
    edge. `populations[j]` is group j's own deconvolved population (None for a group with no
    tracks), and `deconvolved[j]` its weights times `share()[j]`, so the groups and the grid
    together sum to 1.
    """

    u: np.ndarray
    edges: np.ndarray            # (n_groups,) lower edges in frames
    weight: str                  # "tracks" or "detections"
    n_tracks: np.ndarray         # (n_groups,)
    n_detections: np.ndarray     # (n_groups,)
    populations: tuple[Deconvolution | None, ...]  # (n_groups,)
    deconvolved: np.ndarray      # (n_groups, len(u)) each group's weights times its share

    def labels(self) -> list[str]:
        """'3-4', '5-9', '20+': each group's track lengths in frames."""
        hi = list(self.edges[1:] - 1) + [None]
        return [f"{a}+" if b is None else (f"{a}" if a == b else f"{a}-{b}") for a, b in zip(self.edges, hi)]

    def share(self) -> np.ndarray:
        """(n_groups,) each group's fraction of all tracks or detections, as `weight` says."""
        w = self.n_tracks if self.weight == "tracks" else self.n_detections
        return w / w.sum()

    def with_weight(self, weight: str) -> LengthComposition:
        """The same groups' populations, shared by `weight` ("tracks" or "detections") instead: nothing is refitted."""
        if weight not in WEIGHTS:
            raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
        w = self.n_tracks if weight == "tracks" else self.n_detections
        return replace(self, weight=weight, deconvolved=_stack(self.populations, w / w.sum(), len(self.u)))

    def band(self, level: float = .68) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper), each (n_groups, len(u)): every group's pointwise band, times its share like `deconvolved`."""
        lo, hi = np.zeros_like(self.deconvolved), np.zeros_like(self.deconvolved)
        for j, (pop, s) in enumerate(zip(self.populations, self.share())):
            if pop is not None:
                b = pop.band(level)
                lo[j], hi[j] = s * b[0], s * b[1]
        return lo, hi


def _stack(populations, share: np.ndarray, n_u: int) -> np.ndarray:
    out = np.zeros((len(populations), n_u))
    for j, pop in enumerate(populations):
        if pop is not None:
            out[j] = share[j] * pop.weights
    return out


def _groups(n_frames: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, n_frames, side="right") - 1


def by_track_length(likelihoods: GridLikelihoods, u: np.ndarray, edges=None,
                    weight: str = "tracks", n_samples: int = 1000) -> LengthComposition:
    """Deconvolve the population of D in each track-length group, and stack the groups.

    `likelihoods` is `analyze_tracks(..., keep_likelihoods=True).likelihoods`, on the grid `u`
    (`options.u_D()`, evenly spaced).
    `edges` are the groups' lower edges in frames, increasing, the first at most the shortest
    track; by default `LENGTH_EDGES` up to the longest track. `weight` sets each group's share:
    its tracks ("tracks") or its frames ("detections"); `with_weight` switches it without refitting.
    `n_samples` is each group's posterior draws.
    """
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
    lp = np.asarray(likelihoods.loglik_D, float)
    n_frames = np.asarray(likelihoods.n_frames)
    u = np.asarray(u, float)
    if not len(lp):
        raise ValueError("no track likelihoods to split")
    if lp.shape[1] != len(u):
        raise ValueError(f"likelihoods have {lp.shape[1]} grid points, u has {len(u)}")
    if edges is None:
        edges = [e for e in LENGTH_EDGES if e <= n_frames.max()]
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 1 or not len(edges) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"edges must be increasing lower edges in frames, got {edges.tolist()}")
    if n_frames.min() < edges[0]:
        raise ValueError(f"the first edge ({edges[0]}) is above the shortest track ({n_frames.min()} frames)")
    group = _groups(n_frames, edges)
    n_tracks = np.bincount(group, minlength=len(edges))
    n_detections = np.bincount(group, weights=n_frames, minlength=len(edges)).astype(np.int64)
    w = n_tracks if weight == "tracks" else n_detections
    populations = tuple(deconvolve(lp[group == j], u, n_samples=n_samples) if n_tracks[j] else None
                        for j in range(len(edges)))
    return LengthComposition(u=u, edges=edges, weight=weight, n_tracks=n_tracks, n_detections=n_detections,
                             populations=populations, deconvolved=_stack(populations, w / w.sum(), len(u)))
