"""Where along D the tracks of each length sit: the distribution of D split by track length.

Track length is observed, not inferred, and it is tied to D: a fast particle crosses the focal
depth in a few frames, so it makes short tracks, while a slow one stays in focus and makes one
long track. Splitting a distribution of D by track length shows which tracks each part of it
comes from.

Each track i (n_i frames) contributes its own distribution over the D grid, weighted by w_i
(1 per track, or n_i per detection), and the contributions are summed within length groups:

    contribution_b(D) = sum_{i in b} w_i p_i(D) / sum_i w_i

so that the groups stack to the whole distribution. p_i is the track's own posterior under the
flat prior (unpooled), so a short track spreads wide. The sum describes the tracks with each one's
uncertainty kept; it is not an estimate of the population (it is no likelihood of anything, and a
short track's share of it is mostly the flat prior's shape), which is what `deconvolve_tracks` and
`lognormal_tracks` are for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .data import GridLikelihoods

LENGTH_EDGES = (3, 4, 5, 7, 10, 15, 25, 50)  # lower edges, in frames; the last group is open-ended
WEIGHTS = ("tracks", "detections")


@dataclass(frozen=True)
class LengthComposition:
    """A distribution of D on `u` = ln D split into track-length groups.

    Group j holds tracks with edges[j] <= n_frames < edges[j + 1]; the last group has no upper
    edge. Every contribution is a fraction of all the weight (tracks or detections), so the
    groups and the grid together sum to 1; a row divided by `share()[j]` is that group's own
    distribution.
    """

    u: np.ndarray
    edges: np.ndarray            # (n_groups,) lower edges in frames
    weight: str                  # "tracks" or "detections"
    n_tracks: np.ndarray         # (n_groups,)
    n_detections: np.ndarray     # (n_groups,)
    unpooled: np.ndarray         # (n_groups, len(u)) contributions of the flat-prior posteriors

    def labels(self) -> list[str]:
        """'3', '5-6', '50+': each group's track lengths in frames."""
        hi = list(self.edges[1:] - 1) + [None]
        return [f"{a}+" if b is None else (f"{a}" if a == b else f"{a}-{b}") for a, b in zip(self.edges, hi)]

    def share(self) -> np.ndarray:
        """(n_groups,) each group's fraction of all tracks or detections, as `weight` says."""
        w = self.n_tracks if self.weight == "tracks" else self.n_detections
        return w / w.sum()


def _groups(n_frames: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, n_frames, side="right") - 1


def by_track_length(likelihoods: GridLikelihoods, u: np.ndarray, edges=None,
                    weight: str = "tracks") -> LengthComposition:
    """Split the unpooled distribution of D (the tracks' flat-prior posteriors) by track length.

    `likelihoods` is `analyze_tracks(..., keep_likelihoods=True).likelihoods`, on the grid `u`
    (`options.u_D()`).
    `edges` are the groups' lower edges in frames, increasing, the first at most the shortest
    track; by default `LENGTH_EDGES` up to the longest track. `weight` counts each track once ("tracks") or once per frame ("detections").
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
    one_hot = np.zeros((len(edges), len(lp)))
    one_hot[group, np.arange(len(lp))] = n_frames if weight == "detections" else 1.
    one_hot /= one_hot.sum()

    p = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))  # flat-prior posteriors
    return LengthComposition(
        u=u, edges=edges, weight=weight,
        n_tracks=np.bincount(group, minlength=len(edges)),
        n_detections=np.bincount(group, weights=n_frames, minlength=len(edges)).astype(np.int64),
        unpooled=one_hot @ p)
