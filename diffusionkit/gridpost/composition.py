"""Where along D the tracks of each length sit: the distribution of D split by track length.

Track length is observed, not inferred, and it is tied to D: a fast particle crosses the focal
depth in a few frames, so it makes short tracks, while a slow one stays in focus and makes one
long track. Splitting a distribution of D by track length shows which tracks each part of it
comes from.

Each track i (n_i frames) contributes its own distribution over the D grid, weighted by w_i
(1 per track, or n_i per detection), and the contributions are summed within length groups:

    contribution_b(D) = sum_{i in b} w_i p_i(D) / sum_i w_i

so that the groups stack to the whole distribution. Two choices of p_i:

- **unpooled:** each track's own posterior under the flat prior (no pooling). Their sum
  describes the tracks with each one's uncertainty kept, so a short track spreads wide; it is
  not an estimate of the population (it is no likelihood of anything, and a short track's
  share of it is mostly the flat prior's shape).
- **partially pooled:** each track's posterior with a fitted population g (`deconvolve_tracks`,
  `lognormal_tracks`) as its prior, p_i(D | g) ~ L_i(D) g(D), once per draw of g, so the groups
  carry the population's uncertainty: each track borrows the population's shape. Per track, the
  groups stack to about g; per detection, to the composition of the detections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .data import GridDistribution, GridLikelihoods

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
    partially_pooled: np.ndarray | None  # (n_draws, n_groups, len(u)) under draws of g; None without a population

    def labels(self) -> list[str]:
        """'3', '5-6', '50+': each group's track lengths in frames."""
        hi = list(self.edges[1:] - 1) + [None]
        return [f"{a}+" if b is None else (f"{a}" if a == b else f"{a}-{b}") for a, b in zip(self.edges, hi)]

    def share(self) -> np.ndarray:
        """(n_groups,) each group's fraction of all tracks or detections, as `weight` says."""
        w = self.n_tracks if self.weight == "tracks" else self.n_detections
        return w / w.sum()

    def band(self, level: float = .68) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise (lower, upper) equal-tailed band of the partially pooled contributions, (n_groups, len(u)) each."""
        if self.partially_pooled is None:
            raise ValueError("no population was given, so there is nothing to band")
        q = (1 - level) / 2
        lo, hi = np.quantile(self.partially_pooled, [q, 1 - q], axis=0)
        return lo, hi


def _groups(n_frames: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, n_frames, side="right") - 1


def by_track_length(likelihoods: GridLikelihoods, u: np.ndarray, population: GridDistribution | None = None,
                    edges=None, weight: str = "tracks", n_draws: int = 200) -> LengthComposition:
    """Split the unpooled distribution of D (and, given `population`, the partially pooled one) by track length.

    `likelihoods` is `analyze_tracks(..., keep_likelihoods=True).likelihoods`, on the grid `u`
    (`options.u_D()`); `population` is `deconvolve_tracks` or `lognormal_tracks` on the same
    analysis (any `GridDistribution` on `u` will do).
    `edges` are the groups' lower edges in frames, increasing, the first at most the shortest
    track; by default `LENGTH_EDGES` up to the longest track. `weight` counts each track once ("tracks") or once per frame ("detections").
    `n_draws` evenly spaced draws of the population are used for the partially pooled split.
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
    partial = None
    if population is not None:
        if population.u.shape != u.shape or not np.allclose(population.u, u):
            raise ValueError("the population's grid is not u")
        draws = population.samples[np.linspace(0, len(population.samples) - 1, min(n_draws, len(population.samples)))
                                   .round().astype(int)]
        partial = np.empty((len(draws), len(edges), lp.shape[1]))
        for k, g in enumerate(draws):
            post = lp + np.log(np.maximum(g, 1e-300))
            partial[k] = one_hot @ np.exp(post - logsumexp(post, axis=1, keepdims=True))
    return LengthComposition(
        u=u, edges=edges, weight=weight,
        n_tracks=np.bincount(group, minlength=len(edges)),
        n_detections=np.bincount(group, weights=n_frames, minlength=len(edges)).astype(np.int64),
        unpooled=one_hot @ p, partially_pooled=partial)
