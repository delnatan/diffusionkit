"""The D posterior over a batch of experiments: replicates of a sample, several samples side by side.

Per-track posteriors stay the primary output; nothing here averages them away. Each experiment is
analyzed with its own `Acquisition` (dt and exposure may differ between movies: every track's
likelihood lives on the one D grid in `GridPostOptions`), and what is combined is the kept
per-track log-likelihood rows. Stacking those rows over experiments is what makes an ensemble:
`GridPostBatch.select` hands any subset (a sample, a replicate) to the same functions a single
movie uses (`shared_D_tracks`, `lognormal_tracks`, `deconvolve_tracks`, `by_track_length`).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..data import Acquisition
from ..experiments import Experiment, label, label_as, select_rows, validate_experiments
from .data import GridDistribution, GridLikelihoods, GridPosteriorAnalysis, GridPostOptions
from .deconvolve import deconvolve_tracks
from .lognormal import SharedD, lognormal_tracks, shared_D_tracks
from .workflow import analyze_tracks

GROUPINGS = ("sample", "experiment", "all")
MODELS = {"deconvolve": deconvolve_tracks, "lognormal": lognormal_tracks}  # partial pooling, by shape of g


@dataclass(frozen=True)
class BatchLikelihoods(GridLikelihoods):
    """`GridLikelihoods` stacked over experiments, so `track_ids` repeat across them: a row's identity is
    (experiment, track_id). `sample` and `experiment` are (n,) name arrays aligned with the rows."""
    sample: np.ndarray
    experiment: np.ndarray


@dataclass(frozen=True)
class GridPostBatch:
    fits: pl.DataFrame  # `GridPosteriorAnalysis.fits` for every experiment, with leading sample, experiment columns
    acquisitions: dict[str, Acquisition]  # by experiment name
    samples: dict[str, str]  # experiment name -> sample name
    options: GridPostOptions
    likelihoods: BatchLikelihoods | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_analyses(cls, analyses: Mapping[str, GridPosteriorAnalysis],
                      samples: Mapping[str, str] | None = None) -> GridPostBatch:
        """A batch from per-experiment analyses however they were obtained: fresh from `analyze_tracks`, or
        restored from disk. `analyses` maps experiment name to its `GridPosteriorAnalysis` (with kept
        likelihoods, on one `GridPostOptions` for all); `samples` maps experiment name to sample name
        (default: the experiment's own name). Each analysis needs an `acquisition`.
        """
        if not analyses:
            raise ValueError("no analyses given")
        first = next(iter(analyses.values()))
        samples = {name: (samples or {}).get(name, name) for name in analyses}
        for name, a in analyses.items():
            if a.likelihoods is None:
                raise ValueError(f"experiment {name!r} kept no likelihoods: analyze_tracks(..., keep_likelihoods=True)")
            if a.options != first.options:
                raise ValueError(f"experiment {name!r} was analyzed with different GridPostOptions; "
                                 "combining needs one D grid")
            if a.acquisition is None:
                raise ValueError(f"experiment {name!r} has no acquisition")
        parts = [a.likelihoods for a in analyses.values()]
        fits = pl.concat([label_as(a.fits, n, samples[n]) for n, a in analyses.items()], how="diagonal_relaxed")
        names = np.concatenate([np.full(len(p.track_ids), n, dtype=object) for n, p in zip(analyses, parts)]).astype(str)
        sample_of = np.concatenate([np.full(len(p.track_ids), samples[n], dtype=object)
                                    for n, p in zip(analyses, parts)]).astype(str)
        likelihoods = BatchLikelihoods(np.concatenate([p.track_ids for p in parts]),
                                       np.concatenate([p.n_frames for p in parts]),
                                       np.vstack([p.loglik_D for p in parts]), sample_of, names)
        return cls(fits, {n: a.acquisition for n, a in analyses.items()}, samples, first.options, likelihoods)

    def select(self, sample: str | None = None, experiment: str | None = None) -> GridPosteriorAnalysis:
        """One sample's, one experiment's, or (no arguments) every track's analysis, as `analyze_tracks` returns it.

        Its `acquisition` is None when the selection spans different acquisitions. Anything that takes a
        `GridPosteriorAnalysis` -- `lognormal_tracks`, `by_track_length(result.likelihoods, ...)` -- takes it.
        """
        if self.likelihoods is None:
            raise ValueError("no per-track likelihoods were kept: analyze_experiments(..., keep_likelihoods=True)")
        fits = select_rows(self.fits, sample, experiment)
        rows = self.likelihoods
        keep = np.ones(len(rows.track_ids), bool)
        if sample is not None:
            keep &= rows.sample == sample
        if experiment is not None:
            keep &= rows.experiment == experiment
        acquisitions = {self.acquisitions[n] for n in fits["experiment"].unique().to_list()}
        return GridPosteriorAnalysis(
            fits, acquisitions.pop() if len(acquisitions) == 1 else None, self.options,
            GridLikelihoods(rows.track_ids[keep], rows.n_frames[keep], rows.loglik_D[keep]))

    def groups(self, by: str = "sample") -> dict[str | None, GridPosteriorAnalysis]:
        """`select` for each sample, each experiment, or ("all") everything, keyed by name (None for "all")."""
        if by not in GROUPINGS:
            raise ValueError(f"by must be one of {GROUPINGS}, got {by!r}")
        if by == "all":
            return {None: self.select()}
        names = list(dict.fromkeys(self.fits[by].to_list()))  # input order
        return {n: self.select(**{by: n}) for n in names}

    def populations(self, by: str = "sample", n_samples: int = 1000, rng: np.random.Generator | None = None,
                    model: str = "deconvolve") -> dict[str | None, GridDistribution]:
        """The distribution of D (partial pooling) for each sample, each experiment (replicates side by
        side) or all of it together: `deconvolve_tracks` (`model="deconvolve"`, any shape) or
        `lognormal_tracks` (`model="lognormal"`).

        A sample's replicates enter one fit, their tracks' log-likelihoods added, so a sample's
        population weighs every track equally, not every movie. For a population per replicate,
        `by="experiment"`.
        """
        if model not in MODELS:
            raise ValueError(f"model must be one of {tuple(MODELS)}, got {model!r}")
        out = {}
        for name, analysis in self.groups(by).items():
            try:
                out[name] = MODELS[model](analysis, n_samples=n_samples, rng=rng)
            except ValueError as exc:
                raise ValueError(f"{by} {name!r}: {exc}") from exc
        return out

    def shared_D(self, by: str = "sample") -> dict[str | None, SharedD]:
        """`shared_D_tracks` (complete pooling: one D for all of a group's tracks) for each sample, each
        experiment, or all of it together."""
        out = {}
        for name, analysis in self.groups(by).items():
            try:
                out[name] = shared_D_tracks(analysis)
            except ValueError as exc:
                raise ValueError(f"{by} {name!r}: {exc}") from exc
        return out


def analyze_experiments(experiments: Sequence[Experiment], options: GridPostOptions = GridPostOptions(), *,
                        keep_likelihoods: bool = True,
                        progress: Callable[[int, int], None] | None = None,
                        map_fn: Callable[..., Iterable] = map) -> GridPostBatch:
    """`gridpost.analyze_tracks` for each experiment, one labelled `fits` table and one stack of likelihoods.

    `progress(done, total)` counts tracks over the whole batch, in the caller's thread. `map_fn` is
    `analyze_tracks`'s: a pool's `map` fits tracks concurrently, with the pool owned by the caller
    (one pool serves every experiment). A bad table or acquisition raises before any experiment runs;
    a bad track is an `invalid_input` row as in a single movie.
    """
    experiments = validate_experiments(experiments)
    totals = [e.tracks["track_id"].n_unique() for e in experiments]
    total, before = sum(totals), 0
    analyses = {}
    for e, n in zip(experiments, totals):
        inner = None if progress is None else (lambda done, _n, b=before: progress(b + done, total))
        analyses[e.name] = analyze_tracks(e.tracks, e.acquisition, options, progress=inner,
                                          keep_likelihoods=keep_likelihoods, map_fn=map_fn)
        before += n
    samples = {e.name: e.sample_name for e in experiments}
    if keep_likelihoods:
        return GridPostBatch.from_analyses(analyses, samples)
    fits = pl.concat([label(analyses[e.name].fits, e) for e in experiments])
    return GridPostBatch(fits, {e.name: e.acquisition for e in experiments}, samples, options)


def cdf_distance(a: GridDistribution, b: GridDistribution) -> np.ndarray:
    """(n_samples,) draws of the W1 distance in ln D between two populations: the area between their CDFs.

    The two sets of draws are independent, so they are paired in order. Two draws of the very same
    population are still apart (Monte Carlo and posterior spread), so read a sample-versus-sample
    distance against replicate-versus-replicate distances from the same data, not against zero.
    """
    if a.u.shape != b.u.shape or not np.allclose(a.u, b.u):
        raise ValueError("the populations are on different grids")
    n = min(len(a.samples), len(b.samples))
    gap = np.cumsum(a.samples[:n] - b.samples[:n], axis=1)
    return np.abs(gap).sum(axis=1) * (a.u[1] - a.u[0])
