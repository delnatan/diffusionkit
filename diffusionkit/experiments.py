"""The unit a batch is made of: one movie's tracks, its acquisition, and the sample it belongs to.

A batch is a list of `Experiment`s. Each keeps its own `track_id` space (ids are only unique within a
movie) and its own `Acquisition`, so nothing is renumbered or merged on the way in; results carry
`sample`, `experiment` and `track_id` columns instead. `sample` is what replicates share (a genotype, a
condition); it defaults to the experiment's name, i.e. no replicates.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from .data import Acquisition
from .io import validate_table_schema
from .validation import validate_acquisition

KEY = ("sample", "experiment", "track_id")


@dataclass(frozen=True)
class Experiment:
    """One movie's tracks (a physical-unit table, see README) and how it was acquired."""
    name: str
    tracks: pl.DataFrame
    acquisition: Acquisition
    sample: str | None = None

    @property
    def sample_name(self) -> str:
        return self.name if self.sample is None else self.sample


def validate_experiments(experiments: Sequence[Experiment]) -> list[Experiment]:
    """Names must be unique and every table and acquisition valid, before any work starts."""
    experiments = list(experiments)
    if not experiments:
        raise ValueError("no experiments given")
    names = [e.name for e in experiments]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"experiment names must be unique, repeated: {dup}")
    for e in experiments:
        validate_acquisition(e.acquisition, allow_exposure=True)
        validate_table_schema(e.tracks)
    return experiments


def label(table: pl.DataFrame, experiment: Experiment) -> pl.DataFrame:
    """`table` with leading `sample` and `experiment` columns for `experiment`."""
    return table.select(pl.lit(experiment.sample_name).alias("sample"),
                        pl.lit(experiment.name).alias("experiment"), pl.all())


def select_rows(table: pl.DataFrame, sample: str | None = None, experiment: str | None = None) -> pl.DataFrame:
    """Rows of a labelled table for one sample and/or experiment; a name that matches nothing raises."""
    for column, value in (("sample", sample), ("experiment", experiment)):
        if value is not None:
            if value not in table[column].unique().to_list():
                raise ValueError(f"no {column} named {value!r}; have {sorted(table[column].unique().to_list())}")
            table = table.filter(pl.col(column) == value)
    return table
