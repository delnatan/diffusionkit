"""The per-track joint (alpha, ln D) posterior as data: collect, write, read and pool it.

Each "ok" track's joint posterior (`posterior_alpha.log_joint_posterior`) lives on
`GridPostOptions.alphas()` x `.u_joint_D()` cells (ln of the apparent D); its alpha=1
row is the D posterior binned to cells and its sum over D is the alpha posterior. It is
what a population's (alpha, D) distribution is read from (`deconvolve.deconvolve_joint`),
alone or pooled over samples (`pool_joint_posteriors`), and averaging `exp(log_post)` over
tracks gives the raw pooled posterior, before deconvolution.

On disk (`write_joint_posteriors`) it is one parquet row per track: `sample`, `track_id`,
`n_frames`, `log_post_peak` (the track's largest cell, a normalized log posterior) and
`log_post_q`, an (n_alpha, n_D) UInt16 array of each cell's distance below that peak in
units of `QUANTUM_NATS`, saturating at 65535 (65.5 nats, a relative weight of ~4e-29).
Rounding moves a cell by at most half a quantum, 0.0005 nats, and reading renormalizes each
track (a shift common to its cells, under 0.0002 nats on GEM tracks). The grid, the
acquisition and the format version are in the file's key-value metadata under
`METADATA_KEY`, so a file reads back with its own grid. 2218 tracks on the default 39 x 50
cells take ~8 MB.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import logsumexp

from ..data import Acquisition
from .data import GridPosteriorAnalysis, GridPostOptions

FORMAT = "diffusionkit.joint_posterior/1"
METADATA_KEY = "diffusionkit.joint_posterior"
QUANTUM_NATS = 1e-3
_Q_MAX = np.iinfo(np.uint16).max
# Fields that define the cells; pooled parts must agree on them.
_GRID_FIELDS = ("D_min_um2_s", "D_max_um2_s", "n_D", "joint_D_bin", "alpha_min", "alpha_max", "n_alpha")
# Pooled parts' frame intervals and exposures must agree to this relative tolerance: the
# apparent D and the blur model are defined at each movie's own dt.
ACQUISITION_RTOL = 1e-3


@dataclass(frozen=True)
class JointPosteriors:
    """Per-track joint log posteriors, one row per track, possibly from several samples."""

    sample: np.ndarray    # (n,) str, which sample (movie) each track came from
    track_id: np.ndarray  # (n,) int64, unique within a sample
    n_frames: np.ndarray  # (n,) int64
    log_post: np.ndarray  # (n, n_alpha, n_D) normalized log posterior of each cell
    acquisition: Acquisition
    options: GridPostOptions

    @property
    def alphas(self) -> np.ndarray:
        return self.options.alphas()

    @property
    def u(self) -> np.ndarray:
        """ln of the apparent D (um^2/s) at the cell centres."""
        return self.options.u_joint_D()


def joint_posteriors(analysis: GridPosteriorAnalysis, sample: str = "") -> JointPosteriors:
    """The joint posteriors `analyze_tracks(..., keep_posteriors=True)` kept, labelled `sample`."""
    post = analysis.posteriors
    if post is None:
        raise ValueError("the analysis kept no posteriors: run analyze_tracks(..., keep_posteriors=True)")
    frames = dict(analysis.fits.filter(pl.col("model") == "posterior_alpha")
                  .select("track_id", "n_frames").iter_rows())
    ids = np.asarray(post.alpha_track_ids, dtype=np.int64)
    return JointPosteriors(np.full(len(ids), sample, dtype=object), ids,
                           np.array([frames[i] for i in ids], dtype=np.int64),
                           np.asarray(post.log_post_joint, dtype=float), analysis.acquisition, analysis.options)


def write_joint_posteriors(path: str | Path, joint: JointPosteriors) -> None:
    """Write `joint` to a parquet file (format in the module docstring)."""
    lp = joint.log_post
    peak = lp.max(axis=(1, 2)) if len(lp) else np.zeros(0)
    q = np.minimum(np.rint((peak[:, None, None] - lp) / QUANTUM_NATS), _Q_MAX).astype(np.uint16)
    shape = lp.shape[1:]
    table = pl.DataFrame({
        "sample": pl.Series(joint.sample.astype(str), dtype=pl.String),
        "track_id": pl.Series(joint.track_id, dtype=pl.Int64),
        "n_frames": pl.Series(joint.n_frames, dtype=pl.Int64),
        "log_post_peak": pl.Series(peak, dtype=pl.Float64),
        "log_post_q": pl.Series(q, dtype=pl.Array(pl.UInt16, shape)),
    })
    try:
        package_version = version("diffusionkit")
    except PackageNotFoundError:
        package_version = "unknown"
    meta = {"format": FORMAT, "quantum_nats": QUANTUM_NATS, "scale": "ln apparent D (um^2/s) at cell centres",
            "acquisition": {"dt_s": joint.acquisition.dt_s, "exposure_s": joint.acquisition.exposure_s},
            "options": asdict(joint.options), "diffusionkit": package_version}
    table.write_parquet(path, compression="zstd", compression_level=10, metadata={METADATA_KEY: json.dumps(meta)})


def read_joint_posteriors(path: str | Path) -> JointPosteriors:
    """Read a `write_joint_posteriors` file; each track's log posterior is renormalized after decoding."""
    raw = pl.read_parquet_metadata(path).get(METADATA_KEY)
    if raw is None:
        raise ValueError(f"{path} has no {METADATA_KEY!r} metadata: not a joint posterior file")
    meta = json.loads(raw)
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path} is format {meta.get('format')!r}; this diffusionkit reads {FORMAT!r}")
    options = GridPostOptions(**meta["options"])
    table = pl.read_parquet(path)
    shape = (options.n_alpha, len(options.u_joint_D()))
    q = table["log_post_q"].to_numpy().reshape(-1, *shape)
    lp = table["log_post_peak"].to_numpy()[:, None, None] - meta["quantum_nats"] * q.astype(float)
    lp = lp - logsumexp(lp, axis=(1, 2), keepdims=True) if len(lp) else lp
    return JointPosteriors(table["sample"].to_numpy().astype(object), table["track_id"].to_numpy(),
                           table["n_frames"].to_numpy(), lp, Acquisition(**meta["acquisition"]), options)


def pool_joint_posteriors(parts: Sequence[JointPosteriors]) -> JointPosteriors:
    """Concatenate samples' joint posteriors for a pooled read.

    The parts must share the cells (`_GRID_FIELDS`) and agree on dt and exposure to
    `ACQUISITION_RTOL`; the result keeps the first part's acquisition and options. A
    (sample, track_id) pair must not repeat: label each part's sample distinctly.
    """
    if not parts:
        raise ValueError("nothing to pool")
    first = parts[0]
    for i, part in enumerate(parts[1:], 1):
        for name in _GRID_FIELDS:
            if getattr(part.options, name) != getattr(first.options, name):
                raise ValueError(f"part {i} has {name}={getattr(part.options, name)}, "
                                 f"part 0 has {getattr(first.options, name)}: the cells differ")
        for name in ("dt_s", "exposure_s"):
            a, b = getattr(part.acquisition, name), getattr(first.acquisition, name)
            if not np.isclose(a, b, rtol=ACQUISITION_RTOL, atol=0):
                raise ValueError(f"part {i} has {name}={a}, part 0 has {b}: the apparent D and blur differ")
    pooled = JointPosteriors(np.concatenate([p.sample for p in parts]),
                             np.concatenate([p.track_id for p in parts]),
                             np.concatenate([p.n_frames for p in parts]),
                             np.concatenate([p.log_post for p in parts]), first.acquisition, first.options)
    keys = list(zip(pooled.sample, pooled.track_id))
    if len(set(keys)) != len(keys):
        raise ValueError("a (sample, track_id) pair repeats: give each part its own sample label")
    return pooled
