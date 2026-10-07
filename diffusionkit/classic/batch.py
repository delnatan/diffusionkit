"""MSD analysis over a batch of experiments: per-track time-averaged MSDs, and the ensemble-averaged MSD vs lag.

Two different curves, both textbook:

- the **time-averaged MSD** (TA-MSD) is one track's mean squared displacement over time along that
  track, at each lag: what `analyze_tracks` fits per track;
- the **ensemble-averaged MSD** (EA-MSD) averages over tracks, built here from the TA-MSDs:
  `ensemble_msd(batch, by="sample")` averages each sample's, experiment's or everything's.

For a Brownian particle the two agree; when they do not (non-ergodic motion, a mixture of slow and
fast tracks) that is itself the finding. The EA-MSD at lag l is a weighted mean of the tracks'
TA-MSD, with weight w_i = n_pairs_i(l) ("pairs", the standard ensemble estimator: every squared
displacement counts once, so long tracks dominate) or 1 ("tracks": the plain mean of the TA-MSDs).

Building the curve and fitting it are separate steps. `ensemble_msd` resamples tracks (or whole
experiments) once and keeps the resampled curves; `EnsembleMSD.fit(n_points, alpha_points=...)` then fits
D over the first `n_points` lags and alpha over the first `alpha_points`, of the averaged curve and of every
resample, so a new window or offset rule refits without resampling. `n_points` is required: the window is
part of the answer. Overlapping pairs within
a track are correlated, so pairs are never the resampling unit; a track is the unit that can be treated
as independent, and only under the model's own assumptions: the intervals do not cover miscalibrated
localization SDs or drift shared by a movie's tracks. A group must share one frame interval, since its
lags are pooled by index; pool experiments with different dt separately.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..data import Acquisition
from ..experiments import Experiment, label, validate_experiments
from .data import ClassicAnalysis, MSDCurve, MSDFit, MSDOptions
from .estimators import fit_brownian_msd, fit_linear_msd, fit_loglog_msd
from .workflow import analyze_tracks

GROUPINGS = ("sample", "experiment", "all")
WEIGHTS = ("pairs", "tracks")
RESAMPLE = ("track", "experiment")
PARAMETERS = ("D_um2_s", "offset_um2", "localization_sd_um", "K_um2_s_alpha", "alpha")


@dataclass(frozen=True)
class ClassicBatch:
    fits: pl.DataFrame  # `ClassicAnalysis.fits` for every experiment, with leading sample, experiment columns
    msd: pl.DataFrame  # likewise `ClassicAnalysis.msd`
    acquisitions: dict[str, Acquisition]  # by experiment name
    samples: dict[str, str]  # experiment name -> sample name
    options: MSDOptions


@dataclass(frozen=True)
class EnsembleMSD:
    """Ensemble-averaged MSD curves, one group per sample, experiment or everything ("all"), with their resamples.

    `curves` is one row per group and lag: `n_units` (resampling units reaching the lag), `n_pairs`,
    `msd_um2` (the weighted mean, raw), `localization_offset_um2` (the same mean of the supplied SDs'
    offsets; zero when ignored) and `msd_se_um2` (bootstrap SD of the raw mean, null without a bootstrap).
    `fit(n_points, alpha_points=...)` fits them.
    """
    curves: pl.DataFrame
    by: str
    weight: str
    resample: str
    n_boot: int
    means: dict[str, MSDCurve] = field(repr=False, compare=False)  # group -> the averaged curve
    resamples: dict[str, list[MSDCurve]] = field(repr=False, compare=False)  # group -> the bootstrap curves
    n_units: dict[str, int] = field(repr=False, compare=False)  # group -> resampling units (tracks or experiments)

    def curve(self, group: str = "all") -> MSDCurve:
        """The averaged curve of one group, for any estimator in `classic.estimators`."""
        if group not in self.means:
            raise ValueError(f"no group named {group!r}; have {sorted(self.means)}")
        return self.means[group]

    def fit(self, n_points: int, offset: str = "fit", level: float = .9, *,
            alpha_points: int | None = None) -> pl.DataFrame:
        """Linear then log-log fits of each group's first lags: one row per group and model.

        The linear fit (D, and with `offset="fit"` the offset) uses the first `n_points` lags; the log-log fit
        (alpha, K) the first `alpha_points` (default `n_points`). They are separate windows because the two
        fits want different ones: D from the short, best-measured lags (`analysis.window_lags` gives the
        25-40% rule of thumb), alpha from a span of lags wide enough to show curvature in log-log, often the
        whole curve. Either must be in [3, a group's lags]. `offset="fit"` takes the localization offset from
        the linear fit's intercept, over its own window, and subtracts it before the log-log fit; `"provided"`
        takes it from the supplied SDs. Each bootstrap resample is fitted the same way, so the offset estimate
        is inside the interval; `_lo`/`_hi` are its equal-tailed `level` interval, null without a bootstrap or
        when fewer than two resamples gave the parameter. Rows: model `linear` (or `brownian`) with `D_um2_s`,
        `offset_um2`, `localization_sd_um`, and `power_law` with `K_um2_s_alpha`, `alpha`; each row's
        `n_points` is its own window.
        """
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        if alpha_points is None:
            alpha_points = n_points
        q = [(1 - level) / 2, (1 + level) / 2]
        rows = []
        for name, curve in self.means.items():
            for arg, n in (("n_points", n_points), ("alpha_points", alpha_points)):
                if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or not 3 <= n <= len(curve.lag):
                    raise ValueError(f"group {name!r}: {arg} must be an integer in [3, {len(curve.lag)}], got {n!r}")
            span = max(n_points, alpha_points)
            fits = _fit_window(curve, n_points, offset, alpha_points)
            boot = [_fit_window(c, n_points, offset, alpha_points) for c in self.resamples[name]
                    if len(c.lag) >= span and np.array_equal(c.lag[:span], curve.lag[:span])]
            for k, fit in enumerate(fits):
                row = {"group": name, "model": fit.model, "method": fit.method, "status": fit.status,
                       "message": fit.message, "n_lags": fit.n_lags,
                       "n_points": int(alpha_points if fit.model == "power_law" else n_points),
                       "n_units": self.n_units[name],
                       "uncertainty_method": f"cluster_bootstrap_{self.resample}" if boot else "not_estimated",
                       **{p: None for p in PARAMETERS}, **{f"{p}_{e}": None for p in PARAMETERS for e in ("lo", "hi")}}
                row.update(fit.parameters)
                for p in fit.parameters:
                    draws = np.array([b[k].parameters[p] for b in boot if b[k].parameters.get(p) is not None], float)
                    if len(draws) >= 2:
                        row[f"{p}_lo"], row[f"{p}_hi"] = (float(x) for x in np.quantile(draws, q))
                rows.append(row)
        return pl.DataFrame(rows, schema=FIT_SCHEMA)


def _fit_window(curve: MSDCurve, n_points: int, offset: str, alpha_points: int | None = None) -> tuple[MSDFit, MSDFit]:
    """The two fits `EnsembleMSD.fit` reports: D over the curve's first `n_points` lags, then alpha and K over
    its first `alpha_points` (default `n_points`) after the offset.

    offset="fit": the linear fit's intercept is the offset (clipped at 0 when negative, which the linear row flags).
    offset="provided": the curve's own offset from supplied SDs, and D through the origin.
    """
    alpha_curve = curve.head(n_points if alpha_points is None else alpha_points)
    curve = curve.head(n_points)
    if offset == "provided":
        return fit_brownian_msd(curve), fit_loglog_msd(alpha_curve)
    if offset != "fit":
        raise ValueError(f"offset must be 'fit' or 'provided', got {offset!r}")
    linear = fit_linear_msd(curve)
    b = linear.parameters["offset_um2"]
    return linear, fit_loglog_msd(alpha_curve, 0. if b is None else max(b, 0.))


def analyze_experiments(experiments: Sequence[Experiment], options: MSDOptions = MSDOptions(), *,
                        progress: Callable[[int, int], None] | None = None) -> ClassicBatch:
    """`classic.analyze_tracks` for each experiment, in labelled tables. `progress` counts tracks over the batch."""
    experiments = validate_experiments(experiments)
    totals = [e.tracks["track_id"].n_unique() for e in experiments]
    total, before = sum(totals), 0
    fits, msds = [], []
    for e, n in zip(experiments, totals):
        inner = None if progress is None else (lambda done, _n, b=before: progress(b + done, total))
        result: ClassicAnalysis = analyze_tracks(e.tracks, e.acquisition, options, progress=inner)
        before += n
        fits.append(label(result.fits, e))
        msds.append(label(result.msd, e))
    return ClassicBatch(pl.concat(fits), pl.concat(msds), {e.name: e.acquisition for e in experiments},
                        {e.name: e.sample_name for e in experiments}, options)


@dataclass
class _Group:
    """Per-unit sums at each lag: N = sum of weights, S = weighted MSD, O = weighted offset."""
    lag: np.ndarray
    tau: np.ndarray
    N: np.ndarray  # (n_units, n_lags)
    S: np.ndarray
    O: np.ndarray
    pairs: np.ndarray  # (n_lags,) pairs over all units
    units: np.ndarray  # (n_lags,) units with any weight


def _group_sums(rows: pl.DataFrame, weight: str, resample: str, name) -> _Group:
    lags = np.sort(rows["lag"].unique().to_numpy())
    tau = rows.group_by("lag").agg(pl.col("tau_s").unique().alias("tau")).sort("lag")
    if tau["tau"].list.len().max() > 1:
        raise ValueError(f"group {name!r} mixes frame intervals; pool experiments with different dt separately")
    tau = tau["tau"].list.first().to_numpy()
    unit = (rows["experiment"] if resample == "experiment" else
            rows["experiment"] + "/" + rows["track_id"].cast(pl.String))
    codes, index = np.unique(unit.to_numpy(), return_inverse=True)
    column = np.searchsorted(lags, rows["lag"].to_numpy())
    n_pairs = rows["n_pairs"].to_numpy().astype(float)
    w = n_pairs if weight == "pairs" else np.ones_like(n_pairs)
    shape = (len(codes), len(lags))
    out = {}
    for key, value in (("N", w), ("S", w * rows["msd_um2"].to_numpy()),
                       ("O", w * rows["localization_offset_um2"].to_numpy())):
        out[key] = np.zeros(shape)
        np.add.at(out[key], (index, column), value)
    pairs = np.zeros(len(lags))
    np.add.at(pairs, column, n_pairs)
    seen = np.zeros(shape, bool)
    seen[index, column] = True
    return _Group(lags, tau, out["N"], out["S"], out["O"], pairs, seen.sum(0))


def _curve(g: _Group, counts: np.ndarray | None = None) -> MSDCurve | None:
    """The ensemble MSDCurve over the lags that have weight, for the units resampled `counts` times."""
    N, S, O = ((g.N.sum(0), g.S.sum(0), g.O.sum(0)) if counts is None else (counts @ g.N, counts @ g.S, counts @ g.O))
    keep = N > 0
    if not keep.any():
        return None
    return MSDCurve(g.lag[keep], g.tau[keep], g.pairs[keep].astype(np.int64), S[keep] / N[keep], O[keep] / N[keep])




def ensemble_msd(batch: ClassicBatch, by: str = "sample", weight: str = "pairs", resample: str = "track",
                 n_boot: int = 200, rng: np.random.Generator | None = None) -> EnsembleMSD:
    """The ensemble-averaged MSD of each sample, each experiment, or all tracks, with `n_boot` bootstrap resamples.

    The curve spans every lag the batch's per-track windows (`MSDOptions.max_lag` or `lag_fraction`)
    reached, so its high lags rest on fewer tracks (`n_units`). Tracks are resampled with replacement
    (`resample="track"`) or whole experiments (`"experiment"`, the only choice that sees replicate-to-
    replicate variation; it needs several replicates per sample). `n_boot=0` skips resampling. A group
    with no MSD rows (every track excluded, e.g. `exposure_s > 0`) is left out; no group at all raises.
    Fit with `.fit(n_points)`.
    """
    for name, value, allowed in (("by", by, GROUPINGS), ("weight", weight, WEIGHTS), ("resample", resample, RESAMPLE)):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    if n_boot < 0:
        raise ValueError("n_boot must be >= 0")
    rng = np.random.default_rng(0) if rng is None else rng
    names = ["all"] if by == "all" else list(dict.fromkeys(batch.fits[by].to_list()))
    curve_rows, means, boots, units = [], {}, {}, {}
    for name in names:
        rows = batch.msd if by == "all" else batch.msd.filter(pl.col(by) == name)
        if rows.height == 0:
            continue
        g = _group_sums(rows, weight, resample, name)
        curve, n_units = _curve(g), len(g.N)
        resampled = []
        if n_boot and n_units > 1:
            for _ in range(n_boot):
                c = _curve(g, rng.multinomial(n_units, np.full(n_units, 1 / n_units)).astype(float))
                if c is not None:
                    resampled.append(c)
        se = np.full(len(curve.lag), np.nan)
        for i, lag in enumerate(curve.lag):
            values = [c.msd_um2[c.lag == lag][0] for c in resampled if lag in c.lag]
            if len(values) > 1:
                se[i] = np.std(values, ddof=1)
            curve_rows.append({"group": name, "lag": int(lag), "tau_s": float(curve.tau_s[i]),
                               "n_units": int(g.units[g.lag == lag][0]), "n_pairs": int(curve.n_pairs[i]),
                               "msd_um2": float(curve.msd_um2[i]),
                               "localization_offset_um2": float(curve.localization_offset_um2[i]),
                               "msd_se_um2": None if np.isnan(se[i]) else float(se[i])})
        means[name], boots[name], units[name] = curve, resampled, n_units
    if not means:
        raise ValueError("no MSD rows in any group: every track was excluded (exposure_s > 0 has no MSD fit)")
    return EnsembleMSD(pl.DataFrame(curve_rows, schema=CURVE_SCHEMA), by, weight, resample, n_boot, means, boots, units)


CURVE_SCHEMA = {"group": pl.String, "lag": pl.Int64, "tau_s": pl.Float64, "n_units": pl.Int64, "n_pairs": pl.Int64,
                "msd_um2": pl.Float64, "localization_offset_um2": pl.Float64, "msd_se_um2": pl.Float64}
FIT_SCHEMA = {"group": pl.String, "model": pl.String, "method": pl.String, "status": pl.String, "message": pl.String,
              "n_lags": pl.Int64, "n_points": pl.Int64, "n_units": pl.Int64, "uncertainty_method": pl.String,
              **{c: pl.Float64 for p in PARAMETERS for c in (p, f"{p}_lo", f"{p}_hi")}}
