"""Functional single-track and table MSD workflows; no ensemble or plotting side effects."""
from collections.abc import Callable

import numpy as np
import polars as pl

from ..data import Acquisition
from ..io import validate_table_schema, validated_track_frame
from ..validation import validate_acquisition
from .analysis import compute_msd, fit_windows, validate_options
from .data import ClassicAnalysis, MSDCurve, MSDOptions, TrackAnalysis
from .estimators import empty_fit, fit_anomalous_msd, fit_brownian_msd, fit_loglog_msd


FIT_SCHEMA = {
    "track_id": pl.Int64, "n_frames": pl.Int64, "model": pl.String,
    "method": pl.String, "status": pl.String, "message": pl.String,
    "D_um2_s": pl.Float64, "K_um2_s_alpha": pl.Float64, "alpha": pl.Float64,
    "n_lags": pl.Int64, "residual_sum_squares_um4": pl.Float64,
    "optimizer_status": pl.Int64, "nfev": pl.Int64,
    "localization": pl.String, "uncertainty_method": pl.String,
}
MSD_SCHEMA = {
    "track_id": pl.Int64, "lag": pl.Int64, "tau_s": pl.Float64,
    "n_pairs": pl.Int64, "msd_um2": pl.Float64,
    "localization_offset_um2": pl.Float64, "corrected_msd_um2": pl.Float64,
}


def analyze_track(track: pl.DataFrame, acquisition: Acquisition,
                  options: MSDOptions = MSDOptions()) -> TrackAnalysis:
    """Fit MSD D and alpha for one track's rows: D over the D window, alpha over its own (`MSDOptions`).

    Invalid input raises; short tracks are explicit results. `status="excluded"`
    when `exposure_s > 0`: the MSD estimators assume instantaneous positions,
    no motion-blur model.
    """
    validate_acquisition(acquisition, allow_exposure=True)
    validate_options(options)
    track = validated_track_frame(track, acquisition, require_localization=options.localization == "provided")
    track_id = int(track["track_id"][0])
    n = track.height
    if n < options.min_frames:
        message = f"{n} frames; min_frames={options.min_frames}"
        return TrackAnalysis(track_id, n, None,
                             empty_fit("brownian", "excluded", message),
                             empty_fit("power_law", "excluded", message), acquisition, options)
    if acquisition.exposure_s > 0:
        message = "MSD estimators assume exposure_s=0 (no motion-blur model)"
        return TrackAnalysis(track_id, n, None, empty_fit("brownian", "excluded", message),
                             empty_fit("power_law", "excluded", message), acquisition, options)
    msd = compute_msd(track, acquisition, options)
    n_d, n_alpha = fit_windows(n, options)
    alpha_curve = msd.head(n_alpha)
    alpha = (fit_loglog_msd(alpha_curve) if options.alpha_fit == "loglog"
             else fit_anomalous_msd(alpha_curve, max_nfev=options.max_nfev))
    return TrackAnalysis(track_id, n, msd, fit_brownian_msd(msd.head(n_d)), alpha, acquisition, options)


def analyze_tracks(table: pl.DataFrame, acquisition: Acquisition,
                   options: MSDOptions = MSDOptions(), *,
                   progress: Callable[[int, int], None] | None = None) -> ClassicAnalysis:
    """Two fit rows per input track, including exclusions and invalid tracks.

    Schema/configuration errors raise before work starts. Invalid individual
    tracks get status='invalid_input' and do not prevent other tracks fitting.
    Progress runs in the caller's thread; no global state or worker is created.
    It counts tracks done: the vectorized ones at once, then the rest one by one.

    The same results as `analyze_track` on each track, computed over the whole
    table at once: every track's MSD at lag l comes from one pass over the
    stacked positions, and the D and log-log fits are per-track sums. A track
    that fails a check `validated_track_frame` makes goes through
    `analyze_track` itself, for its message.
    """
    validate_acquisition(acquisition, allow_exposure=True)
    validate_options(options)
    validate_table_schema(table)
    table = table.sort("track_id", "frame")
    ids = table["track_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]) if len(ids) else np.zeros(0, np.int64)
    lengths = np.diff(np.r_[starts, len(ids)])
    n_tracks = len(starts)
    if progress is not None:
        progress(0, n_tracks)
    valid = _valid_tracks(table, acquisition, options, starts, lengths)
    fast = valid & (lengths >= options.min_frames) & (acquisition.exposure_s == 0)
    long = _msd_long(table, acquisition, options, starts[fast], lengths[fast])
    # A nonfinite MSD (overflow) is what `_arrays` refuses: those tracks take the per-track path too.
    finite = np.ones(fast.sum(), bool)
    np.logical_and.at(finite, long["track"], np.isfinite(long["msd"]) & np.isfinite(long["offset"]))
    if not finite.all():
        fast[np.flatnonzero(fast)[~finite]] = False
        keep = finite[long["track"]]
        renumber = np.cumsum(finite) - 1
        long = {k: v[keep] for k, v in long.items()} | {"track": renumber[long["track"][keep]]}
    fit_parts, msd_parts = [], []
    if fast.any():
        fit_parts.append(_fast_fits(ids[starts[fast]], lengths[fast], np.flatnonzero(fast), long, acquisition, options))
        msd_parts.append(pl.DataFrame({
            "track_id": ids[starts[fast]][long["track"]], "lag": long["lag"], "tau_s": long["tau"],
            "n_pairs": long["n_pairs"], "msd_um2": long["msd"], "localization_offset_um2": long["offset"],
            "corrected_msd_um2": long["msd"] - long["offset"], "_track": np.flatnonzero(fast)[long["track"]]},
            schema=MSD_SCHEMA | {"_track": pl.Int64}))
    done = int(fast.sum())
    if progress is not None and done:
        progress(done, n_tracks)
    fit_rows, msd_rows = [], []
    for i in np.flatnonzero(~fast):
        track_id, n = int(ids[starts[i]]), int(lengths[i])
        if valid[i] and n < options.min_frames:
            message = f"{n} frames; min_frames={options.min_frames}"
            result = TrackAnalysis(track_id, n, None, empty_fit("brownian", "excluded", message),
                                   empty_fit("power_law", "excluded", message), acquisition, options)
        elif valid[i] and acquisition.exposure_s > 0:
            message = "MSD estimators assume exposure_s=0 (no motion-blur model)"
            result = TrackAnalysis(track_id, n, None, empty_fit("brownian", "excluded", message),
                                   empty_fit("power_law", "excluded", message), acquisition, options)
        else:
            result = _checked_track(table.slice(int(starts[i]), n), track_id, acquisition, options)
        _track_rows(result, i, options, fit_rows, msd_rows)
        done += 1
        if progress is not None:
            progress(done, n_tracks)
    if fit_rows:
        fit_parts.append(pl.DataFrame(fit_rows, schema=FIT_SCHEMA | _ORDER))
        msd_parts.append(pl.DataFrame(msd_rows, schema=MSD_SCHEMA | _ORDER))
    fits = pl.concat([pl.DataFrame(schema=FIT_SCHEMA | _ORDER), *fit_parts], how="diagonal_relaxed")
    msd = pl.concat([pl.DataFrame(schema=MSD_SCHEMA | {"_track": pl.Int64}), *msd_parts], how="diagonal_relaxed")
    return ClassicAnalysis(fits.sort("_track", "_model").select(list(FIT_SCHEMA)).cast(FIT_SCHEMA),
                           msd.sort("_track", "lag").select(list(MSD_SCHEMA)).cast(MSD_SCHEMA), acquisition, options)


_ORDER = {"_track": pl.Int64, "_model": pl.Int64}  # sort keys: the track's place in the table, brownian before power_law


def _checked_track(group: pl.DataFrame, track_id: int, acquisition: Acquisition, options: MSDOptions) -> TrackAnalysis:
    try:
        return analyze_track(group, acquisition, options)
    except ValueError as exc:
        return TrackAnalysis(track_id, group.height, None,
                             empty_fit("brownian", "invalid_input", str(exc)),
                             empty_fit("power_law", "invalid_input", str(exc)), acquisition, options)


def _track_rows(result: TrackAnalysis, position: int, options: MSDOptions, fit_rows: list, msd_rows: list) -> None:
    for k, fit in enumerate((result.brownian, result.anomalous)):
        fit_rows.append({"track_id": result.track_id, "n_frames": result.n_frames,
                         "model": fit.model, "method": fit.method, "status": fit.status,
                         "message": fit.message, "n_lags": fit.n_lags,
                         "residual_sum_squares_um4": fit.residual_sum_squares_um4,
                         "optimizer_status": fit.optimizer_status, "nfev": fit.nfev,
                         "localization": options.localization,
                         "uncertainty_method": fit.uncertainty_method, **fit.parameters,
                         "_track": int(position), "_model": k})
    curve = result.msd
    if curve is not None:
        for i, lag in enumerate(curve.lag):
            msd_rows.append({"track_id": result.track_id, "lag": int(lag),
                             "tau_s": float(curve.tau_s[i]), "n_pairs": int(curve.n_pairs[i]),
                             "msd_um2": float(curve.msd_um2[i]),
                             "localization_offset_um2": float(curve.localization_offset_um2[i]),
                             "corrected_msd_um2": float(curve.msd_um2[i] - curve.localization_offset_um2[i]),
                             "_track": int(position)})


def _valid_tracks(table: pl.DataFrame, acquisition: Acquisition, options: MSDOptions,
                  starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Tracks that pass every check `validated_track_frame` makes, on a table sorted by track and frame.
    A track it would refuse is never marked valid; one marked invalid is re-checked by it for the message."""
    n_rows = table.height
    track = np.repeat(np.arange(len(starts)), lengths)
    bad = np.zeros(n_rows, bool)
    if "track_length" in table.columns:
        bad |= ~(table["track_length"].to_numpy() == lengths[track])
    frames = table["frame"].to_numpy()
    if "t_s" in table.columns:
        times = table["t_s"].to_numpy()
        expected = frames * acquisition.dt_s
        with np.errstate(invalid="ignore"):
            bad |= ~(np.isfinite(times) & np.isclose(times, expected, rtol=1e-9, atol=1e-12))
    bad |= frames < 0
    step = np.diff(frames.astype(np.int64))
    bad[1:] |= (step != 1) & (track[1:] == track[:-1])
    bad |= ~np.isfinite(table.select("x_um", "y_um").to_numpy()).all(1)
    if options.localization == "provided":
        if "sigma_x_um" not in table.columns:
            return np.zeros(len(starts), bool)
        errors = table.select("sigma_x_um", "sigma_y_um").to_numpy()
        with np.errstate(invalid="ignore"):
            bad |= ~(np.isfinite(errors) & (errors >= 0)).all(1)
    return np.bincount(track, bad, minlength=len(starts)) == 0


def _windows(lengths: np.ndarray, options: MSDOptions) -> tuple[np.ndarray, np.ndarray]:
    """`fit_windows` for each track length."""
    table = {int(n): fit_windows(int(n), options) for n in np.unique(lengths)}
    windows = np.array([table[int(n)] for n in lengths], np.int64).reshape(-1, 2)
    return windows[:, 0], windows[:, 1]


def _msd_long(table: pl.DataFrame, acquisition: Acquisition, options: MSDOptions,
              starts: np.ndarray, lengths: np.ndarray) -> dict[str, np.ndarray]:
    """`compute_msd` of the tracks at `starts` (rows of a sorted table), as one long table sorted by track
    then lag: `track` (index into `starts`), `lag`, `tau`, `n_pairs`, `msd`, `offset`.

    The tracks are stacked longest window first, so the tracks that reach lag l are a prefix of the stack
    and lag l is one pass over that prefix's rows: the work is the sum over tracks of frames x lags.
    """
    n_d, n_alpha = _windows(lengths, options)
    reach = np.maximum(n_d, n_alpha)
    order = np.argsort(-reach, kind="stable")
    sizes, reach_sorted = lengths[order], reach[order]
    ends = np.cumsum(sizes)
    rows = np.repeat(starts[order] - (ends - sizes), sizes) + np.arange(ends[-1] if len(ends) else 0)
    positions = table.select("x_um", "y_um").to_numpy()[rows]
    variances = (np.sum(table.select("sigma_x_um", "sigma_y_um").to_numpy()[rows]**2, axis=1)
                 if options.localization == "provided" else np.zeros(len(rows)))
    stacked = np.repeat(np.arange(len(order)), sizes)
    parts = {k: [] for k in ("track", "lag", "msd", "offset")}
    for lag in range(1, int(reach_sorted[0]) + 1 if len(order) else 1):
        k = int(np.searchsorted(-reach_sorted, -lag, side="right"))  # tracks reaching this lag
        end = int(ends[k - 1])
        first, second = stacked[:end - lag], stacked[lag:end]
        same = first == second
        squared = np.sum((positions[lag:end] - positions[:end - lag])**2, axis=1)
        pairs = sizes[:k] - lag
        parts["track"].append(order[:k])
        parts["lag"].append(np.full(k, lag))
        parts["msd"].append(np.bincount(first[same], squared[same], minlength=k) / pairs)
        parts["offset"].append(np.bincount(first[same], (variances[lag:end] + variances[:end - lag])[same],
                                           minlength=k) / pairs)
    long = {k: np.concatenate(v) if v else np.zeros(0, np.int64 if k in ("track", "lag") else float)
            for k, v in parts.items()}
    sort = np.lexsort((long["lag"], long["track"]))
    long = {k: v[sort] for k, v in long.items()}
    long["tau"] = long["lag"] * acquisition.dt_s
    long["n_pairs"] = lengths[long["track"]] - long["lag"]
    return long


def _fast_fits(track_ids: np.ndarray, lengths: np.ndarray, positions: np.ndarray, long: dict,
               acquisition: Acquisition, options: MSDOptions) -> pl.DataFrame:
    """`fit_brownian_msd` over each track's D window and the alpha fit over its alpha window, as per-track sums
    over `_msd_long`'s table; the bounded NLS (`alpha_fit="nls"`) is still one optimizer run per track."""
    n = len(track_ids)
    n_d, n_alpha = _windows(lengths, options)
    t, lag, tau = long["track"], long["lag"], long["tau"]
    y = long["msd"] - long["offset"]

    def total(values, mask):
        return np.bincount(t[mask], values[mask], minlength=n)

    # Brownian: OLS through the origin in q = tau / tau_1.
    in_d = lag <= n_d[t]
    q = tau / acquisition.dt_s
    amplitude = total(q * y, in_d) / total(q * q, in_d)
    D = amplitude / (4 * acquisition.dt_s)
    rss_d = total((y - amplitude[t] * q)**2, in_d)
    brownian = {"model": "brownian", "method": "msd_ols", "D_um2_s": D, "n_lags": n_d,
                "residual_sum_squares_um4": rss_d, "optimizer_status": None, "nfev": 0,
                "status": np.where(D < 0, "nonphysical", np.where(D == 0, "boundary", "ok")),
                "message": np.where(D < 0, "Negative diffusion estimate",
                                    np.where(D == 0, "Zero diffusion estimate", ""))}
    if options.alpha_fit == "loglog":
        power = _loglog(t, lag, tau, y, n, n_alpha, total)
    else:
        bounds = np.flatnonzero(np.r_[True, t[1:] != t[:-1]]) if len(t) else np.zeros(0, np.int64)
        fits = []
        for i, b in enumerate(bounds):
            s = slice(int(b), int(b + n_alpha[i]))
            curve = MSDCurve(lag[s], tau[s], long["n_pairs"][s], long["msd"][s], long["offset"][s])
            fits.append(fit_anomalous_msd(curve, max_nfev=options.max_nfev))
        power = {"model": "power_law", "method": [f.method for f in fits],
                 "K_um2_s_alpha": [f.parameters["K_um2_s_alpha"] for f in fits],
                 "alpha": [f.parameters["alpha"] for f in fits], "n_lags": [f.n_lags for f in fits],
                 "residual_sum_squares_um4": [f.residual_sum_squares_um4 for f in fits],
                 "optimizer_status": [f.optimizer_status for f in fits], "nfev": [f.nfev for f in fits],
                 "status": [f.status for f in fits], "message": [f.message for f in fits]}
    common = {"track_id": track_ids, "n_frames": lengths, "localization": options.localization,
              "uncertainty_method": "not_estimated", "_track": positions}
    frames = []
    for k, columns in enumerate((brownian, power)):
        data = {**common, **columns, "_model": k}
        frames.append(pl.DataFrame({c: _column(data[c], n) for c in data if c in FIT_SCHEMA or c in _ORDER},
                                   schema={c: (FIT_SCHEMA | _ORDER)[c] for c in data}))
    # The log-log fit's insufficient rows carry NaN for "not fitted"; the table's word for it is null.
    return pl.concat(frames, how="diagonal_relaxed").with_columns(
        pl.col("K_um2_s_alpha", "alpha", "residual_sum_squares_um4").fill_nan(None))


def _column(value, n):
    return [value] * n if value is None or isinstance(value, (str, int, float)) else value


def _loglog(t, lag, tau, y, n, n_alpha, total) -> dict:
    """`fit_loglog_msd` with the curve's own offset, over each track's alpha window, from per-track sums."""
    in_alpha = lag <= n_alpha[t]
    keep = in_alpha & (y > 0)
    count = np.bincount(t[keep], minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        x, ly = np.log(tau), np.log(np.where(keep, y, 1.))
        mx, my = total(x, keep) / count, total(ly, keep) / count
        dx, dy = x - mx[t], ly - my[t]
        alpha = total(dx * dy, keep) / total(dx * dx, keep)
        intercept = my - alpha * mx
        rss = total((ly - alpha[t] * x - intercept[t])**2, keep)
        K = np.exp(intercept) / 4
    ok = count >= 3
    dropped = n_alpha - count
    status, message = [], []
    for i in range(n):
        if not ok[i]:
            status.append("insufficient_data")
            message.append(f"{int(count[i])} lags with positive corrected MSD; at least three are required")
            continue
        m = f"{int(dropped[i])} lag(s) with corrected MSD <= 0 dropped from the log fit" if dropped[i] else ""
        if not 0 <= alpha[i] <= 2:
            status.append("nonphysical")
            message.append("; ".join(s for s in (f"alpha {alpha[i]:.3g} outside [0, 2]", m) if s))
        else:
            status.append("ok")
            message.append(m)
    return {"model": "power_law", "method": np.where(ok, "msd_loglog", "msd_nls"),
            "K_um2_s_alpha": np.where(ok, K, np.nan), "alpha": np.where(ok, alpha, np.nan), "n_lags": count,
            "residual_sum_squares_um4": np.where(ok, rss, np.nan), "optimizer_status": None, "nfev": 0,
            "status": status, "message": message}
