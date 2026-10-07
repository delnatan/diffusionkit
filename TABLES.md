# Result data

This reference covers `diffusionkit.classic.analyze_track`/`analyze_tracks`
and `diffusionkit.gridpost.analyze_track`/`analyze_tracks` -- two independent
entry points, each with its own `fits` table.

## classic: `ClassicAnalysis` -- `fits`, `msd`, `acquisition`, `options`

The settings must accompany saved tables to reproduce the analysis.

### fits -- one row per track and model (brownian, power_law)

| Column | Meaning |
| --- | --- |
| `track_id`, `n_frames` | Track identity and actual observation count |
| `model` | `brownian` or `power_law` |
| `method` | `msd_ols` or `msd_nls` |
| `status`, `message` | Numerical/input status and an explanation |
| `D_um2_s` | Brownian D; null on power-law rows |
| `K_um2_s_alpha`, `alpha` | Generalized coefficient and exponent; null on Brownian rows |
| `n_lags` | Number of lag points actually included |
| `residual_sum_squares_um4` | Unweighted SSE in linear, corrected MSD space; not a goodness-of-model probability |
| `optimizer_status` | SciPy least_squares termination code, or null if not applicable |
| `nfev` | Total nonlinear residual evaluations over three starting points; excludes analytic endpoint checks |
| `localization` | `provided` or explicitly `ignore` |
| `uncertainty_method` | `not_estimated` for both models |

Uncomputable parameters are null. Failed fits can retain numerical estimates
for inspection; check `status` before interpretation. No row is removed for
a negative D, a boundary alpha, or a short track.

| Status | Meaning |
| --- | --- |
| `ok` | Finite numerical estimate passed the implemented checks; precision and model adequacy are not established |
| `nonphysical` | Negative Brownian D, retained without clipping |
| `boundary` | Zero Brownian D or alpha at/near 0 or 2 |
| `unidentified` | Power-law amplitude is zero/negligible or its numerical Jacobian cannot identify alpha |
| `optimizer_failed` | Optimization did not converge or a coefficient was nonfinite |
| `insufficient_data` | Too few lags for the requested fit |
| `excluded` | Fewer frames than `min_frames`, or `exposure_s > 0` (no motion-blur model) |
| `invalid_input` | A batch track failed validation |

`unidentified` is a numerical check, not a complete statistical identifiability
test. Even `ok` alpha fits on short tracks can have large uncertainty and bias.

### msd -- one row per track and selected lag

| Column | Meaning |
| --- | --- |
| `track_id` | Joins to fits; filter fits by model before joining |
| `lag`, `tau_s` | Frame separation and `lag * dt_s` |
| `n_pairs` | Number of overlapping displacement pairs; not an independent sample count |
| `msd_um2` | Mean squared 2D displacement before correction |
| `localization_offset_um2` | Average sum of position-error variances at both endpoints, over both axes |
| `corrected_msd_um2` | Measured MSD minus the offset; negative values are retained |

Only validated, included tracks have MSD rows. Fit rows still record why
other tracks were not analyzed. The maximum lag is capped at `n_frames-1`.

## gridpost: `GridPosteriorAnalysis` -- `fits`, `acquisition`, `options`

### fits -- one row per track (model posterior_D)

| Column | Meaning |
| --- | --- |
| `track_id`, `n_frames` | Track identity and actual observation count |
| `model` | `posterior_D` |
| `method` | `grid_posterior` |
| `status`, `message` | Numerical/input status and an explanation |
| `uncertainty_method` | `credible_interval` |

Posterior columns (see [docs/gridpost.md](docs/gridpost.md)), null unless `status` is `ok`:

| Column | Meaning |
| --- | --- |
| `D_post_mean_um2_s` | Posterior mean E[D], the one point estimate, under a flat prior in ln D over `[GridPostOptions.D_min_um2_s, D_max_um2_s]`; reads high for short tracks by about 1/(n - 2), so do not average it over tracks (use a population model) |
| `D_post_lo_um2_s`, `D_post_hi_um2_s` | Posterior quantiles at `(1-level)/2` and `(1+level)/2` (`GridPostOptions.level`, default 0.9: an equal-tailed 90% interval) |
| `D_post_info_bits` | Information the track gave about D: relative entropy KL(posterior \|\| prior) in bits, prior flat in ln D over the grid. 0 = data left the prior unchanged; each bit is about a halving of the plausible ln D range. Comparable only between runs on the same `[D_min_um2_s, D_max_um2_s]` |
| `D_floor_um2_s` | Localization floor: the D at which a displacement's motion variance equals its localization noise, `<s^2> / (dt - exposure/3)`, `<s^2>` the track's mean per-frame localization variance over both axes (`posterior.localization_floor`). A reference scale to show next to D, not a mobility threshold. Scales with the square of the reported SDs |

A quantile-defined interval is not a multiple of a standard deviation. For a
normal distribution specifically, a 90% equal-tailed interval is +/-1.645 SD,
not +/-1 SD (+/-1 SD covers ~68.3% of a normal, not 90%) -- and these
posteriors are frequently asymmetric or wide enough on short tracks that a
Gaussian sigma wouldn't describe them well anyway.

Uncomputable parameters are null. An `ok` `posterior_D` row whose posterior
is cut by a grid edge (edge weight above 5% of the peak) says so in
`message` and in `D_grid_edge`: "low" (the data only bound D from above:
read the row as upper bounds), "high" (only from below), "both" (a flat
posterior, no information), null otherwise. Its summary then depends on
where that edge is. `status` values:

| Status | Meaning |
| --- | --- |
| `ok` | Finite numerical estimate passed the implemented checks |
| `excluded` | Fewer frames than `GridPostOptions.min_frames` |
| `invalid_input` | A batch track failed validation, or a localization SD is zero |

## Batches: `GridPostBatch`, `ClassicBatch`, `EnsembleMSD`

`gridpost.analyze_experiments` and `classic.analyze_experiments` return the
per-movie tables above, concatenated, with `sample` and `experiment` as the
first two columns; a track is identified by (`experiment`, `track_id`), and
`track_id` repeats across movies. Alongside: `options`, `acquisitions` (by
experiment name) and `samples` (experiment -> sample). `GridPostBatch.likelihoods`
holds the kept per-track log-likelihoods with `sample` and `experiment` name
arrays aligned to its rows.

### EnsembleMSD.curves -- one row per group and lag

| Column | Meaning |
| --- | --- |
| `group` | sample, experiment, or `all`, as `by` says |
| `lag`, `tau_s` | Frame separation and its time; a group has one dt |
| `n_units` | Resampling units (tracks, or experiments) contributing at this lag |
| `n_pairs` | Displacement pairs over all tracks; not an independent sample count |
| `msd_um2` | Weighted mean MSD (`weight`: pairs or tracks), raw |
| `localization_offset_um2` | The same weighted mean of the tracks' supplied-SD offsets; 0 when ignored |
| `msd_se_um2` | Bootstrap SD of `msd_um2` over resamples; null without `n_boot` |

### EnsembleMSD.fit(n_points, offset="fit", level=.9, *, alpha_points=None) -- one row per group and model

The linear fit uses the first `n_points` (required) lags, the log-log fit the
first `alpha_points` (default `n_points`); with `offset="fit"` the log-log fit
subtracts the intercept of the linear fit over its own window. Each row's
`n_points` is its own window. `model` is `linear` (`offset="fit"`, free
intercept; `method` `msd_ols_intercept`) or `brownian` (`offset="provided"`),
and `power_law` (`method` `msd_loglog`, after subtracting the offset). Columns:
`group`, `model`, `method`, `status`, `message`, `n_lags` (points actually used),
`n_points`, `n_units`, and the parameters, each with `_lo`/`_hi`:

| Parameter | Meaning |
| --- | --- |
| `D_um2_s` | Slope / 4 of the linear fit (or the through-origin D with `offset="provided"`) |
| `offset_um2` | The linear fit's intercept b, the localization offset subtracted before the log-log fit |
| `localization_sd_um` | `sqrt(b / 4)`; null for a negative intercept |
| `K_um2_s_alpha`, `alpha` | Log-log fit; alpha is not constrained to [0, 2] (outside it: `nonphysical`) |

`_lo`/`_hi` are the equal-tailed `level` interval over bootstrap refits of the
same window (including the offset estimate), null when `n_boot=0` or fewer
than two refits produced the parameter. `uncertainty_method` is
`cluster_bootstrap_track`, `cluster_bootstrap_experiment`, or `not_estimated`.
The interval does not cover miscalibrated localization SDs or shared drift.
