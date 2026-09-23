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

### fits -- one row per track and model (posterior_D, posterior_alpha)

| Column | Meaning |
| --- | --- |
| `track_id`, `n_frames` | Track identity and actual observation count |
| `model` | `posterior_D` or `posterior_alpha` |
| `method` | `grid_posterior` or `grid_posterior_marginal_K` |
| `status`, `message` | Numerical/input status and an explanation |
| `uncertainty_method` | `credible_interval` |

Columns only filled on `posterior_D` rows (see [docs/gridpost.md](docs/gridpost.md)):

| Column | Meaning |
| --- | --- |
| `D_post_median_um2_s` | Posterior 0.5 quantile (median) of D under a flat prior in ln D over `[GridPostOptions.D_min_um2_s, D_max_um2_s]` |
| `D_post_lo_um2_s`, `D_post_hi_um2_s` | Posterior quantiles at `(1-level)/2` and `(1+level)/2` (`GridPostOptions.level`, default 0.9: an equal-tailed 90% interval) |

Columns only filled on `posterior_alpha` rows -- D and alpha are independent
per-track measurements (see [docs/gridpost.md](docs/gridpost.md)), not a
joint fit:

| Column | Meaning |
| --- | --- |
| `alpha_post_median` | Posterior 0.5 quantile (median) of the fBm exponent alpha, K marginalized out |
| `alpha_post_lo`, `alpha_post_hi` | Posterior quantiles at `(1-level)/2` and `(1+level)/2` |

A quantile-defined interval is not a multiple of a standard deviation. For a
normal distribution specifically, a 90% equal-tailed interval is +/-1.645 SD,
not +/-1 SD (+/-1 SD covers ~68.3% of a normal, not 90%) -- and these
posteriors are frequently asymmetric or wide enough on short tracks that a
Gaussian sigma wouldn't describe them well anyway.

Uncomputable parameters are null. An `ok` `posterior_D` row whose posterior
is cut by a grid edge (edge weight above 5% of the peak) says so in
`message`: its summary then depends on where that edge is. `status` values:

| Status | Meaning |
| --- | --- |
| `ok` | Finite numerical estimate passed the implemented checks |
| `excluded` | Fewer frames than `GridPostOptions.min_frames`; `posterior_alpha` rows also when `exposure_s > 0` (no blur model) or `GridPostOptions.compute_alpha=False` |
| `invalid_input` | A batch track failed validation, or a localization SD is zero |
