# Results tables -- column reference

Companion to `README.md`. Every column written by every script in
`scripts/`, and what it means. Terminology follows the README glossary.

Both pipelines write `track_id` and `track_length` with the same meaning
(track ID from the input CSV; number of localizations in the track), so
their per-track tables always join cleanly on `track_id`. Column names
otherwise follow one convention throughout: a `_um`/`_um2_s`/`_um2_s_alpha`
suffix marks physical units; no suffix means dimensionless (`alpha`) or
log10 units (`log10_*`). Quantities that only exist in one framework (e.g.
the Bayesian posterior/Laplace interval columns, `r2_*` goodness-of-fit from
the MSD fit) are kept under their own names rather than forced into a shared
column -- see the method notes above for why they aren't directly
comparable.

### `classic/per_track_msd_fits.csv` (classic, `run_msd_analysis.py`)

One row per track. `msd.py`'s TAMSD passed through `fitting.fit_all_tracks`.

| Column | Meaning |
| --- | --- |
| `track_id` | Track ID. |
| `track_length` | Number of localizations in the track. |
| `n_points_used` | Number of lags used in the normal-diffusion (and uncorrected anomalous) fit. |
| `at_min_points` | True if `n_points_used` hit the `min_points` floor rather than the fractional rule. |
| `D_um2_s`, `D_stderr_um2_s` | Diffusion coefficient from the linear `MSD=4*D*tau+b` fit, weighted-least-squares standard error. |
| `intercept_um2`, `intercept_stderr_um2` | Fitted offset `b` of the same linear fit and its standard error. |
| `r2_normal` | R^2 of the linear (normal-diffusion) fit. |
| `D_negative` | True if the fitted `D_um2_s` is negative (unphysical -- kept, not dropped). |
| `intercept_negative` | True if the fitted intercept is negative (unphysical for R=0 static noise -- kept, not dropped). |
| `alpha`, `alpha_stderr` | Anomalous exponent from the log-log `MSD=4*D_alpha*tau^alpha` fit (OLS), and its standard error. |
| `n_points_used_alpha` | Lags actually used in that fit (may be fewer than `n_points_used` if some points were non-positive). |
| `D_alpha_um2_s_alpha` | Generalized diffusion coefficient from the same log-log fit. |
| `r2_anomalous` | R^2 of the log-log fit. |
| `alpha_corrected`, `alpha_corrected_stderr` | Same anomalous fit after subtracting the track's estimated localization offset from MSD first (see `offset_um2`) -- isolates the tau^alpha signal from the localization-noise plateau. |
| `n_points_used_alpha_corrected` | Lags surviving the offset subtraction (points driven non-positive are dropped). |
| `D_alpha_corrected_um2_s_alpha`, `r2_anomalous_corrected` | Generalized D and R^2 of the offset-corrected fit. |
| `offset_um2` | Per-track expected localization-noise MSD offset, `2*(mean(sigma_x_um^2)+mean(sigma_y_um^2))`, from the raw localization precision -- an independent check on `intercept_um2` and the value subtracted for `alpha_corrected`. |

### `classic/ensemble_msd.csv` (classic, `run_msd_analysis.py`)

One row per lag of the n_pairs-weighted ensemble MSD curve.

| Column | Meaning |
| --- | --- |
| `lag` | Lag index (in frames). |
| `tau_s` | Lag time in seconds (`lag * dt_s`). |
| `n_tracks` | Number of tracks contributing to this lag (lags below `min_tracks` are dropped upstream). |
| `n_pairs_total` | Total displacement pairs behind this lag's MSD estimate, summed across contributing tracks -- the ensemble-fit weight. |
| `msd_um2` | n_pairs-weighted ensemble-averaged MSD at this lag. |
| `msd_sem` | Standard error of the ensemble MSD at this lag. |

`classic/per_track_tamsd.parquet` holds the per-track, per-lag TAMSD this
table is averaged from (`track_id`, `track_length`, `lag`, `tau_s`,
`msd_um2`, `n_pairs`) -- intermediate data, not a fit result, kept for
re-plotting without recomputing TAMSD from raw localizations.

### `validate_localization_bias/simulation_recovery.csv` (classic validation, `validate_localization_bias.py`)

Same columns as `classic/per_track_msd_fits.csv`, plus:

| Column | Meaning |
| --- | --- |
| `true_D_um2_s` | Ground-truth D used to simulate this track (true alpha is always 1 here -- see the script's docstring). |

### `bayes/per_track_bayes_fits.csv` (Bayesian, `run_bayes_analysis.py`)

One row per track: the Brownian-constrained normal model and the anomalous
model, batched exact MAP (`inference.fit_batch_map`) joined on
`track_id`/`track_length`/`n_disp`. D, D_alpha and sigma are Laplace-fit in
log-space and back-transformed to an asymmetric `_median`/`_lo`/`_hi`
interval (see method notes above); alpha keeps a symmetric physical-space
interval.

| Column | Meaning |
| --- | --- |
| `track_id` | Track ID. |
| `track_length` | Number of localizations in the track. |
| `n_disp` | Number of per-frame displacements fit (`track_length - 1`). |
| `normal_converged` | L-BFGS-B convergence flag for the normal-model sub-batch containing this track (per sub-batch, not per track -- see `fit_batch_map`'s docstring). |
| `D_median_um2_s`, `D_lo_um2_s`, `D_hi_um2_s` | Normal-model (Brownian-constrained) D: back-transformed posterior/Laplace median and asymmetric 1-sigma interval. **Primary D estimate.** |
| `log10_D`, `log10_D_stderr` | The same normal-model D fit in log10 space (symmetric there by construction). |
| `sigma_normal_median_um`, `sigma_normal_lo_um`, `sigma_normal_hi_um` | Localization precision sigma from the normal-model fit, same median/interval convention as D. |
| `log10_sigma_normal`, `log10_sigma_normal_stderr` | That sigma in log10 space. |
| `anomalous_converged` | L-BFGS-B convergence flag for the anomalous-model sub-batch containing this track. |
| `D_alpha_median_um2_s_alpha`, `D_alpha_lo_um2_s_alpha`, `D_alpha_hi_um2_s_alpha` | Anomalous-model generalized diffusion coefficient, same median/interval convention. **Secondary/diagnostic** -- degrades faster than D on short tracks (FINDINGS.md). |
| `log10_D_alpha`, `log10_D_alpha_stderr` | That D_alpha in log10 space. |
| `sigma_anom_median_um`, `sigma_anom_lo_um`, `sigma_anom_hi_um` | Localization precision sigma from the anomalous-model fit. |
| `log10_sigma_anom`, `log10_sigma_anom_stderr` | That sigma in log10 space. |
| `alpha`, `alpha_stderr` | Anomalous exponent MAP and symmetric Laplace standard error (physical space -- not log-transformed). **Primary alpha estimate.** |

### `bayes/bayes_vs_classic_comparison.csv` (`run_bayes_analysis.py`)

`bayes/per_track_bayes_fits.csv`'s columns, inner-joined on `track_id`
against the classic table, plus:

| Column | Meaning |
| --- | --- |
| `D_classic_um2_s` | Classic pipeline's `D_um2_s` for the same track, carried over for direct comparison. |
| `alpha_classic` | Classic pipeline's `alpha` for the same track. |

### `validate_bayes_recovery/bayes_validate_null_D_alpha_bias.csv` (Bayesian validation, `validate_bayes_recovery.py`, check 1)

One row per simulated track (true alpha=1, true D swept), flat-prior and
informative-prior fits side by side.

| Column | Meaning |
| --- | --- |
| `track_id`, `track_length`, `n_disp` | As above. |
| `D_weak`, `D_stderr_weak`, `sigma_normal_weak`, `sigma_stderr_normal_weak` | Normal-model D/sigma, flat (`WEAK_NORMAL_PRIOR`) fit. |
| `D_bayes`, `D_stderr_bayes`, `sigma_normal_bayes`, `sigma_stderr_normal_bayes` | Normal-model D/sigma, informative-prior fit. |
| `D_alpha_weak`, `D_alpha_stderr_weak`, `sigma_weak`, `sigma_stderr_weak`, `alpha_weak`, `alpha_stderr_weak` | Anomalous-model fit, flat prior. |
| `D_alpha_bayes`, `D_alpha_stderr_bayes`, `sigma_bayes`, `sigma_stderr_bayes`, `alpha_bayes`, `alpha_stderr_bayes` | Anomalous-model fit, informative prior. |
| `true_D_um2_s_alpha` | Ground-truth D used to simulate this track. |

(This check uses the SVI comparison path, `fit_all_tracks`, not the
production `fit_batch_map` -- hence the un-suffixed `D`/`alpha` names rather
than `_median`/`_lo`/`_hi`, and the `_weak`/`_bayes` prior-comparison suffix
in place of `_um2_s`.)

### `validate_bayes_recovery/bayes_validate_alpha_recovery.csv` (Bayesian validation, `validate_bayes_recovery.py`, check 2)

One row per simulated track (true alpha swept at fixed D, flat prior).

| Column | Meaning |
| --- | --- |
| `track_id`, `track_length`, `n_disp` | As above. |
| `D_alpha`, `D_alpha_stderr`, `sigma`, `sigma_stderr`, `alpha`, `alpha_stderr` | Anomalous-model fit (flat prior, SVI path). |
| `true_D_um2_s_alpha`, `true_alpha` | Ground truth used to simulate this track. |

### `validate_bayes_recovery/bayes_validate_short_track_degeneracy.csv` (Bayesian validation, `validate_bayes_recovery.py`, check 3)

One row per simulated `track_length`, not per track.

| Column | Meaning |
| --- | --- |
| `track_length` | Simulated track length tested. |
| `n_replicates` | Number of simulated tracks at this length. |
| `degenerate_frac_weak` | Fraction of flat-prior fits landing on a parameter's support boundary (`D_alpha` below floor, or `alpha` within epsilon of 0 or 2). |
| `degenerate_frac_bayes` | Same, informative-prior fit. |

### `anisotropy/per_track_master.csv` (anisotropy, `run_anisotropy_analysis.py`)

One row per short (track_length 5-10) track: `per_track_log_bayes_factor`'s
Bayes factor joined against `sample_posterior_table`'s (NUTS) `eps`/`psi`
posterior and each track's mean field-of-view position -- the table the
plots in FINDINGS.md ("Visual inspection") are built from. Every quantity
keeps its own explicit name; `log_bf10` and the `eps_*`/`psi_*` columns
answer different questions with very different per-track reliability at
this N (see Method notes above and FINDINGS.md) and should not be conflated.

| Column | Meaning |
| --- | --- |
| `track_id`, `track_length`, `n_disp` | As above. |
| `log_bf10` | Log Bayes factor for anisotropy (`bayes_factor.log_bayes_factor_anisotropy`) -- the quantity meant to be trusted per-track at this N; near 0 for nearly every real track here (see FINDINGS.md). |
| `eps_median`, `eps_lo`, `eps_hi` | Anisotropy-fraction posterior median and 90% HPDI (NUTS) -- a secondary, honestly-wide descriptive interval, not a per-track detector (FINDINGS.md's sampling-noise-floor result). |
| `psi_median_rad`, `psi_lo_rad`, `psi_hi_rad` | Orientation posterior median/HPDI, radians in [0, pi) -- expect this to be poorly constrained whenever `eps_hi` is small (the psi-ridge FINDINGS.md documents). |
| `D_mean_median_um2_s`, `D_par_median_um2_s`, `D_perp_median_um2_s` | Mean/parallel/perpendicular diffusivity posterior medians from the same anisotropic-model fit (with matching `_lo_um2_s`/`_hi_um2_s` columns, omitted here for brevity). |
| `x_mean_um`, `y_mean_um` | Track's mean position in the field of view -- used for `plot_spatial_map`; join any future per-track spatial/structural label onto this table by `track_id` to group by it (`bayes.anisotropy.aggregate_log_bayes_factor`). |

`anisotropy/per_track_log_bf.csv` and `anisotropy/per_track_eps_posterior.csv`
hold the two halves of this table before the join (same columns, no
`x_mean_um`/`y_mean_um`); `anisotropy/ensemble_log_bf_all.csv` and
`_by_track_length.csv` hold the population-level `sum_log_bf10` this
script's verdict is based on; `anisotropy/null_calibration_ensemble_sums.csv`
holds the matched-composition null distribution (`null_sum_log_bf10`) it's
calibrated against.
