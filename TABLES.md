# Classical result data

This reference covers the new `analyze_track`/`analyze_tracks` API only.
Old schemas are preserved in [docs/archive/TABLES.md](docs/archive/TABLES.md).

`ClassicAnalysis` contains `fits`, `msd`, `acquisition`, `options`, and `mle_options`.
The settings must accompany saved tables to reproduce the analysis.

## fits — one row per track and model

| Column | Meaning |
| --- | --- |
| `track_id`, `n_frames` | Track identity and actual observation count |
| `model` | `brownian`, `power_law`, or `brownian_mle` |
| `method` | `msd_ols`, `msd_nls`, or `displacement_mle` |
| `status`, `message` | Numerical/input status and an explanation |
| `D_um2_s` | Brownian D (MSD fit or MLE); null on power-law rows |
| `K_um2_s_alpha`, `alpha` | Generalized coefficient and exponent; null on Brownian rows |
| `n_lags` | Number of lag points actually included |
| `residual_sum_squares_um4` | Unweighted SSE in linear, corrected MSD space; not a goodness-of-model probability |
| `optimizer_status` | SciPy least_squares termination code, or null if not applicable |
| `nfev` | Total nonlinear residual evaluations over three starting points; excludes analytic endpoint checks |
| `localization` | `provided` or explicitly `ignore` |
| `uncertainty_method` | `not_estimated` for MSD rows; `profile_likelihood_asymptotic` for `brownian_mle` |

Columns only filled on `brownian_mle` rows (see [docs/classical.md](docs/classical.md)):

| Column | Meaning |
| --- | --- |
| `D_upper_um2_s` | One-sided profile-likelihood upper limit on D (`MLEOptions.upper_level`), asymptotic |
| `log_likelihood` | Maximized Gaussian log-likelihood of both axes' displacements |
| `lr_motion`, `p_motion` | 2[l(D_hat) - l(0)] and its asymptotic ½χ²₀ + ½χ²₁ p-value |
| `z_nonbrownian`, `p_nonbrownian` | Bootstrap-calibrated signed deviation from the Brownian + noise model; ~N(0,1) under it |
| `z_nonbrownian_asymptotic` | The same score with the uncalibrated normal reference |
| `alpha_1step`, `alpha_1step_se` | One-step linearized alpha, 1 + U/I_eff, and 1/sqrt(I_eff) |
| `n_boot`, `n_boot_valid` | Replicates drawn and replicates that also resolved motion |

Uncomputable parameters are null. Failed fits can retain numerical estimates
for inspection; check `status` before interpretation. No row is removed for
a negative D, a boundary alpha, or a short track.

| Status | Meaning |
| --- | --- |
| `ok` | Finite numerical estimate passed the implemented checks; precision and model adequacy are not established |
| `nonphysical` | Negative Brownian D, retained without clipping |
| `boundary` | Zero Brownian D or alpha at/near 0 or 2 |
| `unresolved` | `brownian_mle`: D_hat = 0, localization noise explains the motion; no z |
| `unidentified` | Power-law amplitude is zero/negligible or its numerical Jacobian cannot identify alpha |
| `optimizer_failed` | Optimization did not converge or a coefficient was nonfinite |
| `insufficient_data` | Too few lags for the requested fit |
| `excluded` | Fewer frames than `min_frames`; MSD rows when `exposure_s > 0`; MLE rows with `localization="ignore"` |
| `invalid_input` | A batch track failed validation, or (MLE row only) a localization SD is zero |
| `failed` | `brownian_mle`: likelihood maximum not bracketed |

`unidentified` is a numerical check, not a complete statistical identifiability
test. Even `ok` alpha fits on short tracks can have large uncertainty and bias.

## msd — one row per track and selected lag

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
