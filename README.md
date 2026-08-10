# Diffusion analysis: SPT mini-study

Two parallel pipelines for estimating diffusion coefficient (D) and
anomalous exponent (alpha) from single-particle tracking data, run on the
same raw localizations for direct comparison:

- **`analysis/`** -- classic MSD-curve-fitting approach.
- **`bayes/`** -- exact-likelihood Bayesian approach (numpyro), fit directly
  to per-frame displacements; never computes an MSD curve.

Empirical results, comparisons, and known pitfalls from running these
pipelines are documented in `FINDINGS.md`, not here -- this file covers
layout and how to reproduce a run.

## Layout

```
analysis/                   classic MSD pipeline, data-oriented design:
  io.py                       load CSV -> tidy polars DataFrame, physical units
  msd.py                      per-track TAMSD + n_pairs-weighted ensemble MSD
  fitting.py                  normal-diffusion (linear) & anomalous (log-log) fits,
                               localization-offset diagnostics, quality flags
  simulate.py                  ground-truth Brownian track generator (same schema
                               as io.load_tracks, for bias validation)
  viz.py                      plotting functions (pure: data in, Figure out)
bayes/                      exact-likelihood Bayesian pipeline (numpyro), no MSD:
  likelihood.py                covariance of observed displacements, jax.numpy
                               (Brownian + fGn-generalized anomalous, + static
                               localization noise) -- pure math, no numpyro import
  model.py                     numpyro models built on likelihood.py: normal_/
                               anomalous_diffusion_model (single track, used for
                               NUTS) and batched_normal_/batched_anomalous_
                               diffusion_model (numpyro.plate over many tracks
                               of the same length at once, used for the full
                               per-track table -- see its docstring)
  priors.py                    prior hyperparameters (dataclasses), incl.
                               WEAK_*_PRIOR (near-flat, MLE-equivalent) and a
                               per-track sigma prior from measured
                               x_std_um/y_std_um
  inference.py                 fit_map (single track, exact MAP + Laplace via
                               numpyro's own potential_fn + JAX autodiff),
                               sample_posterior (single track, full NUTS),
                               fit_batch_map (grouped by track_length,
                               sub-batched exact MAP -- production full-table
                               path), fit_batch_svi + fit_all_tracks (grouped
                               by track_length, mean-field SVI -- comparison/
                               validation path)
  simulate.py                   ground-truth fBm(+noise) track generator, any
                               alpha (same schema as analysis.io.load_tracks);
                               simulate_anisotropic_tracks does the same for
                               the anisotropic model below
  bayes_factor.py              prior-predictive Monte Carlo log Bayes factor
                               for anisotropic_diffusion_model vs. its eps=0
                               (isotropic) restriction, plus per-track and
                               label-grouped aggregation across many tracks
                               -- see Method notes and FINDINGS.md
                               ("Anisotropy detection")
  viz.py                      posterior/fit diagnostic plots
scripts/
  run_msd_analysis.py         classic pipeline: real data -> tables/figures
  validate_localization_bias.py  classic pipeline: simulated ground truth ->
                               checks whether a fitting bias is an artifact
                               vs. real physics
  run_bayes_analysis.py       exact-likelihood pipeline: real data -> tables/
                               figures, incl. comparison against the classic
                               pipeline's saved results
  validate_bayes_recovery.py  exact-likelihood pipeline: simulated ground
                               truth -> null-bias check, alpha-recovery check,
                               short-track degeneracy rate
  validate_anisotropy_recovery.py  anisotropic model: simulated ground truth
                               -> eps reduction/recovery/coverage checks and
                               log-Bayes-factor null-calibration/ensemble-
                               aggregation checks (see FINDINGS.md)
  run_anisotropy_analysis.py  anisotropic model: real data (track_length
                               5-10) -> per-track log Bayes factor + eps/psi
                               posterior, ensemble aggregation, matched-null
                               calibration, and the trajectory/spatial/
                               interval plots in FINDINGS.md ("Visual
                               inspection")
results/
  tables/
    classic/                    real-data per-track and ensemble fit tables,
                               from run_msd_analysis.py
    bayes/                      real-data per-track fit table and the
                               vs.-classic comparison table, from
                               run_bayes_analysis.py (the comparison table
                               lives here since it's that script's output,
                               not a third pipeline)
    anisotropy/                  real-data per-track log Bayes factor, eps/psi
                               posterior, joined master table, and ensemble/
                               null-calibration tables, from
                               run_anisotropy_analysis.py
    validate_localization_bias/  simulation-recovery table, classic pipeline
    validate_bayes_recovery/     simulation-recovery tables, Bayesian pipeline
    validate_anisotropy_recovery/  simulation-recovery tables, anisotropic model
  figures/                     same split as tables/, same reasons:
                               classic/, bayes/, anisotropy/,
                               validate_localization_bias/,
                               validate_bayes_recovery/, validate_anisotropy_recovery/
```

Every script owns one `results/{tables,figures}/<subfolder>/`: the two
production scripts get one each named for their workflow (`classic`,
`bayes`), and the two `validate_*.py` scripts (which fit *simulated* ground
truth, not the real dataset) get one each named for the script. Nothing is
written to the flat top level of `results/tables/` or `results/figures/`
directly -- this keeps real-data production results, and simulation/
validation results, and the two pipelines' own results, from ever mixing in
one folder.

Every function in both `analysis/` and `bayes/` is pure: numpy/jax arrays or
polars DataFrames in, new data out, no shared mutable state -- which is what
makes the two pipelines directly composable for comparison
(`run_bayes_analysis.py` joins its own per-track table against `analysis`'s
saved per-track table on `particle`) despite estimating from entirely
different statistics (a fitted MSD curve vs. the raw displacement
likelihood).

## Input data

Raw localization CSV, one row per (particle, frame): columns `particle`,
`frame`, `x`, `y` (pixels), `x_std`, `y_std` (localization precision,
pixels), `track_length`. `analysis.io.load_tracks` converts to physical
units given an `AcquisitionParams(pixel_size_um, dt_s)`.
`analysis.io.assert_contiguous_tracks` checks every track sits on a gapless,
uniform frame grid -- both pipelines' time-averaged statistics assume this.

## Run

```
python3 scripts/run_msd_analysis.py
python3 scripts/validate_localization_bias.py
python3 scripts/run_bayes_analysis.py          # run the classic pipeline first,
python3 scripts/validate_bayes_recovery.py     # so the vs.-classic comparison table gets built
```

Each run overwrites its own tables/figures in `results/` in place (by
filename, not a timestamped subfolder) -- `results/` always reflects the
most recent run of each script, not a history of past runs.

## Method notes

### Classic pipeline (`analysis/`)

- **TAMSD**: `MSD(lag) = mean_i[(x[i+lag]-x[i])^2 + (y[i+lag]-y[i])^2]` per
  track (`msd.py::_track_tamsd_arrays`).
- **Ensemble MSD**: per-track TAMSD averaged across tracks at each common
  lag, weighted by each track's `n_pairs` at that lag. Lags supported by
  fewer than `min_tracks` tracks are dropped.
- **Fit range**: normal- and anomalous-diffusion fits use only the first
  `n_fit_points` lags (`fitting.n_fit_points`: a fraction of a track's lags,
  capped at a fixed maximum -- see its docstring for why both parts of the
  rule matter).
- **D fit**: `MSD(tau) = 4*D*tau + b`, weighted least squares (weight =
  n_pairs at each lag).
- **alpha fit**: `MSD(tau) = 4*D_alpha*tau^alpha`, OLS in log-log space.
- **Localization-offset diagnostic**: `b` should approximate
  `2*(mean(x_std^2) + mean(y_std^2))` for static, R=0 (no motion-blur
  correction) localization noise. `weighted_expected_offset` computes this
  with the same n_pairs weighting as the ensemble fit. Camera exposure/duty
  cycle isn't recorded in the input schema here, so R=0 is a simplification.

### Exact-likelihood pipeline (`bayes/`)

- **Likelihood**: the observed per-frame displacement sequence is modeled as
  zero-mean multivariate Gaussian, with covariance = true-motion covariance
  (fractional Gaussian noise, generalizing Brownian motion to anomalous
  diffusion) + static localization-noise covariance. See `likelihood.py` and
  `model.py` for the full derivation and references. No MSD curve is
  computed anywhere in this pipeline.
- **Models**: `normal_diffusion_model` (2 params: D, sigma; alpha pinned to
  1) and `anomalous_diffusion_model` (3 params: D_alpha, sigma, alpha).
  `batched_*` variants fit many same-length tracks at once via
  `numpyro.plate` -- see `inference.fit_all_tracks`'s docstring for when and
  why this matters.
- **Priors**: LogNormal on D/D_alpha and sigma, Beta (rescaled to (0,2)) on
  alpha; see `priors.py`. A near-flat preset (`WEAK_*_PRIOR`) makes the same
  inference code behave like a flat-prior MLE.
- **Inference**: `inference.fit_map`/`sample_posterior` for single-track
  exact MAP / full NUTS. For the full per-track table, `inference.fit_batch_map`
  (batched exact MAP via L-BFGS-B, sub-batched to a `max_batch_size` cap) is
  production; `inference.fit_all_tracks` (batched SVI/Adam) remains for
  comparison/validation -- see FINDINGS.md for why and the accuracy/speed
  tradeoff between them.
- **D and D_alpha are reported in log-space with an asymmetric interval**
  (`{name}_median`/`_lo`/`_hi`, `log10_{name}`/`_stderr`), not a symmetric
  mean +/- stderr in linear units -- see FINDINGS.md ("D should be reported
  in log-space, with an asymmetric interval") for why a linear-space
  symmetric interval is a poor description of D's uncertainty, especially on
  short tracks. alpha keeps a symmetric physical-space interval.
- **D (normal model) and alpha (anomalous model) are the primary per-particle
  diffusive-behavior metrics; D_alpha is a secondary/diagnostic quantity** --
  D_alpha's posterior degrades much faster than D's on short tracks
  (FINDINGS.md).
- **Two distinct D-vs-alpha comparisons**: pairing D from
  `normal_diffusion_model` against alpha from `anomalous_diffusion_model`
  (cross-model) answers a different question than pairing D_alpha and alpha
  *within* the same joint anomalous fit -- see `model.py`'s docstring. Don't
  conflate the two; both scripts report them separately.

### Anisotropy detection (`bayes/anisotropic_diffusion_model`, `bayes/bayes_factor.py`)

Targets short (N=5-10) tracks specifically -- a directional generalization
of `normal_diffusion_model`, not a separate pipeline. See FINDINGS.md
("Anisotropy detection") for the full derivation, critique of the earlier
draft it replaced, and the empirical results below; this section is only
the operating summary.

- **Model**: rotated anisotropic diffusion tensor, parameterized by
  `D_mean` (mean of the two principal diffusivities), `eps` in [0,1) (their
  normalized difference -- the anisotropy fraction/eccentricity, 0 =
  isotropic), and `psi` (orientation, mod pi). `eps=0` reduces *exactly* to
  `normal_diffusion_model` (verified in `validate_anisotropy_recovery.py`),
  so H0 is nested in H1 rather than a separately-constructed alternative.
- **A point/interval estimate of `eps` has limited value at N=5-10**: 4-9
  displacement vectors carry a large sampling-noise floor on any continuous
  eccentricity estimate (even a genuinely isotropic track often "looks"
  substantially elongated by chance) -- no fixed prior strength gives both a
  controlled false-positive rate and real sensitivity from one track's data
  alone. Read `eps`'s posterior as an honestly wide, descriptive interval,
  not a detector.
- **The Bayes factor is the detector.** `bayes_factor.log_bayes_factor_anisotropy`
  (single track) / `batched_log_bayes_factor_anisotropy` (many tracks
  sharing n_disp) answer a better-posed question instead: "is this data more
  consistent with some anisotropy than with none," via prior-predictive
  Monte Carlo estimation of `p(data|H1)`/`p(data|H0)` -- no separate
  reference-distribution simulation needed, since how much apparent
  elongation is expected from sampling noise alone at this track length
  falls directly out of the marginal-likelihood integral.
- **Individual short tracks are almost always "inconclusive" by this
  measure -- correctly, not a bug.** Evidence only becomes decisive when
  summed across many tracks that share real anisotropic behavior (see
  FINDINGS.md's ensemble-aggregation check). `per_track_log_bayes_factor`
  computes `log_bf10` for every track in a table (same input schema as
  `inference.fit_all_tracks`); `aggregate_log_bayes_factor(per_track,
  label_col)` sums it grouped by **any** column already on (or joined onto)
  that table -- there is nothing anisotropy-specific about what defines a
  group. Group by track-length-derived buckets, an experimental condition,
  or (once available) a per-particle spatial/structural classification, by
  joining that classification onto `per_track_log_bayes_factor`'s output and
  passing its column name as `label_col`; no code change needed to add a new
  kind of grouping.

## Model equations

Compact reference for the model each fit actually solves. `x_i`/`y_i` are
observed positions, `Δt` the frame interval, `D`/`D_alpha` diffusion
coefficients, `alpha` the anomalous exponent, `sigma` static localization
precision. Full derivations, references, and empirical checks are in
`FINDINGS.md`; this is only "what is the equation."

**Classic MSD (`analysis/`)** -- fit to the time-averaged mean squared
displacement, `MSD(n*dt) = mean_i[ (x[i+n]-x[i])^2 + (y[i+n]-y[i])^2 ]`:

- Normal diffusion: `MSD(tau) = 4*D*tau + b` (weighted least squares; `b`
  is the static-localization-noise offset, `~2*sigma^2`).
- Anomalous diffusion: `MSD(tau) = 4*D_alpha*tau^alpha` (OLS in log-log
  space).

**Exact-likelihood displacement model (`bayes/normal_diffusion_model`,
`anomalous_diffusion_model`)** -- no MSD curve; the per-frame displacement
sequence `dx_k = x[k+1]-x[k]` (dy likewise) is modeled directly as one
zero-mean multivariate Gaussian, `dx ~ N(0, Sigma)`, `Sigma = Sigma_motion +
Sigma_noise`:

- True-motion term (fractional Gaussian noise; generalizes Brownian motion
  to anomalous diffusion): `gamma(k) = D_alpha * dt^alpha * (|k+1|^alpha -
  2*|k|^alpha + |k-1|^alpha)`, `Sigma_motion[i,j] = gamma(|i-j|)`. At
  `alpha=1`: `gamma(0) = 2*D*dt`, `gamma(k>=1) = 0` -- ordinary Brownian
  motion (independent increments), i.e. `normal_diffusion_model`; general
  `alpha` is `anomalous_diffusion_model`.
- Static localization-noise term (`x_obs = x_true + eps`, `eps ~
  N(0,sigma^2)` iid per frame): `Sigma_noise[i,i] = 2*sigma^2`,
  `Sigma_noise[i, i+/-1] = -sigma^2`, `0` otherwise.
- `dx` and `dy` are independent (isotropic motion) and fit as two separate
  length-`n_disp` Gaussians sharing the same `Sigma`.

**Anisotropic diffusion model (`bayes/anisotropic_diffusion_model`)** --
generalizes the `alpha=1` model above to a rotated, directional diffusion
tensor. Per-step true-motion covariance (2x2, replacing the scalar
`2*D*dt`):

```
Sigma_step(psi) = R(psi) . diag(2*D_par*dt, 2*D_perp*dt) . R(psi)^T
R(psi) = [[cos(psi), -sin(psi)], [sin(psi), cos(psi)]]
D_par = D_mean*(1+eps),  D_perp = D_mean*(1-eps)
```

`eps` in `[0,1)` is the anisotropy fraction (0 = isotropic), `psi` in
`[0,pi)` the orientation. `dx`/`dy` are no longer independent (`Sigma_step`
has off-diagonal terms unless `psi` is 0 or pi/2), so this is one joint
`2*n_disp`-dim Gaussian over the interleaved `(dx_1,dy_1,dx_2,dy_2,...)`
sequence, block-tridiagonal in 2x2 blocks: diagonal block =
`Sigma_step + 2*sigma^2*I2`, adjacent off-diagonal block = `-sigma^2*I2`.
`eps=0` reduces exactly to the isotropic model above, any `psi`.

**Anisotropy Bayes factor (`bayes/bayes_factor.py`)** -- model comparison,
not parameter estimation. H0: `eps=0` (isotropic); H1: `eps ~ Beta(1,b)`
(shrunk toward isotropy), `psi ~ Uniform(0,pi)`; `D_mean`/`sigma` share one
`LogNormal` prior under both. Each marginal likelihood is a prior-predictive
Monte Carlo average (no NUTS/MAP involved):

```
p(data | H) ~= (1/S) * sum_s p(data | theta_s),   theta_s ~ p(theta | H)
log_BF10 = log p(data | H1) - log p(data | H0)
```

`log_BF10 > 0` favors anisotropy, `< 0` favors isotropic diffusion, `~0` is
inconclusive. Individual N=5-10 tracks are almost always inconclusive by
design (see FINDINGS.md); `aggregate_log_bayes_factor` sums `log_BF10`
across many tracks (grouped by any label column) to accumulate real
evidence for a population that genuinely shares anisotropic behavior.

## Results tables -- column reference

Both pipelines write `particle` and `track_length` with the same meaning
(particle ID from the input CSV; number of localizations in the track), so
their per-track tables always join cleanly on `particle`. Column names
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
| `particle` | Track/particle ID. |
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
| `offset_um2` | Per-track expected localization-noise MSD offset, `2*(mean(x_std_um^2)+mean(y_std_um^2))`, from the raw localization precision -- an independent check on `intercept_um2` and the value subtracted for `alpha_corrected`. |

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
table is averaged from (`particle`, `track_length`, `lag`, `tau_s`,
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
`particle`/`track_length`/`n_disp`. D, D_alpha and sigma are Laplace-fit in
log-space and back-transformed to an asymmetric `_median`/`_lo`/`_hi`
interval (see method notes above); alpha keeps a symmetric physical-space
interval.

| Column | Meaning |
| --- | --- |
| `particle` | Track/particle ID. |
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

`bayes/per_track_bayes_fits.csv`'s columns, inner-joined on `particle`
against the classic table, plus:

| Column | Meaning |
| --- | --- |
| `D_classic_um2_s` | Classic pipeline's `D_um2_s` for the same particle, carried over for direct comparison. |
| `alpha_classic` | Classic pipeline's `alpha` for the same particle. |

### `validate_bayes_recovery/bayes_validate_null_D_alpha_bias.csv` (Bayesian validation, `validate_bayes_recovery.py`, check 1)

One row per simulated track (true alpha=1, true D swept), flat-prior and
informative-prior fits side by side.

| Column | Meaning |
| --- | --- |
| `particle`, `track_length`, `n_disp` | As above. |
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
| `particle`, `track_length`, `n_disp` | As above. |
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
| `particle`, `track_length`, `n_disp` | As above. |
| `log_bf10` | Log Bayes factor for anisotropy (`bayes_factor.log_bayes_factor_anisotropy`) -- the quantity meant to be trusted per-track at this N; near 0 for nearly every real track here (see FINDINGS.md). |
| `eps_median`, `eps_lo`, `eps_hi` | Anisotropy-fraction posterior median and 90% HPDI (NUTS) -- a secondary, honestly-wide descriptive interval, not a per-track detector (FINDINGS.md's sampling-noise-floor result). |
| `psi_median_rad`, `psi_lo_rad`, `psi_hi_rad` | Orientation posterior median/HPDI, radians in [0, pi) -- expect this to be poorly constrained whenever `eps_hi` is small (the psi-ridge FINDINGS.md documents). |
| `D_mean_median_um2_s`, `D_par_median_um2_s`, `D_perp_median_um2_s` | Mean/parallel/perpendicular diffusivity posterior medians from the same anisotropic-model fit (with matching `_lo_um2_s`/`_hi_um2_s` columns, omitted here for brevity). |
| `x_mean_um`, `y_mean_um` | Track's mean position in the field of view -- used for `plot_spatial_map`; join any future per-particle spatial/structural label onto this table by `particle` to group by it (`bayes.aggregate_log_bayes_factor`). |

`anisotropy/per_track_log_bf.csv` and `anisotropy/per_track_eps_posterior.csv`
hold the two halves of this table before the join (same columns, no
`x_mean_um`/`y_mean_um`); `anisotropy/ensemble_log_bf_all.csv` and
`_by_track_length.csv` hold the population-level `sum_log_bf10` this
script's verdict is based on; `anisotropy/null_calibration_ensemble_sums.csv`
holds the matched-composition null distribution (`null_sum_log_bf10`) it's
calibrated against.
