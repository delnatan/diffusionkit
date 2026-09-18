# Short-track inference audit — 2026-09-18

This is the audit of the implementation before the classical rebuild. Classical
source links now point to its preserved legacy copy. See README.md for the
current API; the Bayesian findings still await their separate revision.

The core covariance mathematics passed the independent checks below. The
package is a useful research prototype, but its uncertainty reporting,
failure handling, and validation do not yet justify presenting its default
outputs as dependable measurements for 5–20-frame trajectories.

Scope: current working tree, including the pre-existing, uncommitted GLS
additions. Inspected classical fitting, Bayesian inference, anisotropy,
simulation, validation scripts, and the relevant spotsolve/spt-pipeline
interfaces. No production code or pre-existing edits were changed.

Reproduction: `.venv/bin/python scripts/audit_short_tracks.py`.
Saved output: [audit/short_tracks.json](/Users/delnatan/Projects/github/diffusionkit/audit/short_tracks.json).
The probes characterize existing behavior; they are not a regression suite
whose passing status certifies the algorithms. Environment: NumPy 2.5.2,
SciPy 1.18.1, Polars 1.44.1, JAX 0.11.1, NumPyro 0.21.0.

## Findings requiring correction

### 1. Public fits silently accept the wrong trajectory or time grid — P1

[bayes/api.py:145](/Users/delnatan/Projects/github/diffusionkit/diffusionkit/bayes/api.py:145)
sorts rows by frame and differences positions without validating one track
ID, consecutive frames, finite values, or agreement with `track_length`.
[classic/msd.py:47](/Users/delnatan/Projects/github/diffusionkit/diffusionkit/legacy/classic/msd.py:47)
also substitutes row separation for frame separation. The standalone
`assert_contiguous_tracks` helper is called by some scripts, not enforced
by the public fitting APIs.

Reproductions:

- Change a five-frame track's frames from 0,1,2,3,4 to 0,2,4,6,8, adjusting
  its timestamps. Both pipelines return identical results at the same
  acquisition `dt_s`, although every interval doubled.
- Pass two five-frame tracks to `fit_track`: it returns `track_id=0`,
  `track_length=5`, and `n_disp=9`, having interleaved two particles.

Fix: one shared validator at each public boundary. Derive length from rows;
require one ID for single-track calls; reject gaps/duplicates explicitly
until irregular-time likelihoods are supported. Check units, positive
finite acquisition parameters, positions, and uncertainty inputs. Sort
before checking contiguity. Current spotsolve linking is frame-to-frame,
so its ordinary output avoids gaps; imported or filtered tracks still
need this protection.

### 2. Intervals change meaning between MAP and NUTS — P1

[bayes/api.py:152](/Users/delnatan/Projects/github/diffusionkit/diffusionkit/bayes/api.py:152)
returns ±1 standard deviation in transformed space for positive parameters,
and a symmetric delta-method interval for alpha. The NUTS branch returns
an HPDI with default probability 0.9. The result uses the same `lo`/`hi`
fields without interval probability/type metadata.

Reproduction: `hpdi_prob=0.68` and `hpdi_prob=0.95` give exactly the same MAP
D interval, [0.03171994, 0.10308280]. Thus changing the method changes both
the approximation and the nominal interval probability. A GUI cannot
honestly compare the apparent widths without additional information.

Fix: expose `interval_prob` consistently and record the interval method.
For transformed Laplace intervals, use the appropriate normal quantile
and transform alpha's logit endpoints back to (0,2), rather than using an
unbounded symmetric interval. This fixes semantics, not approximation
accuracy; the next finding remains.

### 3. Five-frame Laplace accuracy is overstated — P1

[bayes/api.py:136](/Users/delnatan/Projects/github/diffusionkit/diffusionkit/bayes/api.py:136)
claims adequate calibration even at N=5. The committed validation does not
establish this over the intended operating range.

Independent check: simulate four Brownian tracks at N=5, D=0.01 µm²/s,
dt=0.033 s, localization SD=0.025 µm, seed 444. Integrate the two-parameter
normal-model posterior on a log(D), log(sigma) grid, using an independent
sine-basis likelihood and the same default prior. The posterior masses
inside the returned nominal 68.27% intervals were:

| Track | Posterior mass in returned interval |
| --- | ---: |
| 0 | 61.99% |
| 1 | 51.73% |
| 2 | 52.49% |
| 3 | 50.80% |

Doubling grid resolution changed those masses by less than 0.00016.
These are four conditional posterior checks, **not** an estimate of
frequentist coverage over an entire population. They are nevertheless
counterexamples to treating the reported intervals as reliably calibrated
in this regime, even when the assumed Brownian model is exactly correct.

Fix: validate approximation quality across track length and
noise-to-motion ratio. Keep Laplace explicitly labeled approximate;
benchmark low-dimensional quadrature and diagnostic-checked NUTS for
short, weakly identified cases. Transforming D to log space helps but does
not itself validate a Gaussian posterior approximation.

### 4. Nonlinear GLS can report unusable results without a validity status — P1

[classic/fitting.py:532](/Users/delnatan/Projects/github/diffusionkit/diffusionkit/legacy/classic/fitting.py:532)
treats failure to find an improving step as convergence. Exhausting the
iteration budget also returns a result. The inner optimizer is unbounded
in alpha; the outer loop clips only the next pilot, then returns the
unclipped fit. `singular=False` does not imply convergence, finite
uncertainty, or membership in the modeled range.

Fixed-seed stress check: 100 independent Brownian tracks per cell, true
alpha=1, dt=0.033 s, localization SD=0.025 µm, exact localization offset.
Current `alpha_nlgls_corrected` results:

| Frames | D | Median alpha | Alpha outside (0,2) | Nonfinite alpha | Nonfinite stderr with singular=False |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.01 | 1.165 | 39 | 5 | 0 |
| 5 | 0.05 | 0.874 | 26 | 2 | 2 |
| 10 | 0.01 | 1.111 | 24 | 1 | 2 |
| 10 | 0.05 | 0.854 | 6 | 0 | 0 |
| 20 | 0.01 | 0.965 | 4 | 0 | 0 |
| 20 | 0.05 | 0.933 | 0 | 0 | 0 |

Every out-of-range result above had `singular=False`. Unconstrained
classical estimates are not intrinsically implementation errors; preserving
them can be useful. The failure is recommending these as a default without
adequate status flags and short-track validation. The nonfinite uncertainties
also show a concrete failure of the numerical success contract.

Fix: replace the custom Levenberg–Marquardt implementation with SciPy
`least_squares` on Cholesky-whitened residuals. Return termination reason,
finite-result checks, rank/conditioning, and support/boundary flags.
Decide explicitly whether the comparison estimator remains unconstrained.
Do not silently clamp the final result and then call it validated.

### 5. Invalid Laplace curvature can become zero uncertainty — P1

[bayes/inference.py:124](/Users/delnatan/Projects/github/diffusionkit/diffusionkit/bayes/inference.py:124)
inverts the Hessian without checking positive definiteness, then clips
negative diagonal entries of the resulting covariance to zero before
taking square roots. An indefinite Hessian must invalidate the Laplace
approximation; zero uncertainty means the opposite. The broad exception
handler and batch-wide optimizer flag do not communicate that distinction.

This is a confirmed unsafe code path, not a claim that it occurred in the
small production-model smoke check run here. Fix with an SPD factorization,
finite/conditioning checks and separate optimization/uncertainty statuses.
Expose NUTS divergences, ESS and R-hat, and nested-sampling termination
status as well; current result tables discard these diagnostics.

### 6. Short-track and small-selection workflows fail before producing results — P2

[bayes/inference.py:309](/Users/delnatan/Projects/github/diffusionkit/diffusionkit/bayes/inference.py:309)
defaults to excluding tracks shorter than 10 frames; if all tracks are
excluded it calls `pl.concat([])` and raises `ValueError`.
[classic/api.py:89](/Users/delnatan/Projects/github/diffusionkit/diffusionkit/legacy/classic/api.py:89)
requires an ensemble fit before returning any per-track results. Two
five-frame tracks with `min_track_length=5` still fail with
`ZeroDivisionError` under the default ten-track ensemble threshold.

Fix: decouple individual fits from optional ensemble summaries. Return
typed empty tables and explicit exclusion reasons. Revisit the minimum
length for the stated 5–20-frame use case; accepting short tracks must
not imply their parameters are well identified.

### 7. “MAP” and “weak prior = MLE” describe the wrong estimator — P2

[bayes/inference.py:94](/Users/delnatan/Projects/github/diffusionkit/diffusionkit/bayes/inference.py:94)
minimizes NumPyro's unconstrained potential, including change-of-variable
Jacobians. It therefore finds a mode in log(D), log(sigma), logit(alpha/2)
coordinates. Back-transforming that point does not produce the mode of
the posterior density in physical coordinates. It can serve as the median
of a transformed Gaussian approximation, not generally the exact posterior
median either. This behavior agrees with the
[NumPyro implementation](https://num.pyro.ai/en/stable/_modules/numpyro/infer/util.html).

Two analytic probes through the actual `fit_map` implementation:

- A prior-only LogNormal(0,1) returns 1.0; its physical-space mode is
  exp(-1)=0.367879.
- A uniform alpha prior on (0,2) with likelihood Normal(observation=0.2,
  mean=alpha, SD=0.4) returns alpha=0.451256; the MLE and physical posterior
  mode are both 0.2. The logit Jacobian remains despite the uniform prior.

This does **not** invalidate NUTS, which requires those Jacobians, or
transformed-space Laplace as an approximation. Fix the estimator naming
and implement an explicit likelihood-only objective if MLE is wanted.
Broad lognormal priors and mean-field SVI are also not generic synonyms
for a flat-prior MLE.

## Scientific assumptions and validation gaps

**Localization errors are not used per observation.**
[priors.py:89](/Users/delnatan/Projects/github/diffusionkit/diffusionkit/bayes/priors.py:89)
reduces both axes and all frames to one RMS precision, then assigns its
log-prior SD as `max(0.15, 1/sqrt(2N))`. This is a heuristic, not a
calibration of uncertainty in spotsolve's CRLB estimates. Permuting which
frame has a poor precision leaves the prior and likelihood unchanged.
The classic constant offset is likewise not the exact lag-specific
correction for heteroscedastic localization errors.

The existing spt-pipeline adapter correctly maps `se_x`/`se_y` to position
uncertainties. Preserve that information in the likelihood. For independent
position errors with variances s_i², displacement noise has diagonal
s_i²+s_(i+1)² and adjacent covariance -s_(i+1)², separately per axis.
A calibrated multiplicative uncertainty factor can be inferred if needed.
This is a modeling extension, not a defect in the current constant-noise
formula. Brownian likelihoods supporting variable localization error and
intermittent observations already exist in the
[primary literature](https://pubmed.ncbi.nlm.nih.gov/27176323/).

**The validation does not match the production claim.** There is no
`tests/` directory or CI configuration in this tree. The main Bayesian
recovery script runs SVI even though production uses Laplace. The GLS
script uses 60-frame Brownian tracks, exact localization precision, and
reports OLS/log-GLS comparisons; it does not reproduce the documented
nonlinear-GLS-versus-Bayes timing and accuracy comparison. Its per-track
output includes nonlinear fits, but that is not a validation of their
uncertainties or short-track recommendation. Shared simulator/likelihood
formulas should be supplemented by independent oracles, since identical
mistakes can otherwise cancel.

**FGLS is not guaranteed unbiased merely because its weights are SPD.**
The claim in `_gls_solve` and README holds for suitable fixed/exogenous
weights in a correctly specified linear mean model. Here the weights
depend on pilot estimates from the same noisy track. Their dependence on
the response defeats that argument; plug-in standard errors also omit
weight-estimation uncertainty. Log and nonlinear fits add further
approximations. Assess bias and interval behavior empirically.

**Ensemble SEM is not a general weighted-mean standard error.**
`classic/msd.py:92` divides the weighted population variance by the number
of tracks. Even with equal weights it lacks the finite-sample correction;
with unequal weights it is not a general estimate of variance of the
weighted mean. Define the sampling target and variance model, or bootstrap
independent tracks. Unequal track lengths also change population composition
at successive lags, which can mimic curvature in an ensemble MSD.

**Alpha is conditional on an fBm model.** Drift, confinement, switching,
linking errors, motion blur, and non-Gaussian transport are not separated
by this model. An alpha estimate below 1 does not establish a particular
mechanism, and pairing it with small radius of gyration does not prove
confinement. Retain model-free geometry as descriptors; remove causal
claims such as the one in `classic/features.py`'s opening docstring.
Exposure time belongs in the acquisition contract before using camera data
to support claims of an exact observation likelihood.

**Anisotropy needs diagnostic and prior provenance.** The optimized
likelihood passed the independent check below, but this audit did not run
a complete nested-sampling calibration. `tau_log_ratio` is the SD of
each Cartesian h component; the radial half-log axis ratio |h| is
Rayleigh distributed, not Gaussian with that SD. Correct its elicitation
description. `evidence_label(NaN)` also falls through to “weak evidence
for isotropic”; nonfinite evidence must be labeled failed/unknown. A Bayes
factor is evidence conditional on the stated priors and model, not a
frequentist false-positive guarantee.

## What passed

- NumPy and JAX displacement covariances agreed exactly on tested lengths
  5, 10, 20 and alpha 0.3, 1, 1.7.
- Optimized TAMSD covariance agreed with independently constructed window
  matrices and Wick's formula to relative error 1.41e-15, including all lags.
- Anisotropic sine-transform likelihood agreed with a dense SciPy Gaussian
  likelihood to 3.56e-15 log units at the tested nontrivial orientation.
- Actual single-track anomalous and two-track `model="both"` MAP smoke
  calls completed with finite results and optimizer success. Single versus
  batch alpha agreed to about 5e-6 on the checked track.

These checks support retaining the covariance kernels. They do not certify
every parameter limit, posterior approximation, or nested sampler run.

## A smaller engine for spotsolve → diffusionkit → napari

1. **Unify input and result contracts first.** One validated track object or
   table schema containing frame/time, physical positions, per-observation
   uncertainty, and acquisition/exposure metadata. One result schema with
   model, estimator, estimate, interval probability/type, per-track status,
   diagnostics, prior settings and data/version provenance. Keep scientific
   computation independent of widgets and plots.
2. **Keep a small default comparison.** A classical reference plus a normal
   displacement-likelihood estimate of D, and an optional fBm estimate of
   alpha/K. Keep the growing OLS/GLS/corrected variants accessible for
   studies rather than computing and presenting every variant by default.
   Include a covariance-based Brownian estimator as a candidate fast
   reference; the cited
   [Vestergaard paper](https://pubmed.ncbi.nlm.nih.gov/25353527/) explicitly
   develops this regression-free estimator. Benchmark before selecting it.
3. **Exploit independent tracks directly.** Use a reusable likelihood and
   compiled derivative function per track shape; optimize tracks with
   individual stopping/status checks. Compute 2×2 or 3×3 Hessians per track
   instead of a dense matrix over unrelated tracks. Vectorize kernels where
   useful. The current dense batch Hessian and repeated closure compilation
   are implementation costs, not an intrinsic cost of Bayesian inference.
   Brownian tridiagonal or sine-basis likelihoods offer another cheap path;
   heteroscedastic noise preserves tridiagonality but not the same sine basis.
4. **Make uncertainty visible in the UI.** The current spt-pipeline
   `_normalize_map_table` deliberately removes intervals and convergence
   fields from the shared track table, retaining them only in saved full
   results. Carry a usable status and uncertainty summary into filtering,
   selection and tooltips. Do not color a five-frame alpha estimate as a
   confident motion class merely because it lies below or above 1.
5. **Validate the actual intended workflow.** Cover N=5,8,10,15,20; a range
   of diffusion/noise ratios and alpha; calibrated and miscalibrated
   localization errors; blur, drift, missing observations and linking
   mistakes. Report bias, RMSE, interval behavior, failures, prior
   sensitivity, and cold/warm runtime. Separate posterior-approximation
   checks from repeated-sampling coverage and prior-predictive calibration.
   Exercise detection/linking/selection as well as ideal tracks, since
   survival and linking can alter the distribution being analyzed.

For interpretability, D from the normal model should remain explicitly
conditional on Brownian motion. K's units depend on alpha; an optional
MSD amplitude at a stated reference lag can provide a clearer cross-track
comparison within the anomalous model. For many short tracks, a later
hierarchical model with per-track parameters could share information
without imposing a single D on the population; the failure of a common-D
pooled model is not a reason to rule out all population inference.

Suggested order: boundary validation and result semantics; optimizer and
uncertainty failure handling; independent short-track calibration; per-frame
noise/exposure modeling; then performance work and documentation reduction.
Replace historical explanations and “honest/exact/impossible to be wrong”
language with short equations, assumptions, result definitions and measured
validation limits. Preserve the historical experiments in an archive.
