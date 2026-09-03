# Findings

Empirical results from running the pipelines in this repo against
`mobile_beads_1to200.csv` (freely diffusing fluorescent beads, spinning-disc
confocal, MLE localization; pixel size 0.1043 um/px, dt = 0.033 s/frame;
539 tracks, 5-200 frames each). This file holds run results and the
reasoning behind them; `README.md` holds only how to reproduce them.
Re-running the scripts can change the specific numbers below (different
data, different track counts, different random seeds) -- treat this as a
record of one analysis pass, not a spec.

## Classic MSD pipeline (`diffusionkit.classic`)

Produced by `scripts/run_msd_analysis.py` and
`scripts/validate_localization_bias.py`.

### Baseline run

- Ensemble alpha = 1.008 +/- 0.004, per-track median alpha = 0.997 (IQR
  0.86-1.10) -- consistent with pure Brownian motion, as expected for free
  beads. Ensemble-level sanity check that the pipeline is unbiased.
- Ensemble D = 0.054 um^2/s; per-track median D = 0.053 um^2/s (IQR
  0.039-0.068) -- order-of-magnitude spread across beads, no strong
  dependence on track length (`D_vs_track_length.png`).
- The fitted localization-offset intercept trends below the expected value
  from x_std/y_std (8e-5 vs ~1e-3 um^2 at the ensemble level). Plausible
  explanations: motion-blur during camera exposure (which *lowers* the
  effective offset, unmodeled since the fits assume R=0), and/or the offset
  being dominated by the best-localized long tracks even after n_pairs
  weighting.

### Known pitfalls in per-track MSD fitting

The ensemble-averaged fit looks clean (R^2=0.9999, alpha~1). Everything
below lives in the *per-track* statistics. Numbers are from
`results/tables/per_track_msd_fits.csv` (n=365 tracks, track_length >= 10)
unless noted otherwise.

1. **Fitting too many lag points silently wrecks both D and the intercept --
   nothing crashes, it just quietly biases.** On individual 200-frame
   tracks, pushing the fit range past ~20 of the 199 possible lags moved D
   by 30-70% and flipped the sign of the fitted intercept. Root cause: MSD
   at large lag is computed from few, heavily overlapping (correlated)
   displacement pairs, so its variance blows up with lag. `fitting.py`'s
   `n_fit_points` caps the fit range at `max_points=10` regardless of track
   length, on top of a `frac=0.25` rule -- the fractional rule alone was not
   enough once tracks got long.

2. **Negative D estimates do occur** -- 2/365 tracks (0.5%), both short
   tracks pinned at the `min_points=3` floor. `fit_all_tracks` keeps every
   row and flags it (`D_negative`) rather than silently filtering.

3. **A bigger, sneakier issue: negative fitted intercepts.** 46% of all 365
   per-track fits (168 tracks) came back with a *negative* static-
   localization offset -- physically impossible, since it represents
   `2*(sigma_x^2 + sigma_y^2)`, a sum of variances. With only
   `min_points=3` (23% of tracks sit at this floor -- 1 degree of freedom in
   the fit), the intercept is barely constrained. A negative intercept is a
   symptom of this, not a separate bug.

4. **D and intercept are strongly anti-correlated per track (Pearson
   r=-0.72).** With few points, slope and intercept trade off against each
   other: a noisy early uptick reads as either "higher D" or "lower
   intercept" almost interchangeably. Neither parameter from a few-point fit
   should be trusted in isolation.

5. **R^2 does not reliably flag bad fits.** Median R^2 is >0.99 for both
   models across all tracks, even the visibly unreliable short/noisy ones --
   only the 2 outright-negative-D tracks show low R^2 (0.14, 0.91). A naive
   `R^2 > 0.9` quality gate would let essentially everything through.

6. **The big one -- fitted D and fitted alpha are spuriously correlated
   across tracks (Pearson r=0.73 on this dataset, r=0.75 in a matched
   ground-truth simulation), even though D and alpha are independent
   parameters for Brownian motion.** Confirmed by simulation
   (`scripts/validate_localization_bias.py`,
   `results/figures/simulation_alpha_bias.png`): simulated pure-Brownian
   tracks (true alpha=1 for every track) with constant localization noise
   but varying true D come back with median fitted alpha rising from 0.75
   (D=0.02 um^2/s) to 0.94 (D=0.15 um^2/s) -- a clean, monotonic, entirely
   artifactual trend. Mechanism: a fixed absolute localization offset is a
   larger fraction of the observable MSD for a slow particle than a fast
   one, so it flattens the log-log curve at short lag more for slow
   particles, pulling their fitted alpha down. Practical implication: a
   per-track D-vs-alpha trend (or correlation with anything else that
   co-varies with SNR) needs a matched noise-only null simulation before
   it's read as evidence of a real coupling between diffusivity and
   anomalous exponent.

7. **Subtracting the known localization offset before the log-log fit
   works -- but only as well as the offset estimate does.** Since
   `log(MSD_obs) = log(4*D*tau^alpha + b)` conflates the power-law signal
   with the additive offset `b`, subtracting a *known* `b` should fix the
   bias in principle. Tested via `fit_all_tracks(..., localization_offset=...)`
   (`*_corrected` columns) in two regimes:
   - **Simulation, exact known offset** (`results/figures/simulation_alpha_correction.png`):
     median alpha flattens from a 0.75->0.94 ramp with D down to ~0.96-1.00
     across the board, and r(D, alpha) drops from 0.75 to 0.24. Close to a
     full fix.
   - **Real data, offset estimated per-track from x_std/y_std**
     (`results/figures/parameter_distributions.png`): r(D, alpha) drops only
     from 0.73 to 0.59, and the alpha distribution *overshoots* -- median
     moves from 0.997 to 1.089, past the Brownian target rather than
     landing on it.
     The gap between these two outcomes is the point: in the simulation `b` is
     the exact truth; on real data it's `2*(mean(x_std^2)+mean(y_std^2))`
     averaged over as few as 10 frames per track -- a noisy estimate in its
     own right, and the correction inherits that noise. The offset-subtraction
     correction is a real improvement (moderate correlation reduction), not a
     solved problem, and its failure mode is systematic overcorrection, not
     just added noise.

## Exact-likelihood Bayesian pipeline (`diffusionkit.bayes`)

> **Naming note (2026-09-02 refactor).** The three per-track table builders
> were renamed for consistency and now share one grouping loop:
> `fit_batch_map` -> `fit_table_map`, `fit_all_tracks` -> `fit_table_svi`,
> `sample_posterior_table` -> `fit_table_nuts`. The single-batch engines
> (`fit_map`, `sample_posterior`, `fit_batch_svi`) keep their names. Findings
> below are unchanged and were re-verified bit-for-bit against the new code;
> only the identifiers were updated. `fit_table_svi` is no longer reachable
> through `fit_population` -- it is a comparison/validation path now, for the
> calibration reason documented below.

Produced by `scripts/run_bayes_analysis.py` and
`scripts/validate_bayes_recovery.py`.

### A methodological catch found during development

Mid-development (porting the inference engine to numpyro), a run of
`scripts/run_bayes_analysis.py` on the real dataset suddenly showed
r(D, alpha) = 0.73 -- reproducing the classic MSD artifact (pitfall #6
above) this pipeline exists to avoid. Individual per-track fits checked out
fine against full NUTS posteriors and exact single-track MAP, so the
estimator itself wasn't broken; the bug was in *which two columns were
being correlated*.

There are two different "D vs. alpha" questions this pipeline can answer,
and they give different numbers:
- Pairing the **Brownian-constrained D** (`normal_diffusion_model`, alpha
  pinned to 1) against alpha from the **separate** anomalous fit -- a
  short-time D under a forced normal-diffusion assumption, paired against an
  independently-fit power-law exponent. This is what the classic pipeline's
  r=0.73 measures (see `diffusionkit.classic.viz.plot_D_alpha_jointplot`), so it's the
  metric comparable across pipelines.
- Correlating D_alpha and alpha **from the same joint 3-parameter fit**.
  These two are jointly estimated from one likelihood surface with a real,
  expected Fisher-information trade-off (verified separately: r~0.85-0.9 at
  *fixed* true D in simulation, since overestimating alpha and
  underestimating D_alpha both raise the predicted MSD's late-time slope
  similarly). This is not the MSD-curve-fitting artifact the cross-pipeline
  comparison targets, and conflating the two during the numpyro port is what
  produced the spurious r=0.73.

Both scripts now report both, clearly labeled
(`bayes_D_alpha_jointplot.png` = cross-model / classic-comparable,
`bayes_D_alpha_jointplot_within_fit.png` = within-fit).

### Batched inference performance

Fitting tracks one Python-level call at a time -- even a single-track call
built on numpyro's own machinery (`inference.fit_map`: exact JAX
gradient/Hessian MAP) -- pays a fresh JAX trace per call regardless of shape
reuse: looping `fit_map` over 365 real tracks took ~17 minutes. Grouping
tracks by shared `track_length` and fitting each group in one
`numpyro.plate`-vectorized mean-field SVI run (`inference.fit_batch_svi`)
brought the same 365-track, 2-model fit down to ~9 minutes total. Per-track
medians from the batched SVI path agree closely with both full NUTS and
single-track exact MAP on a spot check (mean |alpha difference| = 0.022
across 40 tracks spanning the length range) -- the batching changed
performance, not the answer.

SVI step count matters: on a synthetic recovery test, dropping
`fit_batch_svi`'s default from 2000 to 400 steps degraded the correlation
between recovered and true D from 0.80 to 0.33. 2000 is not an arbitrary
default.

### Key findings, vs. the classic MSD pipeline

- **The D-alpha artifact (classic pitfall #6) is essentially gone.** On the
  real dataset, r(D_normal, alpha_anomalous) [the classic-comparable metric,
  see above] drops from 0.73 (classic MSD) to -0.22 (this pipeline,
  informative prior, batched SVI). A controlled null simulation (true
  alpha=1 for every track, true D swept 0.02-0.15 um^2/s, 60 replicates per
  D, same design as `validate_localization_bias.py`) confirms this isn't
  just real-data noise: classic MSD's median fitted alpha ramps 0.75 -> 0.94
  across that D range (pooled r=0.75); this pipeline's median alpha stays in
  a narrow band (flat-prior [1.06, 1.14], informative-prior [1.00, 1.08])
  with no monotonic trend against D, and pooled r(D_normal, alpha_anomalous)
  stays well under 0.3 for both priors (-0.09 flat, -0.22 informative) --
  see `results/figures/bayes_validate_null_bias_weak.png` vs.
  `results/figures/simulation_alpha_bias.png`. The *within-fit* r(D_alpha,
  alpha) [not the artifact metric] is smaller here than on the real dataset
  (0.31 flat / 0.16 informative vs. 0.73 real-data) -- consistent with the
  Fisher-information trade-off being real but its magnitude depending on the
  spread of true D actually present in a given batch of tracks.
  
- **Negative D / negative intercept (classic pitfalls #2-3) can't happen by
  construction.** D (or D_alpha) and sigma have LogNormal priors/support
  (always positive) and alpha has Beta-rescaled support in (0, 2).
  
  **A short-track boundary-degeneracy pathology was found with the exact single-track MAP engine (`inference.fit_map`) under a flat prior, and an informative prior fixed it -- but the batched mean-field SVI engine now used for the full per-track table (`inference.fit_table_svi`) doesn't reproduce the same pathology even under a flat prior**, on the same
  controlled test (150 simulated tracks at track_length=10/12/15, true
  D=0.05, alpha=1): exact MAP showed 6.0-9.3% of the shortest tracks running
  to the edge of parameter support (D -> ~0 or alpha -> the 0/2 boundary)
  under a flat prior, 0.0% under an informative prior; the batched SVI
  engine shows 0.0% under both. The likely explanation is that SVI's
  finite-step stochastic (Adam) optimization doesn't converge all the way to
  the same boundary-hugging optimum an exact gradient-based optimizer
  reaches, rather than the flat-prior posterior itself being different --
  worth re-checking if the SVI step budget or optimizer changes, since this
  means the current implementation's apparent robustness on short tracks may
  be partly an optimization-convergence effect, not purely a prior effect.
  
- **Alpha recovery across genuine sub-/super-diffusive ground truth** (not
  tested by the classic pipeline's validation, which only ever simulated
  pure Brownian motion): sweeping true alpha from 0.5 to 1.8 at fixed true
  D_alpha=0.05, flat prior, batched SVI: median fitted alpha tracks true
  alpha reasonably well near 1 (e.g. 0.997 at true 0.9, 1.078 at true 1.0)
  but with a real, systematic bias that grows toward the sub-diffusive end
  -- 0.66 at true 0.5 (+0.16), 0.83 at true 0.7 (+0.13) -- and a smaller
  bias at the super-diffusive end (1.75 at true 1.8, -0.05). Plausible
  mechanism: sub-diffusive (anti-persistent) fBm increments and the
  localization-noise term are both negatively autocorrelated at lag 1, so
  the two are harder to tell apart at low alpha, biasing fitted alpha back
  toward 1 (`results/figures/bayes_validate_alpha_recovery.png`). This is a
  real limitation to keep in mind for strongly sub-diffusive tracks, not
  just a validation of the method.
  
- **Full NUTS reveals real, non-Gaussian posterior structure a point
  estimate + stderr misses.** On a 200-frame track (particle 88 in this
  run) the batch fit matches the NUTS posterior median closely. On a
  43-frame track (particle 2948), the NUTS posterior has a genuine heavy
  right tail toward large D_alpha as alpha approaches its upper bound of 2
  -- a real weak-identifiability ridge for short/noisy tracks (near-ballistic
  motion is hard to distinguish from a large diffusion coefficient over few
  points), not a sampler artifact: the chain traces are stationary and
  well-mixed (`results/figures/bayes_trace_particle2948.png`), they just
  repeatedly visit that ridge.
  
- **Agreement with the classic per-track fit is only moderate**
  (r(D_classic, D_bayes)=0.45, r(alpha_classic, alpha_bayes)=0.68) --
  expected: the two estimators use different information (a short-lag MSD
  curve vs. the full displacement likelihood), and the classic alpha in
  particular carries the D-dependent bias documented above.
  
- Ensemble-level estimates land close to the classic pipeline's (per-track
  median D: classic 0.053, this pipeline's normal model 0.046 um^2/s; median
  alpha: classic 0.997, this pipeline's anomalous model 1.125) -- consistent
  with the same underlying physics. A number closer to 1 doesn't by itself
  mean an estimator is less biased -- see the null-simulation result above.

### Inference-engine choice for production: Laplace/L-BFGS-B over SVI/Adam

The batched-SVI engine (`inference.fit_batch_svi`, mean-field `AutoNormal`,
Adam-optimized ELBO -- the "Batched inference performance" section above)
was the first working batched approach and is what `fit_table_svi` still
runs today. Follow-up checks comparing it against exact per-track/per-group
optimization (`inference.fit_map`, L-BFGS-B on numpyro's own unconstrained
`potential_fn` with an exact JAX Hessian) found consistent, material
advantages for L-BFGS-B, motivating a switch for production:

- **Point-estimate accuracy.** On a 28-track synthetic recovery test (7 true
  alpha values 0.5-1.8, 4 replicates each, flat prior), L-BFGS-B was closer
  to true alpha at every value except one (e.g. true alpha=1.0: L-BFGS-B
  bias -0.078 vs. SVI/Adam bias +0.148; true alpha=0.7: -0.038 vs. +0.070).
  It was also faster at this scale (11.7s vs. 29.0s for the whole group).
- **Uncertainty calibration.** A 3-track study (particles 88/200 frames,
  1921/12 frames, 2948/43 frames; anomalous model) compared NUTS (reference)
  against Laplace/L-BFGS-B, SVI mean-field (`AutoNormal`), and SVI
  full-covariance (`AutoMultivariateNormal`), measuring each method's
  posterior mean bias (in units of NUTS's own std) and stderr ratio
  (method/NUTS): **SVI mean-field's reported uncertainty was 3-10x too
  narrow** (alpha stderr ratio 0.30-0.36x true; D_alpha stderr ratio as low
  as 0.10x) on all three tracks, because the mean-field independence
  assumption throws away real posterior correlation between D_alpha, sigma,
  and alpha that carries much of the actual uncertainty. SVI full-covariance
  fixed most of this (alpha stderr ratio 0.87-0.95x) at comparable cost.
  Laplace/L-BFGS-B was the best-calibrated for alpha specifically (stderr
  ratio 0.99-1.04x on all three tracks) and had the lowest mean bias in most
  rows, though it underestimates D_alpha's spread on the shortest track (see
  next section for why).

  Full numeric table and setup: `/tmp/posterior_study.log` from this
  session (not preserved in the repo); representative corner plots in
  `results/figures/laplace_vs_nuts_particle{88,1921,2948}.png` (NUTS samples
  with the Laplace mean+covariance ellipse overlaid).

- **Scaling limitation.** L-BFGS-B's accuracy comes from an exact Hessian
  computed over *all* free parameters in a batched group at once (`jax.hessian`
  on the full flattened parameter vector) -- dense, so its cost is
  superlinear in group size: 10 tracks ~6s, 20 ~8s, 40 ~27s, and 140 tracks
  **OOM-crashed** the process. Production batching therefore needs a
  conservative per-call track cap (looping over sub-batches within each
  track_length group and appending results, rather than one call per group
  regardless of size) -- unlike `fit_batch_svi`, which had no such ceiling.
  A proper fix would replace the dense Hessian with a block-diagonal one
  (exploiting that tracks are independent given the plate, e.g. via
  `jax.vmap` over per-track Hessian blocks) -- not yet implemented.

### Production deployment: `fit_table_map`, log-space, asymmetric intervals

`diffusionkit.bayes.inference.fit_table_map` implements the production decision from the
sections above: batched exact MAP via L-BFGS-B, sub-batched at
`max_batch_size` (default 20) tracks per `fit_map` call to stay within the
dense-Hessian memory limit, with each track's D (or D_alpha, sigma) reported
via its log-space Laplace fit as an asymmetric interval
(`{name}_median`/`_lo`/`_hi` in physical units from `exp(logD_mean -+ logD_stderr)`,
`log10_{name}`/`_stderr` alongside) rather than a symmetric mean +/- stderr
in linear units. alpha keeps a symmetric physical-space interval. D (normal
model) and alpha (anomalous model) are the primary per-particle metrics;
D_alpha (anomalous model) is retained as a secondary/diagnostic column.
`scripts/run_bayes_analysis.py` runs this in production; `fit_table_svi`
(SVI/Adam) remains for comparison/validation use.

Full real-dataset run (365 tracks, both models + a flat-prior comparison
fit, `max_batch_size=20`):

- **Correctness**: results match a small-scale (30-track) test exactly,
  and match the earlier single-track-loop `fit_map` run's numbers closely
  (as expected -- batched joint MAP of independent tracks is mathematically
  the per-track MAP done separately, just faster to compute). All 365 rows
  have `D_lo <= D_median <= D_hi` and strictly positive D; zero L-BFGS-B
  non-convergence flags across all sub-batches.
- **Headline numbers**: log10(D) median -1.34 (D median 0.045 um^2/s, IQR
  [0.040, 0.051]); alpha median 1.115 (IQR [1.007, 1.199]); D_alpha median
  0.065 um^2/s^a (secondary). r(D_normal, alpha_anomalous) [classic-
  comparable] = **-0.16** (vs. classic MSD's 0.73), consistent with the
  earlier SVI-based production run's -0.22 and the null-simulation result.
  r(D_alpha, alpha) [within-fit, not the artifact metric] = 0.76, matching
  the earlier real-data value.
- **Timing tradeoff, stated plainly**: this run took ~800s (normal model) +
  ~484s (anomalous model) = **~21.4 minutes** for the two batched fits
  alone (whole script, including the flat-prior comparison fit and NUTS on
  3 example tracks: ~26-27 minutes wall-clock). The earlier SVI/Adam
  (`fit_table_svi`) production run did the same two fits in **~9.4
  minutes**. `fit_table_map` is *slower in aggregate* than SVI at full
  dataset scale despite being faster *and* more accurate per sub-batch in
  isolated tests (e.g. 11.7s vs. 29.0s for one 28-track group) -- the
  `max_batch_size` cap forces many more, smaller L-BFGS-B/Hessian calls than
  `fit_table_svi`'s one-call-per-length-group (e.g. the largest real
  length-group, 37 tracks, needs 2 sub-batches instead of 1), and each call
  pays its own JAX trace. This was an accepted, explicit tradeoff (better
  calibration and point-estimate accuracy over raw throughput) rather than
  an oversight -- worth revisiting via the block-diagonal-Hessian fix noted
  above if the ~3x slowdown becomes a real constraint.
- Progress visibility: `fit_table_map`/`fit_table_svi` gained a `tqdm`
  progress bar (`show_progress=True` by default) after this run was
  already in flight -- this run itself had none, but future runs will show
  live per-track progress rather than fully-buffered output that only
  appears at the end (relevant for backgrounded/redirected runs).

### D's posterior is much better-behaved than D_alpha on short tracks

Repeating the NUTS-vs-Laplace comparison on real `track_length=5` tracks
(n_disp=4, the shortest in the dataset; particles 107, 112, 143), for both
the normal (D, sigma) and anomalous (D_alpha, sigma, alpha) models:

- **Normal model (D):** Laplace's D estimate is biased low by roughly 25%
  and its stderr is roughly 40-45% too narrow vs. NUTS across all three
  particles -- real degradation, but the posterior stays close to unimodal.
- **Anomalous model (D_alpha):** far worse. Laplace's mean is **2-3x too
  low** and its stderr **4-6x too narrow** on all three particles (e.g.
  particle 107: NUTS D_alpha = 0.170 +/- 0.686, Laplace says 0.073 +/- 0.113
  -- the true 1-sigma NUTS interval is wider than Laplace's entire reported
  range). Cause, visible directly in the corner plot: with only 4
  displacements and 3 free parameters, large D_alpha paired with alpha near
  its upper boundary (2) explains the data almost as well as small D_alpha
  with moderate alpha, producing a long, heavy right tail in D_alpha that a
  Gaussian approximation cannot represent. Alpha's own marginal is *still*
  reasonably matched by Laplace even here -- it's specifically D_alpha (and
  its coupling to alpha) that breaks down.

  This is on top of the classic pipeline's own well-known difficulty
  estimating anything from very short tracks, and directly motivates
  reporting D (not D_alpha) as the primary diffusivity metric, with D_alpha
  kept as a secondary/diagnostic quantity rather than a headline number.
  Figures: `results/figures/laplace_vs_nuts_n5_particle{107,112,143}_{normal,anomalous}.png`.

### D should be reported in log-space, with an asymmetric interval

numpyro's default unconstraining transform for a positive-real
(`LogNormal`-supported) site is `ExpTransform` -- verified directly
(`biject_to(dist.LogNormal(...).support)` returns an `ExpTransform`, and its
inverse applied to a sample equals `log` of that sample to floating-point
precision). This means `inference.fit_map`'s L-BFGS-B optimization is
**already** running in log(D) space internally; the only question was
whether to also report results there, instead of pushing the Laplace
covariance forward through `exp()` via the delta method (as `MAPFit.cov`/
`stderr` did) and reporting a symmetric interval in linear D.

Comparing NUTS's D marginal (linear) against its log(D) marginal, on the
same three tracks used above (particles 107/n_disp=4, 1921/n_disp=11,
88/n_disp=199), for both models:

| track (n_disp) | model | linear-space skewness | log-space skewness |
|---|---|---|---|
| 107 (4) | normal | 5.80 | -1.69 |
| 107 (4) | anomalous | 18.96 | -1.07 |
| 1921 (11) | normal | 1.54 | 0.08 |
| 1921 (11) | anomalous | 8.42 | 0.44 |
| 88 (199) | normal | 0.29 | 0.075 |
| 88 (199) | anomalous | 1.05 | 0.236 |

Linear-space skewness blows up as tracks get shorter (up to 19 at n_disp=4);
log-space skewness stays roughly bounded across the whole range, even
though it is not perfectly zero at n_disp=4 either (nothing will be, with 4
data points) -- see `results/figures/logspace_check_particle{107,1921,88}_{normal,anomalous}.png`
for the full NUTS-histogram-vs-Laplace-curve comparison in both spaces side
by side. At particle 107 (n_disp=4, normal model), for instance, linear
space shows an unresolvable spike near D=0 with a long invisible tail past
2.5 um^2/s that the linear Laplace curve cannot represent at all; log space
shows a recognizable, left-skewed-but-unimodal bump that the log-space
Laplace curve visibly tracks.

**Decision:** production reports D via its log-space Laplace fit
(`MAPFit.unconstrained_params`/`unconstrained_cov`, added to `inference.py`
specifically for this -- previously computed internally as an intermediate
step and then discarded), back-transformed to an **asymmetric** interval in
physical units (`exp(logD_mean - logD_stderr)`, `exp(logD_mean)`,
`exp(logD_mean + logD_stderr)`) rather than a symmetric `D +/- stderr(D)`,
which the skewness numbers above show is not a meaningful description of
the uncertainty for short tracks. Same treatment applies to D_alpha and to
sigma (also `LogNormal`-supported), for whatever secondary/diagnostic value
they retain per the previous section.

## Anisotropy detection (`diffusionkit.bayes.anisotropic_diffusion_model`, N=5-10 tracks)

A Bayesian anisotropy score was drafted externally (Gemini) as a
polar-turning-angle Bayes-factor scheme; reviewed and replaced with a direct
extension of the existing exact-likelihood MVN machinery instead: H1 is a
rotated anisotropic diffusion tensor (`D_mean`, `eps` in [0,1) =
(D_par-D_perp)/(D_par+D_perp), orientation `psi`) at alpha=1, with H0 nested
exactly at `eps=0` (verified: `anisotropic_displacement_covariance(eps=0,
any psi)` reduces bit-for-bit to the isotropic `displacement_covariance`,
`scripts/validate_anisotropy_recovery.py`'s `_reduction_sanity_check`). This
replaces the draft's separately-constructed, undifferentiated H0/H1 pair and
its underived turning-angle "noise convolution" step with one generative
model the way `normal_diffusion_model`/`anomalous_diffusion_model` already
relate. Output is a posterior credible interval on `eps` (median + HPDI),
not a Bayes-factor point score -- see the project's anisotropy-detection
plan for the full critique/design rationale.

### The real obstacle is a sampling-noise floor, not localization noise

The Gemini draft's stated concern (Section 6.1) was localization noise
creating spurious apparent anisotropy. `validate_anisotropy_recovery.py`
found a **larger and more fundamental** problem: even with *zero*
localization noise, the sample eigenvalue ratio of a genuinely isotropic 2D
Gaussian cloud is enormous at n=4 points (n_disp for a 5-frame track):

| n_disp (track_length) | median eigval ratio | p90 eigval ratio | median "apparent" eps | p90 "apparent" eps |
|---|---|---|---|---|
| 4 (N=5) | 5.8 | 37.0 | 0.41 | 0.72 |
| 9 (N=10) | 2.5 | 5.6 | 0.22 | 0.40 |
| 19 (N=20) | 1.8 | 2.9 | 0.14 | 0.26 |
| 49 (N=50) | 1.4 | 1.9 | 0.09 | 0.16 |

(computed directly from n iid draws of a unit isotropic 2D Gaussian, no
model/noise/Bayes involved -- pure sampling variability of the eigenvalue
ratio of a tiny empirical covariance matrix.) At N=5, a **majority** of
genuinely isotropic tracks have a raw eccentricity that would read as
"eps=0.4-ish" if taken at face value. This is the quantitative version of
the draft's own opening claim ("isotropic 2D random walks frequently appear
elongated over short step sequences") -- it was correct, just never
measured, and it turns out to dominate localization noise as a source of
spurious apparent anisotropy at this track length.

Consequence, from `check_1_null_false_positive_rate` /
`check_2_eps_recovery_and_psi_ridge` (N=5, D_mean=0.05 um^2/s, dt=0.033s,
sigma_loc 0.01-0.05 um): sweeping the `eps` prior's shrinkage strength
(`AnisotropicModelPrior(eps_a=1, eps_b=...)`) trades false-positive rate
against sensitivity almost as a step function, not a gentle dial --

| eps_b | prior mean(eps) | false-positive rate @ true eps=0 | detection rate @ true eps=0.6 |
|---|---|---|---|
| 1.0 | 0.50 | 100% | 100% |
| 1.5 | 0.40 | 94% | 100% |
| 2.0 | 0.33 | 32% | 54% |
| 3.0 (shipped default) | 0.25 | 0% | 0% |
| 5.0 | 0.17 | 0% | 0% |

("false-positive rate" = fraction of true-eps=0 tracks with posterior
median eps > 0.3; "detection rate" = fraction of true-eps=0.6 tracks with
P(eps>0.3\|data) > 0.5; both are simple operational thresholds, not the
recommended usage -- see below.) There is no eps_b that gives both a low
false-positive rate and useful sensitivity at N=5 on a **single track's own
data alone** -- the shrinkage prior strong enough to suppress the sampling
noise floor is also strong enough to swamp real signal up to eps=0.6, and a
prior weak enough to detect eps=0.6 also flags the majority of isotropic
tracks. Interval coverage checks confirm this isn't just a threshold-metric
artifact: even at eps_b=1.5, nominal 50/80/95% HPDI coverage of the true eps
came out at 27%/58%/80% (should equal the nominal level) -- the posterior
genuinely isn't tracking the per-track truth at this N, for any prior
strength tried.

**This is the correct, honest thing for the model to do** -- it is not a
bug in the covariance/likelihood (confirmed via the reduction check and via
directly recovering eps from simulated data with negligible noise and many
replicates pooled). A single 5-frame track does not contain enough
information to reliably separate real anisotropy up to eps~0.6 from pure
sampling noise, and a Bayesian model that reported a narrow, confident
interval anyway would be *wrong*, not more sensitive. This is the
concrete argument -- not just the a priori one in the design plan -- for why
per-track-only inference (this iteration's scope) has limited standalone
value at N=5, and why hierarchical/population-level pooling of `eps` (and
`D_mean`) across tracks is the necessary next step, not an optional
refinement: pooling is what lets the ensemble's information add up faster
than any single 4-point track's sampling floor. Until that exists,
`eps`'s posterior interval should be read/reported per-track as-is (wide
intervals at N=5 are the honest answer, not a defect to fix by tightening
the prior), not collapsed into a flag/threshold call the way the numbers
above were only for characterizing the trade-off.

Reduction check, false-positive/coverage tables, and recovery scatter plots
reproducible via `scripts/validate_anisotropy_recovery.py`
(`results/tables/validate_anisotropy_recovery/`,
`results/figures/validate_anisotropy_recovery/`).

### Reframing fixes it: ask "more anisotropic than free diffusion at this N", not "what is eps"

Point/interval estimation of `eps` asks the data to pin down a continuous
value; the sampling-noise floor above shows 4-9 displacements usually can't.
Model *comparison* -- "is this track's data more consistent with some
anisotropy than with none" -- is a different, better-posed question at this
N: it doesn't need to resolve *how much* anisotropy, and a correctly
constructed Bayes factor automatically weighs in how much apparent
elongation is expected from sampling noise alone at this exact n_disp/D/sigma
(no separately-simulated reference distribution needed -- that expectation
falls directly out of integrating the isotropic model's own likelihood over
its prior). Implemented as `diffusionkit.bayes.bayes_factor.log_bayes_factor_anisotropy`
/ `batched_log_bayes_factor_anisotropy`: prior-predictive Monte Carlo
estimate of `p(data|H1)` and `p(data|H0)` (H0 = H1 with `eps` pinned to 0),
sharing Monte Carlo draws of `D_mean`/`sigma` between the two integrals
(common random numbers) to cancel most of their variance out of the ratio.
Chosen over a Savage-Dickey ratio off the NUTS posterior specifically
because `eps=0` is a boundary point that no continuous posterior sample ever
lands on exactly -- direct marginal-likelihood MC needs no density
estimation there at all. This only works because the likelihood is weak
relative to the prior at N=5-10 (confirmed above); a much longer, more
informative track would need a smarter estimator (bridge/nested sampling).

`scripts/validate_anisotropy_recovery.py` checks 3-4 (D_mean=0.05 um^2/s,
dt=0.033s, n_mc=20000, `AnisotropicModelPrior()` default):

- **Check 3 (null calibration, true eps=0):** median/mean logBF10 stays at
  or below 0 at every track length (5/7/10) and both sigma regimes tested
  (0.01, 0.05 um) -- e.g. track_length=5: median -0.008 (low sigma), -0.014
  (high sigma); track_length=10: median -0.042/-0.052. 0% of tracks reach
  even "moderate" evidence (logBF10>1.1) for anisotropy on truly isotropic
  data, at any condition tested. No false-positive inflation, unlike the raw
  `eps` threshold rule in checks 1-2.

- **Check 4 (ensemble aggregation, track_length=5):** individual-track
  logBF10 is small regardless of true eps (consistent with check 3 -- a
  single 5-frame track rarely reaches decisive evidence on its own, honestly
  reflecting how little one track can say). But summing logBF10 across M
  independent tracks sharing the same true (eps, psi) behaves exactly like
  proper Bayesian evidence accumulation should:

  | true eps | M=10 (median) | M=30 (median) | M=100 (median) |
  |---|---|---|---|
  | 0.0 | -0.06 | -0.14 | -0.46 |
  | 0.3 | -0.01 | -0.04 | -0.15 |
  | 0.6 | +0.17 | +0.47 | +1.55 |
  | 0.8 | +0.35 | +1.07 | +3.69 |

  Isotropic (eps=0) and mildly-anisotropic (eps=0.3) populations stay flat
  or trend *more negative* as M grows (expected: the log BF's expectation
  under a true H0 is `-KL(p0||p1) <= 0`, so summing more evidence for a true
  null drives the total down, not up -- no risk of manufacturing false
  significance by pooling more tracks). Genuinely anisotropic populations
  (eps=0.6, 0.8) cross into "moderate" (>1.1) evidence by M~30 and approach
  "strong"/"decisive" (>2.3-4.6) by M~100.

**Practical recommendation:** report both outputs, for different questions.
`eps`'s posterior interval (checks 1-2) describes one track's own
diffusivity tensor, honestly wide at N=5-10 -- useful as a rough per-track
descriptive summary, not a detector. `log_bayes_factor_anisotropy`
(checks 3-4) answers "is there evidence of anisotropy" -- per track it will
almost always read "inconclusive" at this N (correctly, not a bug), but
summed across a population of tracks expected to share the same anisotropic
behavior (e.g. all tracks from one condition/structure), it is a
well-calibrated, genuinely sensitive detector. This is the corrected version
of the original Gemini draft's own instinct (a Bayes-factor "score",
aggregated via sec. 5.3's log-sum-across-tracks) -- valid here specifically
because H0 and H1 share one generative model and one well-specified prior,
which the draft's version did not have.

### Real-data anisotropy check (`mobile_beads_1to200.csv`, track_length 5-10)

`scripts/run_anisotropy_analysis.py`. 185/539 tracks fall in the N=5-10
target range. These are freely-diffusing beads, so this doubles as a
real-data negative control for everything validated on simulation above.

- **Individual tracks behave exactly as predicted**: per-track log BF10
  median -0.0005, mean +0.019, IQR [-0.058, +0.083]; 0% reach even
  "moderate" evidence (>1.1) alone. Matches the simulated null-calibration
  result -- no single short track is ever overconfident.
- **The ensemble sum does not**: summed across all 185 tracks, log BF10 =
  **+3.54** -- "strong" on the generic Jeffreys scale. Stable across MC
  seed (0-3) and n_mc (20000/60000): range +3.26 to +3.54, so not Monte
  Carlo noise.
- **Calibrated against its own null, this holds up as a real excess, not
  sampling luck.** The generic Jeffreys buckets say nothing about how much
  an M=185-track sum can fluctuate by chance -- calibrated instead against
  200 simulated isotropic replicates at the *exact* real composition
  (39/49/32/24/30/11 tracks at track_length 5/6/7/8/9/10, D_mean=0.05,
  sigma_loc=0.025 from the prior's own central values): null median -1.71,
  p90 +0.33, p99 +1.94, max +3.46 across all 200 replicates. The observed
  +3.54 exceeds every one of them (empirical p<0.005).
- **A persistent instrumental x/y asymmetry (astigmatism, non-square
  pixels) was checked and ruled out**: pooling the *full* dataset (539
  tracks, all lengths, ~25700 displacements -- far more statistical power)
  gives var(dy)/var(dx) = 1.0004, essentially perfectly balanced. A real
  optical/camera effect should show up there too and doesn't. The
  short-track subset alone shows var(dy)/var(dx) = 1.11 pooled, and 69% of
  its track_length=7 tracks individually lean positive (not a couple of
  outliers driving the sum -- top 3 of 32 tracks account for only 1.1 of
  that subgroup's +2.54).
- **Interpretation, held loosely: this is not evidence the beads are
  physically anisotropic.** It's evidence that this particular short-track
  subset deviates from the null model in a way neither Monte Carlo noise
  nor matched-composition sampling variance explains, and neither does a
  simple optics artifact. Plausible untested mechanisms: (a)
  motion-direction-dependent track-length censoring (a known SPT effect --
  particles that move far/fast in some direction during a window may be
  disproportionately likely to leave the tracking gate in that direction,
  which correlates *why a track is short* with its displacement direction
  in a way this model has no way to distinguish from real anisotropy); (b)
  unmodeled transient/correlated noise specific to short acquisition
  windows (stage/vibration); (c) the short-track subset isn't a random
  draw from the full population, and whatever caused early loss might
  correlate with something else. None were testable with the metadata
  available here (no acquisition timestamps or stage log in this CSV).
- **Practical takeaway**: the method performed as designed -- never
  overconfident on any individual track, and sensitive enough at the
  ensemble level to surface a real, calibration-confirmed deviation that
  per-track methods would have missed entirely. That sensitivity is exactly
  the point of the ensemble-aggregation piece; it just surfaced a data
  question (why is this specific subset non-isotropic) rather than
  confirming the expected null outright. Worth a follow-up with more
  metadata before drawing any physical conclusion.

Reproducible via `scripts/run_anisotropy_analysis.py`
(`results/tables/anisotropy/`, `results/figures/anisotropy/`).

### Visual inspection: mapping per-track outputs back onto the real tracks

Same script, extended to also fit each track's `eps`/`psi` posterior
(`inference.fit_table_nuts`, NUTS) and join it onto `log_bf10` in
one `per_track_master.csv` (`particle`, `track_length`, `x_mean_um`/
`y_mean_um`, `log_bf10`, `eps_median`/`_lo`/`_hi`, `psi_median_rad`/`_lo_rad`/
`_hi_rad`, `D_mean_median_um2_s`, `D_par_median_um2_s`,
`D_perp_median_um2_s` -- see README's column reference). Every quantity keeps
its own explicit name (`log_bf10` vs. `eps_median`) specifically so the two
methods' outputs are never mistaken for each other -- they answer different
questions and, per the checks above, have very different reliability at this
N.

- **`plot_trajectory_gallery`** (top-9 by `log_bf10` vs. a random-9 sample,
  `trajectory_gallery_top_logbf.png`/`_random.png`): the top-log_bf10 tracks
  visibly look more directional/elongated by eye (e.g. particles 2987, 1903,
  3678 trace an almost straight diagonal) than the random sample (more
  wandering, e.g. particles 62, 4102) -- a useful sanity check that the score
  tracks something visually real, not just a number.
- **`plot_eps_vs_log_bf`** (`eps_vs_log_bf_scatter.png`): `eps_median` and
  `log_bf10` are strongly monotonically related (expected -- both come from
  the same underlying data), but every point's 90% HPDI is wide and crosses
  `eps=0` except the single most extreme track -- the plotted picture of
  exactly the "eps alone has limited per-track power, log_bf10 is the piece
  to trust" finding above, not just a claim about it.
- **`plot_eps_forest`** (`eps_forest_top30.png`): confirms the same point for
  the 30 tracks driving the ensemble result specifically -- all 30 have a 90%
  HPDI upper bound near 0.6, i.e. even the tracks `log_bf10` is most
  confident about remain individually honest/wide on `eps`.
- **`plot_spatial_map`** (`spatial_map_log_bf.png`, diverging red/blue
  centered at 0; `spatial_map_eps.png`, sequential): no obvious large-scale
  spatial gradient or clustering across the ~55x55 um field of view for
  either quantity -- but the two single most extreme tracks by both measures
  (particles 2987 and 1903, the top two in the gallery above) sit right next
  to each other spatially (~20,17 um). n=2 is not evidence of a real
  region-specific effect on its own, but it's a concrete, checkable lead for
  the follow-up this section already flagged (motion-direction-dependent
  censoring, acquisition-window effects) rather than a settled finding.

Adding a new per-particle label column (e.g. once a spatial/structural
classification exists) is a join onto `per_track_master.csv` plus
`diffusionkit.bayes.anisotropy.aggregate_log_bayes_factor(..., label_col=...)` -- no new plotting or
fitting code needed; `plot_spatial_map`/`plot_trajectory_gallery` already
take any table with the right column names.
