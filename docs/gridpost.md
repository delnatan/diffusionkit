# Grid posteriors: exact-likelihood D and alpha, no MSD curve

`diffusionkit.gridpost` is a separate, independent measurement of the same
tracks `diffusionkit.classic` fits MSD curves for -- not a replacement or an
uncertainty estimate bolted onto the MSD fits. It fits the exact Gaussian
displacement likelihood directly, on a fixed numerical grid, with no MCMC
and no MSD curve anywhere.

## D: grid posterior over the displacement likelihood (no lag window)

`diffusionkit.gridpost.posterior` uses every consecutive displacement once,
with each frame's localization SD -- the same exact Gaussian displacement
likelihood a maximum-likelihood point estimate would maximize, but reported
as a posterior rather than collapsed to a point. Per axis, the m = n-1
displacements are Gaussian:

```
Sigma(D) = D A + B
A_ii = 2 dt (1 - 2R),  A_(i,i+1) = 2 dt R,  R = exposure_s / (6 dt)
B_ii = s_i² + s_(i+1)², B_(i,i+1) = -s_(i+1)²
```

A describes Brownian motion averaged over a continuous exposure (Berglund
2010, closed form); with `exposure_s=0` it is `2 dt I`. B describes
independent localization errors, which make neighboring displacements
negatively correlated.

Implementation: with B = L Lᵀ and L⁻¹AL⁻ᵀ = Q diag(λ) Qᵀ, the whitened data
y = QᵀL⁻¹Δ are independent with variances 1 + Dλ_k. This makes the
likelihood a cheap one-dimensional function of D, evaluated exactly on a
fixed grid in u = ln D (`GridPostOptions.u_D()`) rather than maximized: a prior flat
in u is log-uniform in D (scale-invariant), and the posterior weights are
`exp(ln L + ln prior)`, normalized. The production default is a flat prior
over the whole grid -- the least-informative choice, no empirical-Bayes
fitting across tracks. The grid's range, `[D_min_um2_s, D_max_um2_s]`
(default 1e-4 to 10 um^2/s, 501 points), is therefore the prior's support:
a posterior that has not died out by an edge is cut there, and its median
and interval move with the edge. The workflow flags such tracks in
`message` (`posterior.edge_ratios`); localization-limited, near-immobile
tracks reach the lower edge this way, since their data only bound D from
above. No module-level grid exists to fall back on: the workflow reads the
grids from `GridPostOptions`, and the lower-level functions take them as
required arguments.

`posterior.summary` reports the `(1-level)/2`/0.5/`(1+level)/2` quantiles of
the posterior (`D_post_lo_um2_s`, `D_post_median_um2_s`, `D_post_hi_um2_s`;
`GridPostOptions.level` defaults to 0.9) -- an equal-tailed credible interval
read directly off the CDF, not a multiple of a standard deviation (a
normal's 90% equal-tailed interval is +/-1.645 SD, not +/-1 SD, and these
posteriors are often far from normal on short tracks anyway). Localization
SDs must be strictly positive and are treated as known. Unlike a point
estimate, a short or noise-dominated track does not report a falsely
confident number: its posterior stays wide, which is the honest answer, not
a defect. `D_post_info_bits` (`posterior.information_bits`) puts a number on
that: the relative entropy from the flat prior to the posterior, in bits.
It is invariant to reparametrizing D but relative to the prior's range, so
it compares tracks and experiments only on the same grid; a
localization-limited track that only bounds D from above earns the bits of
the prior it rules out. Calibration is a simulation check under the model (see
`prototypes/README.md`), not a claim about experimental tracks.

The track summary also includes `D_motion_lrt`, available directly as
`gridpost.brownian_motion_lrt(track, acquisition)`. This compares a noise-only
covariance `c² B` against `D A + c² B`, fitting a global localization SD
multiplier `c > 0` in both hypotheses and `D >= 0` in the alternative. It is
twice the maximized log-likelihood difference (using the supremum when the
alternative approaches `c = 0`). Near zero means Brownian motion adds little
fit improvement beyond rescaling localization noise; larger values indicate
greater improvement. Failure to distinguish the models does not establish
immobility. This is a likelihood-ratio score, not a Bayes factor or a p-value;
short-track significance thresholds require simulation calibration, rather
than an ordinary chi-square cutoff at the `D = 0` boundary.

The comparison profiles out the common variance analytically in the existing
whitened coordinates, then scans and refines the remaining covariance-shape
parameter, including both limiting shapes. It is independent of the posterior
grid and prior, and invariant to a global scaling of positions or reported
localization SDs. It only tolerates a global noise-scale mismatch, not arbitrary
errors in temporal covariance or relative per-frame uncertainties. The D
posterior and its information bits still condition on the reported SDs (`c=1`);
they are not replaced by this separate comparison. Exactly zero displacements
give an unbounded likelihood as the scale tends to zero, so the score is null
and the summary message explains why. Excluded/invalid rows also have null
scores. The value appears on `posterior_D` rows in `analysis.fits`, and in
`result.posterior_D.parameters` for single-track analysis.

This module started as, and is adapted from, `prototypes/posterior_1d.py`
(a standalone reference implementation with its own extensive calibration
checks). A previous production estimator (`fit_brownian_mle`, a
maximum-likelihood D with a bootstrap-calibrated non-Brownian z-score
testing deviation from alpha=1) has been retired now that the posterior
supersedes its point-estimate role.

## alpha: grid posterior over the fBm exponent, K marginalized out

`diffusionkit.gridpost.posterior_alpha`, adapted from
`prototypes/posterior_alpha.py`, answers a different question from the D
posterior above: not "how big are the steps" (the alpha=1 model) but "how
are consecutive steps correlated". Per axis, the same m = n-1 displacements
are Gaussian under the fBm model:

```
Sigma(K, alpha) = K A(alpha) + B
A(alpha)_k = dt^alpha (|k+1|^alpha - 2|k|^alpha + |k-1|^alpha)   (lag k)
```

with B the same localization-noise covariance as the D posterior. For any
*fixed* alpha this is linear in K exactly as the D posterior's model is
linear in D, so the same whitening trick applies -- but A(alpha) itself
changes shape with alpha, so each alpha grid point (`GridPostOptions.alphas()`,
39 points by default) needs its own eigendecomposition, unlike the D posterior's single
decomposition reused across the whole D grid. This is a bounded, linear-in
-grid-size cost (a few milliseconds per track), not a bottleneck at the
track lengths this project targets.

The alpha posterior is the 1D marginal of the 2D (alpha, ln K) log-likelihood
surface, integrating the nuisance K out with its own (hand-set, not
empirical-Bayes) prior via `logsumexp` -- `posterior_alpha.summary` reports
the median and credible interval (`alpha_post_median`, `alpha_post_lo`,
`alpha_post_hi`). D and alpha are reported as two independent per-track
measurements, not a joint (K, alpha) fit: `D` comes from the alpha=1 model,
`alpha` from the fBm model with K integrated out entirely, which sidesteps
the well-known K/alpha MLE degeneracy rather than inheriting it.

No exposure-blur model exists at a general alpha (the D posterior's
closed-form Berglund R average is specific to alpha=1's linear-motion
double integral), so `acquisition.exposure_s` must be 0; nonzero exposure
excludes the alpha posterior. The D posterior itself still models blur and
is not excluded.

## Workflow: `gridpost.analyze_track`/`analyze_tracks`

`GridPostOptions(min_frames=3, level=.9)` sets the short-track exclusion
threshold (the whitening step's own hard minimum) and the credible-interval
mass; its grid fields (`D_min_um2_s`, `D_max_um2_s`, `n_D`, `alpha_min`,
`alpha_max`, `n_alpha`, and `n_K` for the nuisance K grid, which spans D's
range) set every grid the run evaluates, and `compute_alpha=False` skips the
alpha posterior. `analyze_tracks(..., keep_posteriors=True)` returns each
`ok` track's normalized log posterior too (`GridPosteriors`), for population
reads such as `deconvolve.deconvolve`. `analyze_track`/`analyze_tracks` mirror `classic`'s workflow contract:
invalid input raises for a single track, a batch keeps going and marks the
offending track `invalid_input`, and `status="excluded"` records *why* a
track wasn't fit rather than silently dropping it. See
[TABLES.md](../TABLES.md) for the `fits` schema and [WORKFLOW.md](../WORKFLOW.md)
for worked examples.
