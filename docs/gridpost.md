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
fixed grid in u = ln D (`posterior.U`) rather than maximized: a prior flat
in u is log-uniform in D (scale-invariant), and the posterior weights are
`exp(ln L + ln prior)`, normalized. The production default is a flat prior
over the whole grid -- the least-informative choice, no empirical-Bayes
fitting across tracks.

`posterior.summary` reports the `(1-level)/2`/0.5/`(1+level)/2` quantiles of
the posterior (`D_post_lo_um2_s`, `D_post_median_um2_s`, `D_post_hi_um2_s`;
`GridPostOptions.level` defaults to 0.9) -- an equal-tailed credible interval
read directly off the CDF, not a multiple of a standard deviation (a
normal's 90% equal-tailed interval is +/-1.645 SD, not +/-1 SD, and these
posteriors are often far from normal on short tracks anyway). Localization
SDs must be strictly positive and are treated as known. Unlike a point
estimate, a short or noise-dominated track does not report a falsely
confident number: its posterior stays wide, which is the honest answer, not
a defect. Calibration is a simulation check under the model (see
`prototypes/README.md`), not a claim about experimental tracks.

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
changes shape with alpha, so each alpha grid point (`posterior_alpha.ALPHA`,
39 points) needs its own eigendecomposition, unlike the D posterior's single
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
mass. `analyze_track`/`analyze_tracks` mirror `classic`'s workflow contract:
invalid input raises for a single track, a batch keeps going and marks the
offending track `invalid_input`, and `status="excluded"` records *why* a
track wasn't fit rather than silently dropping it. See
[TABLES.md](../TABLES.md) for the `fits` schema and [WORKFLOW.md](../WORKFLOW.md)
for worked examples.
