# Classical core: model and estimator contract

Data classes describe observations and results. Functions perform validation,
MSD construction and fitting. No class owns an optimizer, worker, plotting
backend or file. Both fits receive the same `MSDCurve` and can be called
without the table workflow.

## Localization correction

Let observed position be r_i = X_i + e_i, with independent zero-mean errors
and per-axis variances s²_(i,x), s²_(i,y). For a lag l:

```
MSD_observed[l] = mean_i ||r_(i+l) - r_i||²
offset[l] = mean_i sum_axis(s²_i + s²_(i+l))
MSD_corrected[l] = MSD_observed[l] - offset[l]
```

This reduces to 4 sigma² for constant, isotropic errors. For errors that
vary across observations, the endpoint composition changes with lag, so a
single per-track average is not generally the right correction. Axis errors
need not be equal. Cross-frame error correlation is not modeled.

The SDs are inputs, treated as known. They are not a prior and the correction
does not estimate their calibration. Using fitted localization CRLBs as
these SDs is an assumption that must be checked against detector calibration.
`localization="ignore"` deliberately sets the correction to zero even if
SDs were supplied; its results describe the uncorrected curve.

## D: ordinary least squares with a known offset

For selected lag times tau_l and corrected values y_l:

```
D_hat = sum_l(tau_l * y_l) / (4 * sum_l(tau_l²))
```

There is no additional free intercept. The localization contribution has
already been removed using the known errors. Under a correct Brownian
mean model, known errors and a fixed lag window, this linear estimator has
the correct expectation, even though its MSD points are correlated. It is
not claimed to minimize variance. Negative estimates are possible and
retained; they do not demonstrate negative physical diffusivity.

The historical free-intercept fit is still available under legacy imports
for reproducing previous comparisons, not as an implicit fallback when
localization errors are missing.

## Alpha: constrained least squares in linear MSD space

The objective is `sum_l (y_l - 4 K tau_l**alpha)²`, with K >= 0 and
0 <= alpha <= 2. Internally the model is written as
`A * (tau/t_ref)**alpha`, where t_ref is the geometric mean of selected lag
times, and values are scaled by the largest absolute corrected MSD. These
are conditioning transformations, not priors or extra model parameters.

SciPy's [least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
runs from alpha 0.5, 1 and 1.5 with an analytic Jacobian. The lowest-cost
candidate is compared with exact optimal amplitudes at alpha=0 and alpha=2.
The result records optimizer status and flags boundaries, failed optimization,
and zero/negligible amplitudes or numerically ill-conditioned Jacobians.
The three starting points reduce sensitivity to initialization; they are
not a proof of finding every possible global minimum.

No logarithm of the measured MSD is taken. Negative corrected values remain
in the objective. If the corrected signal cannot identify an amplitude,
alpha and K are null rather than arbitrary numerical values. The power law
describes a mean curve; it does not identify an fBm mechanism, confinement,
drift or switching from its exponent alone.

## What this revision does not claim

- A calibrated uncertainty interval from correlated MSD regression residuals.
- An optimal lag cutoff or a universally preferable classical estimator.
- Reliable motion classification from five-frame alpha point estimates.
- Corrected estimates when supplied localization SDs are miscalibrated.
- A motion-blur, irregular-time, tracking-error or model-selection treatment.

Overlapping MSD pairs are correlated; `n_pairs` is a count, not an
independent sample size. The
[MSD analysis literature](https://pmc.ncbi.nlm.nih.gov/articles/PMC3055791/)
motivates treating weighting, covariance and lag selection as statistical
choices to validate, rather than equating an optimizer's residual error
with parameter uncertainty.

The current core deliberately reports no standard errors or intervals.
Model-based covariance or bootstrap intervals can be added after their
assumptions and short-track coverage are established, using these data and
estimator functions rather than another parallel workflow.

## D: grid posterior over the displacement likelihood (no lag window)

`diffusionkit.classic.posterior` uses every consecutive displacement once,
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

`posterior.summary` reports the median and an equal-tailed credible
interval (`D_post_median_um2_s`, `D_post_lo_um2_s`, `D_post_hi_um2_s`).
Localization SDs must be strictly positive and are treated as known. Unlike
a point estimate, a short or noise-dominated track does not report a
falsely confident number: its posterior stays wide, which is the honest
answer, not a defect. Calibration is a simulation check under the model
(see `prototypes/README.md`), not a claim about experimental tracks.

This module started as, and is adapted from, `prototypes/posterior_1d.py`
(a standalone reference implementation with its own extensive calibration
checks). A previous production estimator here (`fit_brownian_mle`, a
maximum-likelihood D with a bootstrap-calibrated non-Brownian z-score
testing deviation from alpha=1) has been retired now that the posterior
supersedes its point-estimate role; `prototypes/posterior_alpha.py` is an
active-research replacement for the z-score's role, reporting a full
(honest, possibly wide) posterior over alpha instead of a calibrated
hypothesis-test statistic. It is not yet wired into this module.
