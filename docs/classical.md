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

## Time-averaged and ensemble-averaged curves

The estimators below take an `MSDCurve` and do not care whether it is one
track's time average or an average over tracks. `compute_msd` makes the
time-averaged MSD of one track; `classic.ensemble_msd` averages those over
tracks (`weight="pairs"`, the standard estimator: all squared displacements
pooled; `"tracks"`: the mean of the TA-MSDs) and `EnsembleMSD.fit(n_points)`
fits the first `n_points` lags of the result.

## The fitting window

Both a track's curve and an ensemble curve get noisier at long lags, where few
and heavily overlapping pairs remain. `window_lags(n_available, fraction=.3)`
is the usual rule of thumb (the first 25-40% of the curve, never fewer than
three lags, never more than exist). Per track, `MSDOptions(max_lag=None,
lag_fraction=.3)` applies it to each track's own length, so long tracks use
more of their data. The window is a statistical choice that moves the
estimates; the ensemble fit therefore has no default for it.

## The textbook pair: linear fit, then log-log fit

`fit_linear_msd` fits MSD = 4 D tau + b to the raw curve with a free
intercept: D = slope / 4, and b is the localization offset (4 sigma^2 for an
isotropic 2D SD). `fit_loglog_msd(curve, offset_um2)` subtracts an offset and
regresses log(MSD - offset) on log(tau): alpha is the slope, 4K the
exponential of the intercept. They are separate functions of the same curve,
joined only by the number passed between them (`fit_textbook` is that
composition). The intercept is a two-parameter fit to a handful of correlated
points, so it is noisy on a single short track and well determined on an
ensemble curve. Motion blur and confinement also bend the curve at short lags
and move the intercept: it is an empirical offset, not a measurement of the SD.
A negative intercept is flagged `nonphysical`; `EnsembleMSD.fit` then subtracts
no offset.

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

The historical free-intercept fit is not part of this module; there is no
implicit fallback to it when localization errors are missing.

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

## Alpha in log-log space

Lags with a non-positive corrected MSD cannot be logged and are dropped
(counted in the message). That selection pushes alpha upward, the log of a
noisy mean is biased, and the points are correlated, so this is a cross-check
and an illustration; the constrained linear-space fit above stays the per-track
default.

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

The per-track fits deliberately report no standard errors or intervals. The
ensemble MSD over a batch (`classic.ensemble_msd`, see the README) does:
a cluster bootstrap over tracks or movies, because a track, not a pair, is the
unit that can be resampled as independent.
Model-based covariance or bootstrap intervals can be added after their
assumptions and short-track coverage are established, using these data and
estimator functions rather than another parallel workflow.

The exact-likelihood grid posterior over D is a separate,
independent measurement of the same tracks -- see
[docs/gridpost.md](gridpost.md), not this module.
