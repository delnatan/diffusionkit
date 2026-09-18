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

## D: Brownian displacement maximum likelihood (no lag window)

`fit_brownian_mle` uses every consecutive displacement once, with each
frame's localization SD. Per axis, the m = n-1 displacements are Gaussian:

```
Sigma(D) = D A + B
A_ii = 2 dt (1 - 2R),  A_(i,i+1) = 2 dt R,  R = exposure_s / (6 dt)
B_ii = s_i² + s_(i+1)², B_(i,i+1) = -s_(i+1)²
```

A describes Brownian motion averaged over a continuous exposure (Berglund
2010); with `exposure_s=0` it is `2 dt I`. B describes independent
localization errors, which make neighboring displacements negatively
correlated. D is maximized over `D >= 0`. The result is the constrained
maximum. Unlike the MSD fit, the MLE is never negative. `D_hat = 0` is a
boundary result meaning the localization errors alone explain the
displacements. The status is then `unresolved`, which is not the same as
immobile: a particle confined to a region smaller than the localization
error looks the same.

- `lr_motion = 2[l(D_hat) - l(0)]` and `p_motion` use the asymptotic
  boundary reference ½χ²₀ + ½χ²₁.
- `D_upper_um2_s` is a one-sided profile-likelihood limit at `upper_level`
  (default 0.95), using the same boundary-aware threshold.

Both are asymptotic. Their short-track calibration is measured in
`audit/brownian_mle_validation.json`, not assumed. Localization SDs must be
strictly positive and are treated as known.

Implementation: with B = L Lᵀ and L⁻¹AL⁻ᵀ = Q diag(λ) Qᵀ, the whitened data
y = QᵀL⁻¹Δ are independent with variances 1 + Dλ_k. This makes the
likelihood a cheap one-dimensional function of D. It is maximized by a
log-grid search, golden-section refinement, and an explicit comparison with
D = 0.

## Non-Brownian score: a calibrated second axis

Fitted alpha is a poor second axis for short tracks, because its sampling
spread depends strongly on N and on D dt / s². `z_nonbrownian` instead asks
one question per track: at the Brownian fit, does moving the fBm exponent
away from alpha = 1 improve the likelihood more than chance would?

With G(K, alpha) the fBm displacement covariance (exposure-averaged in the
same way as A) and H = ∂G/∂alpha at alpha = 1, per unit K:

```
U_alpha = ½ Σ_axis [Δᵀ Σ⁻¹ (D H) Σ⁻¹ Δ - tr(Σ⁻¹ D H)]
U       = U_alpha - (I_aD / I_DD) U_D                    (efficient score)
I_eff   = I_aa - I_aD² / I_DD,   I_jk = ½ Σ_axis tr(Σ⁻¹ Σ_j Σ⁻¹ Σ_k)
z_nonbrownian_asymptotic = U / sqrt(I_eff)
alpha_1step = 1 + U / I_eff,   alpha_1step_se = 1 / sqrt(I_eff)
```

Projecting out the D direction removes the ln(dt) part of ∂/∂alpha, so the
score does not depend on time units. To first order it is also
uncorrelated with D_hat. The score is dominated by the excess lag-1
displacement covariance, beyond the -s² that localization error predicts.

**Calibration.** The asymptotic normal reference is unreliable at 4–19
displacements. For each resolved track, `n_boot` (default 500) replicates
are drawn from the fitted Brownian model with that track's own length,
localization SDs, exposure and D_hat. Each replicate is refit and scored.
`z_nonbrownian = Φ⁻¹(p)` uses the mid-rank p-value among replicates that
also resolve motion, and `p_nonbrownian` is two-sided. Under the Brownian
model with correct SDs, z_nonbrownian is therefore approximately N(0,1) for
every track, whatever its N, localization error or D. The bootstrap RNG is
seeded from `(MLEOptions.seed, track_id)`, so results do not depend on track
order.

**Interpretation.** z < 0 means antipersistent, and z > 0 means persistent.
The statistic measures deviation from *this* observation model, not a
mechanism:

| z < 0 can come from | z > 0 can come from |
| --- | --- |
| subdiffusion, confinement | drift, directed transport |
| localization SDs underestimated | localization SDs overestimated |
| linking errors (jump and return) | unmodeled exposure blur |

`alpha_1step` is one Newton step from the Brownian fit toward the fBm MLE.
It is an effect-size companion that cannot hit the 0 or 2 boundaries, but
it is as noisy as the information in the track allows. At D_hat = 0 the
score is identically zero, so it is not reported.

**What this does not do.** It does not beat the information limit. At 5
frames most individual z values are noise, and no statistic classifies
single tracks there. The gain is a common, calibrated null for every
track. Read a 2D histogram of log D_hat against z column by column: each D
column should look N(0,1) if the tracks are Brownian, so a shifted
per-column mean (standard error ≈ 1/sqrt(n_column)) is evidence at the
population level. Count unresolved tracks separately; they have no z.
