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
A(alpha)_k = G((k+1) dt) - 2 G(k dt) + G((k-1) dt)   (lag k)
G(x) = E|x + u - v|^alpha,   u, v ~ Uniform(0, exposure_s)
```

with B the same localization-noise covariance as the D posterior. The true
motion has structure function `E[(z(t+s) - z(t))^2] = 2 K |s|^alpha` per
axis, and each frame records its average over the exposure, so each
displacement's covariance is that structure function averaged over the two
frames' exposure offsets. `u - v` has the triangular density on
`[-te, te]`, which makes the average a closed form:

```
G(x) = (H(x + te) - 2 H(x) + H(x - te)) / te^2,   H(x) = |x|^(alpha+2) / ((alpha+1)(alpha+2))
```

(`likelihood.fgn_motion_covariance`). With `exposure_s=0`, `G(x) = |x|^alpha`
and A is plain fGn, `dt^alpha (|k+1|^alpha - 2|k|^alpha + |k-1|^alpha)`; at
alpha=1 A is the D posterior's Berglund matrix at any exposure, so the two
posteriors share one blur model. Where `te/|x| < 0.1` the difference above
cancels, and the binomial series of `|x|^alpha (1 + w/x)^alpha` over the
triangular moments `E[w^2k] = 2 te^2k / ((2k+1)(2k+2))` is used instead
(through `(te/x)^12`; the two branches agree to ~1e-12 at the switch). The
tests check the closed form against quadrature and against box-averaged,
exactly simulated fine-step fBm.

Blur matters for alpha. It correlates neighbouring displacements positively
(lag-1 correlation +0.12 for Brownian motion at 20 ms of a 33 ms frame),
which is what alpha is read from: fitted without it, simulated alpha=0.5
tracks come out near 0.7 and alpha=1 near 1.14.

For any *fixed* alpha this is linear in K exactly as the D posterior's model
is linear in D, so the same whitening trick applies -- but A(alpha) itself
changes shape with alpha, so each alpha grid point (`GridPostOptions.alphas()`,
39 points by default) needs its own eigendecomposition, unlike the D
posterior's single decomposition reused across the whole D grid. Everything
that does not depend on alpha (validation, B's Cholesky factor, the whitened
displacements) is done once per track, and the grid's eigendecompositions
run as one batched call. That is ~1 ms per track below 20 frames but
~100 ms at 100-200 frames, where the O(m^3) eigendecompositions dominate.

### Long tracks: the debiased Whittle likelihood

The motion covariance is Toeplitz, and Fourier vectors nearly diagonalize
a Toeplitz matrix. The Whittle likelihood uses them in place of the
eigenvectors: with each axis's periodogram I(w_j) = |FFT(d)_j|^2 / m,

```
log L = -1/2 sum_{axis, j} [ln S(w_j) + I(w_j) / S(w_j)] - m ln 2 pi
S(w)  = K S_A(w; alpha) + S_B(w)
```

The *debiased* version (Sykulski, Olhede, Guillaumin, Lilly & Early 2019,
Biometrika 106:251) takes S to be the expected periodogram of the exact
finite-m covariance, not the process's spectral density: S_A is the FFT of
the tapered exact blurred autocovariance (1 - k/m) gamma(k), and S_B comes
from the per-frame localization SDs. That removes the leakage and aliasing
bias of plain Whittle, so the only approximation left is treating the
periodogram ordinates as independent. S is linear in K, so the K
marginalization is unchanged, and each alpha costs one O(m log m) FFT
(`posterior_alpha._whittle_joint_loglik`).

What that approximation loses: Whittle sees the noise only through its
spectrum, not which frames were noisy. `scripts/validate_alpha_whittle.py`
(recorded in `audit/alpha_whittle_validation.json`) compares the two on
simulated blurred tracks. Whittle stays calibrated at every length (90%
coverage 0.83-1.00 across cells, the same scatter as exact's), but it is
less accurate, and not less so for shorter tracks only: the median RMSE
ratio to exact is 1.01-1.06 with constant localization SDs and 1.05-1.13
with per-frame SDs varying 0.015-0.05 um, at every length from 10 to 200
frames. Its speed advantage grows with length: 2.4x at 40 frames, ~10x at
200. `GridPostOptions(alpha_method="auto")`, the default, therefore uses the
exact likelihood below `alpha_whittle_min_frames` (40, the script's
recommendation) and Whittle from there on; "exact" and "whittle" force
one. Each alpha row's `method` says which produced it.

On the 539 bead tracks (20 ms exposure assumed), the hybrid runs the alpha
posteriors in 3.2 s against 13.4 s all-exact, and the per-track medians of
tracks >= 40 frames move by 0.02 (median) and 0.13 (90th percentile) in
alpha. `analyze_tracks(..., map_fn=pool.map)` spreads tracks over a
caller's thread pool (the linear algebra releases the GIL).

The alpha posterior is the 1D marginal of the 2D (alpha, ln K) log-likelihood
surface, integrating the nuisance K out with its own (hand-set, not
empirical-Bayes) prior via `logsumexp` -- `posterior_alpha.summary` reports
the median and credible interval (`alpha_post_median`, `alpha_post_lo`,
`alpha_post_hi`). D and alpha are reported as two independent per-track
measurements, not a joint (K, alpha) fit: `D` comes from the alpha=1 model,
`alpha` from the fBm model with K integrated out entirely, which sidesteps
the well-known K/alpha MLE degeneracy rather than inheriting it.

## D at two timescales: `gridpost.timescale`

`GridPostOptions(D_long_stride=k)` refits the D posterior at tau = k dt and
reports how D changes between the two timescales -- the apparent diffusivity
D(tau) = MSD(tau) / (4 tau), which is constant for Brownian motion. A ratio
below 1 says the motion slows at longer times (confinement, a crowded or
elastic surrounding, fBm with alpha < 1, where it is k^(alpha-1)); above 1,
that it is persistent. It needs no model of why, which is its use next to
the alpha posterior: alpha commits to fBm, and a confined particle is not
one. The gap k is the one choice it asks for -- the timescale being
compared with dt -- and the shortest track it can use is
`(min_frames - 1) * k + 1` frames.

D at tau is the D posterior of the track thinned to every k-th frame: the
same exact likelihood with frame interval k dt, the same exposure (so its
blur is Berglund's average at R = exposure / (6 k dt)), and each retained
frame's own localization SD. Thinning leaves k interleaved phases, which
share the underlying path and so are not independent: their
log-likelihoods are averaged (`timescale.thinned_loglik`, a composite
likelihood with weight 1/k per phase), so every frame counts toward the
estimate while the posterior keeps about one phase's width. The ratio's
posterior (`timescale.ratio_posterior`) combines the two D posteriors as if
independent; they are positively correlated, so this widens the ratio's
interval. Both choices err conservative.

`scripts/validate_D_timescale.py` (recorded in
`audit/D_timescale_validation.json`) checks them on tracks simulated at a
fine time step and averaged over a 20 ms exposure, with per-frame
localization SDs of 0.015-0.05 um, stride 5 (33 -> 165 ms), 300 tracks per
cell:

| | 20 frames | 50 frames | 100 frames |
| --- | --- | --- | --- |
| Brownian: ratio median (truth 1) | 1.08 | 1.01 | 1.01 |
| Brownian: ratio interval covers 1 | 0.99 | 0.99 | 1.00 |
| Brownian: P_D_decrease > 0.95 (false flags) | 0.01 | 0.00 | 0.00 |
| Brownian: D_long 90% coverage, averaged (one phase) | 0.98 (0.90) | 0.96 (0.91) | 0.96 (0.92) |
| Brownian: D_long RMSE in ln D, averaged (one phase) | 0.49 (0.66) | 0.28 (0.35) | 0.19 (0.24) |
| Trapped, L = 0.15 um: ratio median (noise-free 0.50) | 0.52 | 0.49 | 0.49 |
| Trapped, L = 0.15 um: P_D_decrease > 0.95 | 0.07 | 0.49 | 0.92 |
| fBm alpha = 0.6: ratio median (noise-free 0.53) | 0.58 | 0.59 | 0.57 |
| fBm alpha = 0.6: P_D_decrease > 0.95 | 0.04 | 0.30 | 0.79 |

Averaging the phases buys ~20-25% in accuracy over one phase at the same
width. The ratio tracks the noise-free value; per-track detection of a
slowdown needs long tracks, and the intervals' conservatism costs power
there. On the 539 bead tracks (20 ms exposure assumed) the ratio is 1.01
(median) at strides 3, 5 and 10, with P_D_decrease > 0.95 in 1-2% of
tracks, and its log correlates with the alpha posterior median at r = 0.8.

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
