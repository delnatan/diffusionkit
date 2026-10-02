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
the prior it rules out. `D_floor_um2_s` (`posterior.localization_floor`) is the D at
which a displacement's motion variance, 2 D (dt - exposure/3) per axis,
equals its localization noise, `s_i^2 + s_(i+1)^2`: `<s^2> / (dt -
exposure/3)` over the track's own SDs. It is a scale to draw next to any D,
not a detection threshold -- the posterior already accounts for the noise
-- and it moves with the square of any error in the reported SDs. Calibration is a simulation check under the model (see
`scripts/validate_posterior.py`), not a claim about experimental tracks.

A previous production estimator (`fit_brownian_mle`, a
maximum-likelihood D with a bootstrap-calibrated non-Brownian z-score
testing deviation from alpha=1) has been retired now that the posterior
supersedes its point-estimate role.

## alpha: grid posterior over the fBm exponent, its scale marginalized out

`diffusionkit.gridpost.posterior_alpha` answers a different question from the D
posterior above: not "how big are the steps" (the alpha=1 model) but "how
does the spread grow with time". Per axis, the same m = n-1 displacements
are Gaussian under the fBm model:

```
Sigma(D, alpha) = D dt^(1-alpha) A(alpha) + B
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

The scale D is the apparent diffusivity at the frame interval,
MSD(dt) / (4 dt) = K dt^(alpha-1), not fBm's generalized coefficient K
(um^2/s^alpha). The two describe the same motion, but a grid and a flat prior
over ln K do not describe the same prior: K's unit depends on alpha, so a
fixed K range admits different motions at each alpha, and which ones depends
on whether time is in seconds or milliseconds. On a gelled-bead movie (dt 44
ms, D ~ 0.001 um^2/s at dt) the K grid's 1e-4 floor removed low-alpha
hypotheses, since there K = D dt^(1-alpha) falls below 1e-4 as alpha drops:
5-9 frame tracks, which carry ~0.06 bits about alpha, reported a median of
1.15 with the scale on K and 0.79 with it on D at dt; on 40-frame tracks the
two agreed. With D at dt, alpha's nuisance scale is the D posterior's own
grid and prior (`GridPostOptions.u_D`, flat in ln D), a grid edge means the
same thing for both posteriors, the alpha posterior is unchanged by a change
of time unit (tested), and at alpha=1 its likelihood is exactly the D
posterior's.

For any *fixed* alpha this is linear in D, like the D posterior's model, so
the same whitening trick applies -- but A(alpha) itself
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
S(w)  = D S_A(w; alpha) + S_B(w)
```

The *debiased* version (Sykulski, Olhede, Guillaumin, Lilly & Early 2019,
Biometrika 106:251) takes S to be the expected periodogram of the exact
finite-m covariance, not the process's spectral density: S_A is the FFT of
the tapered exact blurred autocovariance (1 - k/m) dt^(1-alpha) gamma(k), and S_B comes
from the per-frame localization SDs. That removes the leakage and aliasing
bias of plain Whittle, so the only approximation left is treating the
periodogram ordinates as independent. S is linear in D, so the D
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

The alpha posterior is the 1D marginal of the 2D (alpha, ln D) log-likelihood
surface, integrating the nuisance D out under the flat ln D prior via
`logsumexp` -- `posterior_alpha.summary` reports the median and credible
interval (`alpha_post_median`, `alpha_post_lo`, `alpha_post_hi`). D and alpha
are reported as two per-track measurements, not a joint (D, alpha) point:
`D` comes from the alpha=1 model, `alpha` from the fBm model with its scale
integrated out entirely, which sidesteps the scale/alpha MLE degeneracy
rather than inheriting it. `alpha_post_info_bits` is
`posterior.information_bits` against the flat alpha prior. A track with too
little motion above its localization noise, or too few frames, has a
near-flat alpha posterior whose median sits near the prior's centre (1.0 on
the default grid) whatever the motion is, so read alpha medians together
with their bits. `deconvolve.deconvolve` accepts these posteriors as
likelihood rows on the alpha grid (flat alpha prior); see its docstring for
what that distribution inherits from the scale's prior.

## Distribution of D across tracks: `deconvolve`

`deconvolve_tracks(table, acquisition)` estimates how D is distributed across
a table of tracks -- a population-level comparator to an ensemble MSD fit,
not a replacement for the per-track posteriors. It uses each track's
likelihood on the D grid, not its posterior or a point estimate, so the prior
is not counted once per track and a short track's uncertainty is not averaged
away. The grid weights g maximize `sum_i log sum_k L_ik g_k` under a Gaussian
smoothness prior on log g (the second-order penalty
`(lam/2) int (log g)''^2 du`, a logistic-Gaussian-process density); lam is
chosen by its Laplace evidence over `deconvolve.LAM_GRID`, not set by hand.
`samples` are posterior draws of the whole distribution, lam integrated out
by its evidence, from Hamiltonian Monte Carlo preconditioned by the Laplace
approximation. Plain Laplace draws are not used: where the data rule a grid
cell out, the Gaussian ignores the cliff in the posterior and puts mass in
empty cells. `band(level, cumulative)` gives pointwise or CDF bands; any mass
or mean has its interval in `samples @ a`.

`scripts/validate_deconvolve.py` (recorded in
`audit/deconvolve_validation.json`) simulates five populations -- a spike, two
narrow modes, a log-normal, two broad modes, and a mode below the
localization floor -- 40 datasets of 1000 tracks each, on the default grid:

| population | lam (median) | W1 in ln D | CDF coverage 68% / 95% |
|---|---|---|---|
| spike at 0.05 | 3e-5 | 0.058 | (CDF is 0 or 1) |
| 0.02 / 0.2, sd .15 | 1.5e-3 | 0.048 | 0.82 / 0.96 |
| log-normal 0.05, sd .8 | 0.28 | 0.042 | 0.72 / 0.96 |
| .3 at 0.01 / .7 at 0.1, sd .4 | 0.026 | 0.058 | 0.76 / 0.99 |
| .4 at 0.002 / .6 at 0.1 | 0.033 | 0.114 | 0.69 / 0.94 |

A mode narrower than the per-track resolution comes out as wide as that
resolution allows, so a peak's width is not a measurement. Below the
localization floor the bands widen, because the tracks cannot tell those D
values apart. For a spike the evidence may run to the rough end of
`LAM_GRID`, which warns. The same function serves the alpha grid
(`deconvolve.deconvolve(lls, alphas)`); a track with a flat likelihood leaves
the result unchanged.

## What is reported, and what is not

Under this model a track's information is its displacement covariance
relative to the localization noise -- equivalently MSD(tau) -- and that has
one well-measured direction and one weak one: its scale, D, and how it bends
with tau, alpha. Other per-track metrics are functions of the same
covariance. On the 2026-09-23 gelatin temperature series (198 nm beads,
liquid to gel), Spearman correlations over tracks show two groups: D with
the mean step (0.94 in the liquid); and alpha with a timescale ratio of D
(0.87), a Brownian-versus-noise likelihood ratio (0.64), the radius of
gyration (0.65) and straightness (0.55-0.77), the likelihood ratio in turn
0.92 with the radius of gyration and 0.00 with D. In the liquid, where every
bead's alpha is ~1, the second group's correlation is chance: a track that
happens to wander further raises all of them at once. They are one
fluctuation under several names, each inviting a different story
("confined", "trapped", "directed") that the data do not separate on short
tracks.

The package therefore reports D and alpha, each with an interval and its
information in bits, plus D's localization floor as a scale. Confinement
and subdiffusion are readings of low alpha, not separate outputs. A ratio of
D at two timescales (`gridpost.timescale`) and a Brownian-versus-noise
likelihood ratio (`D_motion_lrt`) were implemented and removed for this
reason; see the git history before this change.

## Workflow: `gridpost.analyze_track`/`analyze_tracks`

`GridPostOptions(min_frames=3, level=.9)` sets the short-track exclusion
threshold (the whitening step's own hard minimum) and the credible-interval
mass; its grid fields (`D_min_um2_s`, `D_max_um2_s`, `n_D`, `alpha_min`,
`alpha_max`, `n_alpha`; alpha's nuisance scale uses the D grid) set every
grid the run evaluates, and `compute_alpha=False` skips the
alpha posterior. `analyze_tracks(..., keep_posteriors=True)` returns each
`ok` track's normalized log posterior too (`GridPosteriors`), for population
reads such as `deconvolve.deconvolve`. `analyze_track`/`analyze_tracks` mirror `classic`'s workflow contract:
invalid input raises for a single track, a batch keeps going and marks the
offending track `invalid_input`, and `status="excluded"` records *why* a
track wasn't fit rather than silently dropping it. See
[TABLES.md](../TABLES.md) for the `fits` schema and [WORKFLOW.md](../WORKFLOW.md)
for worked examples.
