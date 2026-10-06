# Grid posterior: exact-likelihood D, no MSD curve

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
(default 1e-5 to 10 um^2/s, 601 points), is therefore the prior's support:
a posterior that has not died out by an edge is cut there, and its median
and interval move with the edge. The workflow flags such tracks in
`message` (`posterior.edge_ratios`); localization-limited, near-immobile
tracks reach the lower edge this way, since their data only bound D from
above. The default lower edge sits well below any localization floor so
that such tracks fall into a low tail, read as upper bounds, rather than into
a separate D = 0 class: immobile, tightly confined and jittering spots cannot
be told apart from their steps. No module-level grid exists to fall back on: the workflow reads the
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

## Distribution of D across tracks: `deconvolve`

`deconvolve_tracks(analysis)` estimates how D is distributed across
the tracks of `analyze_tracks(..., keep_posteriors=True)` -- a population-level comparator to an ensemble MSD fit,
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
| spike at 0.05 | 4e-5 | 0.062 | (CDF is 0 or 1) |
| 0.02 / 0.2, sd .15 | 1.8e-3 | 0.063 | 0.69 / 0.89 |
| log-normal 0.05, sd .8 | 0.33 | 0.041 | 0.69 / 0.98 |
| .3 at 0.01 / .7 at 0.1, sd .4 | 0.026 | 0.064 | 0.71 / 0.96 |
| .4 at 0.002 / .6 at 0.1 | 0.036 | 0.109 | 0.73 / 0.97 |

A mode narrower than the per-track resolution comes out as wide as that
resolution allows, so a peak's width is not a measurement; the two narrow
modes are where the 95% band covers least. Below the
localization floor the bands widen, because the tracks cannot tell those D
values apart. For a spike the evidence may run to the rough end of
`LAM_GRID`, which warns. A track with a flat likelihood leaves the result
unchanged. The kept posteriors serve as the likelihood rows: their prior is
flat in ln D, so each row differs from the track's log-likelihood only by a
constant, which the fit ignores. `deconvolve.deconvolve(lls, u)` is the same fit
on any evenly spaced grid of per-track log-likelihood rows.

### Split by track length: `by_track_length`

Track length is observed, and it depends on D. A fast particle crosses the
focal depth in a few frames and makes short tracks; a slow one stays in focus
and makes one long track. `by_track_length(posteriors, u, population)` splits a
distribution of D into groups of track length (`LENGTH_EDGES`, lower edges 3, 4,
5, 7, 10, 15, 25, 50 frames, trimmed at the longest track). Each track i
contributes w_i p_i(D), and the contributions are summed per group and divided
by the total weight, so the groups add up to the whole distribution:

- `pooled`: p_i is the track's flat-prior posterior. This describes the tracks
  without a population model, and a short track spreads wide.
- `deconvolved`: p_i(D | g), proportional to L_i(D) g(D), the track's posterior
  with the population as its prior. It is computed once per draw of g, so the
  groups carry the population's uncertainty. This is partial pooling, and it is
  labelled as such.

`weight="tracks"` counts each track once, and the deconvolved groups then add
up to about g. `weight="detections"` counts each frame, so the result is the
composition of the spots seen. The two differ when fast particles make many short
tracks: in a 49-frame GEM movie (wt_2), D < 0.035 holds 52% of tracks and 67%
of detections. Per detection, the excluded tracks shorter than `min_frames`
(mostly single spots, likely fast) are missing, which pushes the slow share up.
`viz.plot_by_track_length` stacks the groups and shows each group's own
distribution, with the localization floor as a band.

## What is reported, and what is not

Each track gets D: an interval, its information in bits, and the localization
floor as a scale. Nothing else is reported.

The motion's shape (alpha) is not part of `gridpost`. Earlier versions had
three pieces:
- a grid posterior over alpha with the scale marginalized;
- each track's joint (alpha, D) posterior, written to file;
- a 2D deconvolution of that joint posterior.

Across the prototypes, alpha did not survive as a per-track quantity:
- On short tracks it is weakly identified.
- It leans low for immobile tracks (prior volume).
- It confounds localization-SD errors with real caging.
- Its population read over-counts long-lived low-alpha tracks.

They were removed after commit 540ba4a; see the git history. An anomalous
exponent is still available from the MSD fits (`diffusionkit.classic`), and
from `diffusionkit.bayes` as a per-track diagnostic.

Other per-track shape metrics (a ratio of D at two timescales, a
Brownian-versus-noise likelihood ratio, radius of gyration, straightness) were
removed earlier for the same reason. On short tracks they are one fluctuation
of the displacement covariance under several names.

## Workflow: `gridpost.analyze_track`/`analyze_tracks`

`GridPostOptions(min_frames=3, level=.9)` sets the short-track exclusion
threshold (the whitening step's own hard minimum) and the credible-interval
mass; its grid fields (`D_min_um2_s`, `D_max_um2_s`, `n_D`) set the grid the
run evaluates. `analyze_tracks(..., keep_posteriors=True)` returns each
`ok` track's normalized log posterior too (`GridPosteriors`), for population
reads such as `deconvolve.deconvolve`. `analyze_track`/`analyze_tracks` mirror `classic`'s workflow contract:
invalid input raises for a single track, a batch keeps going and marks the
offending track `invalid_input`, and `status="excluded"` records *why* a
track wasn't fit rather than silently dropping it. See
[TABLES.md](../TABLES.md) for the `fits` schema and [WORKFLOW.md](../WORKFLOW.md)
for worked examples.
