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
a posterior that has not died out by an edge is cut there, and its summary
moves with the edge. The workflow flags such tracks in `message` and in
`D_grid_edge` (`posterior.grid_edge`: "low", "high" or "both");
localization-limited, near-immobile
tracks reach the lower edge this way, since their data only bound D from
above. The default lower edge sits well below any localization floor so
that such tracks fall into a low tail, read as upper bounds, rather than into
a separate D = 0 class: immobile, tightly confined and jittering spots cannot
be told apart from their steps. No module-level grid exists to fall back on: the workflow reads the
grids from `GridPostOptions`, and the lower-level functions take them as
required arguments.

`posterior.summary` reports one point estimate, the posterior mean E[D]
(`D_post_mean_um2_s`), and the `(1-level)/2` and `(1+level)/2` quantiles
(`D_post_lo_um2_s`, `D_post_hi_um2_s`; `GridPostOptions.level` defaults to
0.9). Of a posterior cut by the lower edge, E[D] is the summary the edge
barely moves: in simulation (tracks of 5 frames at D = 0.002 um^2/s, 97% of
them cut), moving `D_min_um2_s` from 1e-5 to 1e-7 moves the median about 10x
and E[D] 1.6x. Its cost is that under the flat prior in ln D (a 1/D prior on
D) it reads high for short tracks, by about 1/(n - 2) for n frames (+31% at
5 frames, about x2 at 3) where the median is within a few percent -- so a
column of per-track E[D] is not to be averaged: an average over tracks
belongs to a population model (below). For a cut track, `D_grid_edge` says how
to read it: "low", as an upper bound; "high", as a lower bound. The interval
is equal-tailed, read directly off the CDF, not a multiple of a standard deviation (a
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

## Combining tracks: one shared D, or a population

`analyze_tracks(..., keep_likelihoods=True)` keeps each track's log-likelihood
of D on the grid (`GridLikelihoods.loglik_D`, normalized so that each row's
logsumexp is 0; under the flat prior that is also the track's posterior).
Every population read is built from those rows, and always the same way:
**tracks are combined by adding log-likelihoods.** For a population g of D
across tracks, a track's likelihood is its own averaged over g, and the
tracks multiply:

```
l(theta) = sum_i log int L_i(u) g_theta(u) du,      u = ln D
```

Three choices of g, beside the per-track posteriors, which need none:

| read | g | function | what it answers |
|---|---|---|---|
| per track (unpooled) | none: each track on its own | the per-track posterior (`fits`) | how fast is this track? |
| shared D (complete pooling) | delta(u - u0): one D for all | `fit_shared_D`, `shared_D_tracks` | the one D, if every track had it |
| log-normal population | N(mu, sigma) in ln D | `fit_lognormal`, `lognormal_tracks` | the population's median D and its spread |
| population of any shape | a smooth log density | `deconvolve`, `deconvolve_tracks` | how D is distributed, shape included |

Two things look like population reads and are not:

- **the average of the tracks' posteriors.**
  The sum is outside the log, so it is the likelihood of nothing; it is the
  first EM step of the deconvolution from the flat prior. A short track adds
  the flat prior's shape to it, so it describes the tracks, with their
  uncertainty, rather than estimating the population.
- **a histogram of per-track medians.** A wide posterior's median sits where
  the prior puts it (mid-grid, or at `D_min` for a localization-limited
  track), and the histogram's width is the true spread plus each track's
  estimation noise, with no way to tell them apart.

Complete pooling is the summed log-likelihood itself, and it is a correct
posterior for the model it assumes. When the tracks differ, its median lands
near the population's mean D, weighted by how much each track says about D
(in the study below, 0.070 um^2/s for a log-normal whose mean is 0.069): the
quantity an ensemble MSD reports. Its interval is the problem: it narrows as
1/sqrt(n) however much the tracks differ, as if every track had that D. Two
populations at 0.005 and 0.2 um^2/s give a shared D near 0.13 with an
interval of about +/-1%, a value neither population has. And because the
weighting is by information, mostly track length, it leans toward the long
tracks, which are often the slow particles that stay in focus. The
log-normal's sigma is the check: sigma = 0 is complete pooling, so when its
posterior reaches 0, one shared D describes the tracks.

The words are kept to these meanings throughout:

- **unpooled**: each track's own posterior under the flat prior (`fits`).
  The prior is flat in ln D on
  the grid, so a track's normalized likelihood row is also its posterior.
- **shared** is the one D of complete pooling (`fit_shared_D`,
  `shared_D_tracks`, `SharedD`, `GridPostBatch.shared_D`), and nothing else:
  every track is taken to have that D.
- a **population** is a distribution g of D across tracks (the log-normal,
  the deconvolution). Each track has its own D, a draw from g; g is fitted
  with each track's D integrated out (the l(theta) above), and g is what is
  reported. A population is not a shared D or a shared posterior.
- tracks of several movies are **combined** into one fit, whatever the model.

Not computed: each track's posterior with the fitted g as its prior
(partial pooling, i.e. empirical Bayes). The same tracks would set the prior
and then be judged by it, so a track's number would move with which other
tracks are in the fit and with the population model. Per-track numbers stay
each track's own (`fits`), and questions about the population are answered
by g.

Outside this section "shared" keeps its plain meaning (drift shared by a
movie's tracks), and `classic` does not use "pooled": its ensemble MSD is a
weighted average of the tracks' curves, not a model of the population.

## Log-normal population: `lognormal`

`lognormal_tracks(analysis)` (or `fit_lognormal(lls, u)` on any evenly spaced
grid of rows) is a population with two parameters: ln D ~ N(mu, sigma)
across tracks, truncated to the grid. exp(mu) is the population's median D,
sigma its spread in ln D, and exp(mu + sigma^2/2) its mean D; each track's own
measurement noise is in the model, not in sigma. Priors are flat in mu over
the grid and flat in sigma on [0, `sigma_max`] (default 3, a 90% range of
four decades); a posterior that reaches a bound says so in `problem`.

Between grid points a likelihood row is taken as linear in L (a sum of hat
functions), so its integral against a normal has a closed form
(`hat_weights`): exact for every sigma, sigma = 0 included, smooth in mu, and
with tail cells from survival functions, so a track far from mu contributes
its tiny likelihood rather than a rounding floor. For a set of (mu, sigma) the
whole computation is one matrix product, rows x weights. The posterior is
evaluated on a 61 x 41 grid that zooms in on where the posterior lives:
starting from the whole prior, it is refitted to the cells within 20 nats of
the peak until they fill at least a third of it, so its resolution follows
the posterior's width (about 0.15 s for 1000 tracks and 0.3 s for 5000 on
a laptop). As with the deconvolution, a track with a flat likelihood
leaves the result unchanged.

The result is a `GridDistribution` like a deconvolution: `weights` (g at the
posterior mode), `samples` (g at posterior draws of (mu, sigma), kept in
`draws`), `band`, `mass`, `cdf_distance` and `viz.plot_populations` all
take it. `summary(level)` gives the median D,
sigma and the mean D with equal-tailed intervals. When the truth is sigma = 0
the posterior piles against that bound, so its equal-tailed interval never
contains 0: read sigma's upper bound there.

`scripts/validate_lognormal.py` uses the deconvolution study's simulator and
settings (1000 tracks of 5-20 frames, dt 35 ms, localization SD 30-45 nm, the
default grid), 40 datasets per population. Coverage of the 68% / 95%
intervals:

| population (ln D ~ N) | median D | sigma | mean D | estimate (median over datasets) |
|---|---|---|---|---|
| one D, 0.05 (sigma = 0) | 0.73 / 0.95 | upper bound 1.0 / 1.0 | 0.70 / 0.95 | 0.0498, sigma 0.054 |
| 0.05, sd .3 | 0.73 / 0.98 | 0.58 / 0.93 | 0.73 / 1.0 | 0.0503, sigma 0.31 |
| 0.05, sd .8 | 0.68 / 1.0 | 0.60 / 0.98 | 0.73 / 0.98 | 0.0499, sigma 0.81 |
| 0.003, sd .8 (below the floor) | 0.63 / 0.98 | 0.68 / 0.95 | 0.78 / 0.98 | 0.0030, sigma 0.81 |

With 40 datasets a coverage is known to about +/-0.07 at 68%. sigma's 68%
interval runs a little narrow (0.58-0.68). For one shared D the equal-tailed
interval of sigma never reaches 0 (it is 0.65 at 95%, 0 at 68%), while its
upper bound always covers 0, as it should.

When the population is not log-normal, the fit still runs and says nothing
about it: two modes (0.4 at 0.005, 0.6 at 0.2 um^2/s, each sd .3) come out as
one wide log-normal (sigma 1.9) whose W1 distance in ln D to the population is
0.79, against 0.09 for the deconvolution of the same tracks. A sigma that
large, or a deconvolution with more than one mode, means the log-normal's
numbers describe the wrong shape.

`fit_shared_D(lls, u)` sums the rows and interpolates the sum with a cubic
spline through the cells within 40 nats of its peak, which resolves it below
the grid step: with many tracks the shared D's posterior is narrower than a cell.

## Distribution of D across tracks: `deconvolve`

`deconvolve_tracks(analysis)` estimates how D is distributed across
the tracks of `analyze_tracks(..., keep_likelihoods=True)` -- a population-level comparator to an ensemble MSD fit,
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

`scripts/validate_deconvolve.py` simulates five populations -- a spike, two
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
unchanged. The fit reads the kept likelihood rows, and ignores their per-row
constant. `deconvolve.deconvolve(lls, u)` is the same fit on any evenly spaced
grid of per-track log-likelihood rows.

### Split by track length: `by_track_length`

Track length is observed, and it depends on D. A fast particle crosses the
focal depth in a few frames and makes short tracks; a slow one stays in focus
and makes one long track. `by_track_length(likelihoods, u)` splits the tracks
into a few length groups (`LENGTH_EDGES`, lower edges 3, 5, 10, 20 frames,
trimmed at the longest track) and deconvolves each group's population of D
from its own tracks' likelihood rows (`deconvolve`, above). Group b's
population g_b is stacked by its share of the weight, share_b g_b(D), so the
groups add up to a mixture population (`deconvolved`), with each group's
draws in `populations[b]` and its band in `band()`.

The stack is close to the whole table's deconvolution but not equal to it:
each group's evidence chooses its own smoothness. A group with few tracks is
mostly the smoothness prior; its band shows it. On a simulated mix of D 0.3
and 0.003 um^2/s whose fast share falls with length (200 tracks per group),
each group's fast share came out within 0.01 of the truth, and the fast mode's
spread in log10 D (truly 0) was 0.03-0.11, against 0.12-0.47 from averaging the
tracks' posteriors, which this replaces.

`weight="tracks"` gives each group its share of tracks. `weight="detections"` gives it its
share of frames, so the result is the composition of the spots seen; within a
group each track still counts once. The two differ when fast particles make many short
tracks. Per detection, the excluded tracks shorter than `min_frames`
(mostly single spots, likely fast) are missing, which pushes the slow share up.
`viz.plot_by_track_length` stacks the groups and shows each group's own
distribution, with the localization floor as a band.

## What is reported, and what is not

Each track gets D: an interval, its information in bits, and the localization
floor as a scale. Nothing else is reported.

The motion's shape (alpha) is not part of `gridpost`. On short tracks it is
weakly identified, and it confounds localization-SD errors with real caging.
An anomalous exponent is available from the MSD fits (`diffusionkit.classic`),
and from `diffusionkit.bayes` as a per-track diagnostic.

## Workflow: `gridpost.analyze_track`/`analyze_tracks`

`GridPostOptions(min_frames=3, level=.9)` sets the short-track exclusion
threshold (the whitening step's own hard minimum) and the credible-interval
mass; its grid fields (`D_min_um2_s`, `D_max_um2_s`, `n_D`) set the grid the
run evaluates. `analyze_tracks(..., keep_likelihoods=True)` returns each
`ok` track's normalized log-likelihood too (`GridLikelihoods`), for the
population reads above. `analyze_track`/`analyze_tracks` mirror `classic`'s workflow contract:
invalid input raises for a single track, a batch keeps going and marks the
offending track `invalid_input`, and `status="excluded"` records *why* a
track wasn't fit rather than silently dropping it. See
[TABLES.md](../TABLES.md) for the `fits` schema and [WORKFLOW.md](../WORKFLOW.md)
for worked examples.
