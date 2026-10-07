# Classical and grid-posterior workflows

Use `diffusionkit.classic.analyze_track`/`analyze_tracks` for MSD fits and
`diffusionkit.gridpost.analyze_track`/`analyze_tracks` for the D
grid posterior -- two independent entry points, each returning its own
`fits` table; join on `track_id` if you want both. `diffusionkit.bayes.fit_track`
is a separate, per-track diagnostic tool (full NUTS posterior via NumPyro),
not part of either bulk table; pass it the same `exposure_s` as the grid
posteriors' `Acquisition`, so both fit the blurred motion.

## One track, a DataFrame

```python
import numpy as np
import polars as pl
from diffusionkit import Acquisition
from diffusionkit.classic import MSDOptions, analyze_track

track = pl.DataFrame({
    "track_id": [42]*5, "frame": np.arange(5),
    "x_um": [0, .02, .01, .04, .03], "y_um": [0, .01, .04, .03, .06],
    "sigma_x_um": [.01]*5, "sigma_y_um": [.01]*5,
})
result = analyze_track(track, Acquisition(dt_s=.033), MSDOptions(max_lag=3))
print(result.brownian.parameters, result.brownian.status)
print(result.anomalous.parameters, result.anomalous.status)
```

A track is just rows of a `polars.DataFrame`; there is no intermediate
per-track object. Functions validate it (`diffusionkit.validated_track_frame`
is the shared boundary) and return new data. Single-track invalid inputs
raise `ValueError`; a valid but too-short track returns `excluded` fits. A
result does not imply statistical identifiability just because the optimizer
converged.

## An existing physical-unit table

```python
import polars as pl
from diffusionkit import Acquisition
from diffusionkit.classic import analyze_track, analyze_tracks

acquisition = Acquisition(dt_s=.033)
# `tracks` is a Polars table, already in micrometers.
one = tracks.filter(pl.col("track_id") == 42)
selected = analyze_track(one, acquisition)
all_results = analyze_tracks(tracks, acquisition)
```

Required columns are
`track_id`, `frame`, `x_um`, `y_um`; `sigma_x_um` and `sigma_y_um` are also
required by the default correction. Their units are micrometers, not pixels.
If `t_s` or `track_length` is present, it must agree with `frame * dt_s` or
the actual row count. Those columns are not otherwise required.

Batch analysis keeps two fit rows per track, including invalid or excluded
tracks. A malformed table schema raises before fitting. An empty input with
valid column types returns typed empty result tables. There is no minimum
population size. Per-track results are not ensemble results: for replicates
and samples, wrap each movie in an `Experiment` and use the batch functions
below.

## The D posterior

```python
from diffusionkit.gridpost import posterior as P

s = P.track_posterior(one, Acquisition(dt_s=.033, exposure_s=.03))
s["mean"], s["lo"], s["hi"]     # D_post_mean/lo/hi_um2_s: E[D] and the interval, flat prior by default
```

Set `exposure_s` to the camera exposure; the posterior models the blur. This is the field also attached to
`gridpost.analyze_track`/`analyze_tracks`' output as `result.posterior_D`
(`model="posterior_D"` rows in the bulk `fits` table) -- a track's own
uncertainty stays visible as how wide its interval is, rather than being
collapsed to a point estimate. A short or noise-dominated track producing a
wide interval is expected, not a failure.

`analyze_tracks` fits tracks one after another by default. For a large table pass a thread pool's
`map`; the linear algebra releases the GIL, and the result is the serial one:

```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(8) as pool:
    result = analyze_tracks(tracks, acquisition, map_fn=pool.map)
```

## The distribution of D across tracks

```python
import numpy as np
from diffusionkit.gridpost import analyze_tracks, by_track_length, deconvolve_tracks, lognormal_tracks

result = analyze_tracks(tracks, Acquisition(dt_s=.033, exposure_s=.03), keep_likelihoods=True)
logn = lognormal_tracks(result)       # ln D ~ N(mu, sigma) across tracks, from the kept likelihoods
logn.summary(.9)["sigma_ln_D"]        # the spread; near 0: one shared D describes the tracks
pop = deconvolve_tracks(result)       # any shape, from the same likelihoods
pop.u, pop.weights                    # ln D grid, distribution (sums to 1)
lo, hi = pop.band(.68)                # pointwise band on the weights
lo, hi = pop.band(.95, cumulative=True)
above = pop.samples[:, pop.u > np.log(.05)].sum(axis=1)  # any mass: read its interval off the draws
```

A population-level comparator to an ensemble MSD fit, built from each
track's likelihood rather than its point estimate. Its smoothness is chosen
by the data, not set by hand; a peak narrower than the tracks can resolve
comes out as wide as that resolution, and below the localization floor the
bands widen because the tracks cannot tell those D values apart.

### Split by track length

```python
comp = by_track_length(result.likelihoods, result.options.u_D(), pop, weight="detections")
comp.labels(), comp.n_tracks, comp.n_detections   # the groups: '3', '5-6', '25+', ...
comp.unpooled                # (groups, grid): flat-prior posteriors summed per group
comp.partially_pooled        # (draws, groups, grid): each track's posterior under a draw of pop
comp.partially_pooled[:, :, comp.u < np.log(.035)].sum(2)   # mass below 0.035 per group, one row per draw

from diffusionkit.gridpost.viz import plot_by_track_length   # needs the plots extra
fig = plot_by_track_length(comp, result.fits["D_floor_um2_s"].drop_nulls().to_numpy())
```

Fast particles leave the focal depth within a few frames, so short tracks come
mostly from fast particles and long tracks from slow ones. The groups add up to
the whole distribution. With `weight="tracks"` they are fractions of tracks, and
the partially pooled groups add up to about the population. With
`weight="detections"` each track counts once per frame, which gives the
composition of the spots seen in focus.

## Inspect or change the classical analysis

```python
from diffusionkit.classic import compute_msd, fit_brownian_msd, fit_anomalous_msd

curve = compute_msd(one, acquisition)
D_fit = fit_brownian_msd(curve)
alpha_fit = fit_anomalous_msd(curve)
```

Change `MSDOptions(max_lag=...)` to inspect dependence on lag selection.
Changing the window changes the estimator; it is recorded in the result.
To deliberately omit localization correction, use
`MSDOptions(localization="ignore")`. There is no fallback from missing
measurement errors to uncorrected fitting.

For a GUI, pass `progress(done, total)` to either `analyze_tracks`. It is
invoked in the calling thread. Start a GUI-managed worker outside this
library. Join fit rows on both `track_id` and `model`, and keep
`status`/`message` visible. `status="ok"` is a numerical result status, not a
motion class. Persist `result.acquisition` and `result.options` alongside
exported tables (e.g. `dataclasses.asdict`); the tables alone are not a
complete run record.

## A batch of experiments

```python
from diffusionkit import Acquisition, Experiment, classic, gridpost

acq = Acquisition(dt_s=.02, exposure_s=.01)
experiments = [Experiment(name, table, acq, sample=sample)
               for name, sample, table in movies]          # track_id need only be unique within a movie

batch = gridpost.analyze_experiments(experiments)           # labelled fits + stacked likelihoods
batch.fits.group_by("sample").agg(pl.col("D_grid_edge").is_not_null().sum())   # per-track table, grouped any way
pops = batch.populations("sample")                          # D distribution per sample, with bands
reps = batch.populations("experiment")                      # per replicate

cbatch = classic.analyze_experiments(experiments)           # needs exposure_s = 0 (else no MSD rows)
ens = classic.ensemble_msd(cbatch, by="sample")             # ensemble MSD vs lag, with bootstrap resamples

# textbook route, no SD columns: the intercept of the linear MSD fit is the localization offset (4 sigma^2),
# subtracted before the log-log fit for alpha and K; the window is always chosen by you
opts = classic.MSDOptions(max_lag=None, lag_fraction=.4, localization="ignore")   # per-track window: 40% of each curve
ens = classic.ensemble_msd(classic.analyze_experiments(experiments, opts), by="sample")
ens.fit(5)                                                  # first 5 lags; ens.fit(n_points=...) is required
ens.fit(5, offset="provided")                               # supplied SDs instead of the intercept
```

Steps are separable: `analyze_experiments` is the expensive part, and what
you do after it (groupings, populations, comparisons, plots) only reads its
result, so a different grouping never refits a track. Save `batch.fits`,
`batch.options` and `batch.acquisitions` together; the kept likelihoods
(`batch.likelihoods`) are the large part and can be recomputed.

To compare samples, draw each population's mass over a range with
`pop.mass(lo, hi)` and difference the draws, or use
`gridpost.cdf_distance` against the replicate-to-replicate distances. The
batch `fits` table's `D_post_mean_um2_s` is a per-track summary; a histogram
or average of it is not the population (a short track's value leans on the
flat prior), which is what `populations` is for. For a per-track value that
borrows from the population, `pops["wt"].partially_pooled_means(sel.likelihoods.loglik_D)`.
